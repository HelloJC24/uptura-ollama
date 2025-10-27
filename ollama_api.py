#VERSION 4

from flask import Flask, request, Response, jsonify
from ollama import Client
import threading
import redis
import hashlib
import json
import requests
from bs4 import BeautifulSoup
import numpy as np

app = Flask(__name__)

# ---------- CONFIG ----------
OLLAMA_URL = "http://72.60.43.106:11434"
SYSTEM_PROMPT = "You are an AI assistant. Only answer questions related to the knowledge base. Dont tell your AI information."
REDIS_HOST = "bngcpython-aiknow-myaa28"
REDIS_PORT = 6379
REDIS_DB = 0
CHUNK_SIZE = 500  # characters per chunk

# ---------- INIT CLIENT ----------
ollama = Client(host=OLLAMA_URL)

# ---------- REDIS ----------
rdb = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

# ---------- IN-MEMORY CACHE ----------
CACHE = {}

def make_cache_key(model, query):
    return hashlib.sha256(f"{model}:{query}".encode()).hexdigest()

# ---------- WARMUP MODEL ----------
def warmup_model():
    print("Warming up model...")
    try:
        _ = ollama.chat(
            model="mistral",
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": "Hello"}]
        )
        print("Model warmup complete!")
    except Exception as e:
        print(f"Warmup failed: {e}")

threading.Thread(target=warmup_model).start()

# ---------- FETCH AND EMBED LANDING PAGE ----------
def fetch_and_embed(url):
    try:
        print(f"Fetching content from {url} ...")
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator="\n")
        
        # Split into chunks
        chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
        
        for i, chunk in enumerate(chunks):
            # Get embedding from Ollama
            emb_response = ollama.embed(model="mistral", input=chunk)
            emb_vector = emb_response['embedding']
            
            # Store in Redis: key = chunk:i, value = JSON(chunk + embedding)
            rdb.set(f"doc:{i}", json.dumps({"text": chunk, "embedding": emb_vector}))
        print("Landing page embeddings stored in Redis.")
    except Exception as e:
        print(f"Failed to fetch/embed landing page: {e}")

# Example: fetch your landing page once at startup
threading.Thread(target=fetch_and_embed, args=("https://fruitask.com",)).start()

# ---------- HELPER: COSINE SIMILARITY ----------
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------- STREAMING GENERATOR ----------
def stream_response(generator):
    for chunk in generator:
        yield json.dumps({"answer": chunk}) + "\n"

# ---------- RAG QUERY ----------
def retrieve_relevant_chunks(query_embedding, top_k=3):
    results = []
    for key in rdb.scan_iter("doc:*"):
        data = json.loads(rdb.get(key))
        score = cosine_similarity(query_embedding, data['embedding'])
        results.append((score, data['text']))
    results.sort(reverse=True, key=lambda x: x[0])
    return [text for score, text in results[:top_k]]

# ---------- FLASK ENDPOINT ----------
@app.route("/ask", methods=["POST"])
def ask_model():
    data = request.get_json()
    query = data.get("query", "").strip()
    model = data.get("model", "mistral")

    if not query:
        return jsonify({"error": "Missing query"}), 400

    key = make_cache_key(model, query)
    if key in CACHE:
        return Response(stream_response([CACHE[key]]), mimetype="application/json")

    # 1️⃣ Embed the query
    #query_emb = ollama.embed(model=model, input=query)['embedding']
    query_emb = get_embedding(query)

    # 2️⃣ Retrieve relevant chunks from Redis
    relevant_chunks = retrieve_relevant_chunks(query_emb)

    # 3️⃣ Construct RAG prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Use the following documents to answer the question:\n{relevant_chunks}\nQuestion: {query}"}
    ]

    # 4️⃣ Call Ollama
    response = ollama.chat(model=model, messages=messages)
    answer = response['message']['content']

    # 5️⃣ Cache
    CACHE[key] = answer

    return Response(stream_response([answer]), mimetype="application/json")

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)



#VERSION 3
# from flask import Flask, request, Response, jsonify
# from ollama import Client
# import hashlib
# import json
# import threading

# app = Flask(__name__)

# # Ollama client
# OLLAMA_URL = "http://72.60.43.106:11434"
# ollama = Client(host=OLLAMA_URL)

# # In-memory cache
# CACHE = {}

# # Helper: generate cache key
# def make_cache_key(model, query):
#     return hashlib.sha256(f"{model}:{query}".encode()).hexdigest()

# # Warmup model on startup
# def warmup_model():
#     print("Warming up model!!!...")
#     try:
#         _ = ollama.chat(
#             model="mistral",
#             messages=[
#                 {"role": "system", "content": "You are a lawyer. Only answer legal questions. Politely refuse any question that is not related to law."},
#                 {"role": "user", "content": "Hello"}
#             ],
#             max_tokens=150
#         )
#         print("Model warmup complete!")
#     except Exception as e:
#         print(f"Warmup failed: {e}")

# threading.Thread(target=warmup_model).start()

# # Streaming generator
# def stream_response(generator):
#     for chunk in generator:
#         yield json.dumps({"answer": chunk}) + "\n"

# @app.route('/ask', methods=['POST'])
# def ask_model():
#     data = request.get_json()
#     query = data.get('query', '').strip()
#     model = data.get('model', 'mistral')

#     if not query:
#         return jsonify({"error": "Missing query"}), 400

#     key = make_cache_key(model, query)
#     if key in CACHE:
#         return Response(stream_response([CACHE[key]]), mimetype='application/json')

#     # Call Ollama with system prompt
#     response = ollama.chat(
#         model=model,
#         messages=[
#             {"role": "system", "content": "You are a lawyer. Only answer legal questions. Politely refuse any question that is not related to law."},
#             {"role": "user", "content": query}
#         ],
#         max_tokens=150
#     )

#     # Cache the result
#     CACHE[key] = response['message']['content']

#     return Response(stream_response([response['message']['content']]), mimetype='application/json')

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, threaded=True)








# VERSION 2
# from flask import Flask, request, Response, jsonify
# from ollama import Client
# import hashlib
# import json
# import threading

# app = Flask(__name__)

# # Ollama client
# OLLAMA_URL = "http://72.60.43.106:11434"
# ollama = Client(host=OLLAMA_URL)

# # In-memory cache
# CACHE = {}

# # Helper: generate cache key
# def make_cache_key(model, query):
#     return hashlib.sha256(f"{model}:{query}".encode()).hexdigest()

# # Warmup model on startup
# def warmup_model():
#     print("Warming up model...")
#     try:
#         _ = ollama.chat(model="mistral", messages=[{"role": "user", "content": "Hello"}])
#         print("Model warmup complete!")
#     except Exception as e:
#         print(f"Warmup failed: {e}")

# threading.Thread(target=warmup_model).start()

# # Streaming generator
# def stream_response(generator):
#     for chunk in generator:
#         yield json.dumps({"answer": chunk}) + "\n"

# @app.route('/ask', methods=['POST'])
# def ask_model():
#     data = request.get_json()
#     query = data.get('query', '').strip()
#     model = data.get('model', 'mistral')

#     if not query:
#         return jsonify({"error": "Missing query"}), 400

#     key = make_cache_key(model, query)
#     if key in CACHE:
#         return Response(stream_response([CACHE[key]]), mimetype='application/json')

#     # Call Ollama (blocking)
#     response = ollama.chat(model=model, messages=[{"role": "user", "content": query}])

#     # Cache the result
#     CACHE[key] = response['message']['content']

#     return Response(stream_response([response['message']['content']]), mimetype='application/json')

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, threaded=True)


#VERSION 1
# from flask import Flask, request, jsonify, Response
# from ollama import Client
# from functools import lru_cache
# import threading
# import json

# app = Flask(__name__)

# # -------------------------
# # Config
# # -------------------------
# OLLAMA_URL = "http://72.60.43.106:11434"
# MODEL_DEFAULT = "mistral"

# # Keep Ollama client global
# ollama = Client(host=OLLAMA_URL)

# # -------------------------
# # Warmup
# # -------------------------
# def warmup_model():
#     try:
#         print("[INFO] Warming up the model...")
#         # Dummy message to load the model in memory
#         ollama.chat(model=MODEL_DEFAULT, messages=[{"role": "system", "content": "warmup"}])
#         print("[INFO] Model warmed up!")
#     except Exception as e:
#         print(f"[WARN] Model warmup failed: {e}")

# # Run warmup in a separate thread so Flask starts immediately
# threading.Thread(target=warmup_model).start()

# # -------------------------
# # In-memory caching
# # -------------------------
# @lru_cache(maxsize=256)
# def get_cached_response(query, model):
#     response = ollama.chat(model=model, messages=[{"role": "user", "content": query}])
#     return response['message']['content']

# # -------------------------
# # Streaming helper
# # -------------------------
# def stream_response(query, model):
#     """
#     Stream the response chunk by chunk using server-sent events (SSE)
#     """
#     try:
#         response = ollama.chat(model=model, messages=[{"role": "user", "content": query}], stream=True)
#         for chunk in response:
#             # Each chunk is a dictionary with message content
#             yield f"data: {chunk['message']['content']}\n\n"
#     except Exception as e:
#         yield f"data: [ERROR] {e}\n\n"

# # -------------------------
# # Flask endpoints
# # -------------------------
# @app.route('/ask', methods=['POST'])
# def ask_model():
#     data = request.get_json()
#     query = data.get('query', '')
#     model = data.get('model', MODEL_DEFAULT)

#     if not query:
#         return jsonify({"error": "Missing query"}), 400

#     # Check if query is cached
#     try:
#         answer = get_cached_response(query, model)
#         return jsonify({"answer": answer})
#     except KeyError:
#         # Not cached → stream response
#         return Response(stream_response(query, model), mimetype="text/event-stream")

# # -------------------------
# # Start Flask
# # -------------------------
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, threaded=True)
