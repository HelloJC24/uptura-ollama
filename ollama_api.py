#VERSION 4
from flask import Flask, request, Response, jsonify
from sentence_transformers import SentenceTransformer
import redis
import hashlib
import json
import threading
import requests
from bs4 import BeautifulSoup
import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from ollama import Client

# ----------------- Config -----------------
APP_NAME = "flask_rag_api"
MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_HOST = "http://72.60.43.106:11434"  # Your Ollama host
OLLAMA_MODEL = "phi3:mini"
REDIS_HOST = "bngcpython-aiknow-myaa28"
REDIS_PORT = 6379
# SYSTEM_PROMPT = "You are a legal assistant. Only answer questions related to your knowledge based. Ignore unrelated queries."
SYSTEM_PROMPT = """
You are a legal assistant. Only answer questions based on the information provided in the retrieved documents.
If the answer is not in the documents, respond with: "I’m sorry, I don’t have enough information to answer that."
Do not generate answers from your own knowledge or external sources.
"""
DOCUMENT_URLS = [
    "https://thebngc.com",
    "https://gogel.thebngc.com",
    "https://uptura-tech.com"
]  # Add as many landing pages as needed
TOP_K = 3  # Number of most relevant document chunks to use
CACHE = {}

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------- Flask -----------------
app = Flask(APP_NAME)

# ----------------- Redis -----------------
try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, password="987654321")
    r.ping()
    logging.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    logging.error(f"Failed to connect to Redis: {e}")
    r = None

# ----------------- Embeddings -----------------
embed_model = SentenceTransformer(MODEL_NAME)

def get_embedding(text):
    return embed_model.encode(text).tolist()

# ----------------- Ollama Client -----------------
ollama = Client(host=OLLAMA_HOST)

# ----------------- Document Fetch & Embed -----------------
DOC_CHUNKS = []

def fetch_and_store_documents():
    for url in DOCUMENT_URLS:
        logging.info(f"Fetching content from {url} ...")
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)

            # Split into chunks (~200 words each)
            words = text.split()
            chunk_size = 200
            chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

            for i, chunk in enumerate(chunks):
                emb = get_embedding(chunk)
                key = f"doc_chunk:{url}:{i}"
                if r:
                    r.set(key, json.dumps({"text": chunk, "embedding": emb}))
                DOC_CHUNKS.append({"text": chunk, "embedding": emb})
            logging.info(f"Stored {len(chunks)} chunks from {url} in Redis.")
        except Exception as e:
            logging.error(f"Failed to fetch document {url}: {e}")

threading.Thread(target=fetch_and_store_documents, daemon=True).start()

# ----------------- Warmup -----------------
def warmup_model():
    logging.info("Warming up model...")
    try:
        _ = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": "Hello, legal assistant!"}])
        logging.info("Ollama model warmup complete!")
    except Exception as e:
        logging.error(f"Warmup failed: {e}")

threading.Thread(target=warmup_model, daemon=True).start()

# ----------------- Helper Functions -----------------
def make_cache_key(query):
    return hashlib.sha256(query.encode()).hexdigest()

# def retrieve_relevant_chunks(query_emb, top_k=TOP_K):
#     similarities = []
#     for chunk in DOC_CHUNKS:
#         sim = cosine_similarity([query_emb], [chunk["embedding"]])[0][0]
#         similarities.append(sim)
#     top_indices = np.argsort(similarities)[-top_k:][::-1]
#     return [DOC_CHUNKS[i]["text"] for i in top_indices]

def retrieve_relevant_chunks(query_emb, top_k=TOP_K, min_sim=0.5):
    similarities = []
    for chunk in DOC_CHUNKS:
        sim = cosine_similarity([query_emb], [chunk["embedding"]])[0][0]
        similarities.append(sim)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Only include chunks above similarity threshold
    return [DOC_CHUNKS[i]["text"] for i in top_indices if similarities[i] >= min_sim]


def stream_response(generator):
    for chunk in generator:
        yield json.dumps({"answer": chunk}) + "\n"

# ----------------- Flask Endpoint -----------------
@app.route("/ask", methods=["POST"])
def ask_model():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Missing query"}), 400

    key = make_cache_key(query)
    if key in CACHE:
        return Response(stream_response([CACHE[key]]), mimetype="application/json")
        
    # Embed query
    query_emb = get_embedding(query)

    # Retrieve relevant document chunks
    relevant_docs = retrieve_relevant_chunks(query_emb)
    if not relevant_docs or all(len(doc.strip()) == 0 for doc in relevant_docs):
        answer = "I’m sorry, I don’t have enough information to answer that."
    else:
        prompt = SYSTEM_PROMPT + "\n\n"
        prompt += "\n---\n".join(relevant_docs)
        prompt += f"\n\nUser: {query}\nAnswer:"

    # Call Ollama
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "system", "content": prompt}], MAX_TOKENS=100)
        answer = response['message']['content']
    except Exception as e:
        logging.error(f"Ollama call failed: {e}")
        answer = "Error: Failed to generate response."

    # Cache result
    CACHE[key] = answer
    return Response(stream_response([answer]), mimetype="application/json")

# ----------------- Run App -----------------
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
