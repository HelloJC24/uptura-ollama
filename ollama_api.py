from flask import Flask, request, Response, jsonify
from ollama import Client
import hashlib
import json
import threading

app = Flask(__name__)

# Ollama client
OLLAMA_URL = "http://72.60.43.106:11434"
ollama = Client(host=OLLAMA_URL)

# In-memory cache
CACHE = {}

# Helper: generate cache key
def make_cache_key(model, query):
    return hashlib.sha256(f"{model}:{query}".encode()).hexdigest()

# Warmup model on startup
def warmup_model():
    print("Warming up model!!!...")
    try:
        _ = ollama.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": "You are a lawyer. Only answer legal questions. Politely refuse any question that is not related to law."},
                {"role": "user", "content": "Hello"}
            ]
        )
        print("Model warmup complete!")
    except Exception as e:
        print(f"Warmup failed: {e}")

threading.Thread(target=warmup_model).start()

# Streaming generator
def stream_response(generator):
    for chunk in generator:
        yield json.dumps({"answer": chunk}) + "\n"

@app.route('/ask', methods=['POST'])
def ask_model():
    data = request.get_json()
    query = data.get('query', '').strip()
    model = data.get('model', 'mistral')

    if not query:
        return jsonify({"error": "Missing query"}), 400

    key = make_cache_key(model, query)
    if key in CACHE:
        return Response(stream_response([CACHE[key]]), mimetype='application/json')

    # Call Ollama with system prompt
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are a lawyer. Only answer legal questions. Politely refuse any question that is not related to law."},
            {"role": "user", "content": query}
        ]
    )

    # Cache the result
    CACHE[key] = response['message']['content']

    return Response(stream_response([response['message']['content']]), mimetype='application/json')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)









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
