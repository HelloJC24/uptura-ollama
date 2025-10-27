from flask import Flask, request, jsonify, Response
from ollama import Client
from functools import lru_cache
import threading
import json

app = Flask(__name__)

# -------------------------
# Config
# -------------------------
OLLAMA_URL = "http://72.60.43.106:11434"
MODEL_DEFAULT = "mistral"

# Keep Ollama client global
ollama = Client(host=OLLAMA_URL)

# -------------------------
# Warmup
# -------------------------
def warmup_model():
    try:
        print("[INFO] Warming up the model...")
        # Dummy message to load the model in memory
        ollama.chat(model=MODEL_DEFAULT, messages=[{"role": "system", "content": "warmup"}])
        print("[INFO] Model warmed up!")
    except Exception as e:
        print(f"[WARN] Model warmup failed: {e}")

# Run warmup in a separate thread so Flask starts immediately
threading.Thread(target=warmup_model).start()

# -------------------------
# In-memory caching
# -------------------------
@lru_cache(maxsize=256)
def get_cached_response(query, model):
    response = ollama.chat(model=model, messages=[{"role": "user", "content": query}])
    return response['message']['content']

# -------------------------
# Streaming helper
# -------------------------
def stream_response(query, model):
    """
    Stream the response chunk by chunk using server-sent events (SSE)
    """
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": query}], stream=True)
        for chunk in response:
            # Each chunk is a dictionary with message content
            yield f"data: {chunk['message']['content']}\n\n"
    except Exception as e:
        yield f"data: [ERROR] {e}\n\n"

# -------------------------
# Flask endpoints
# -------------------------
@app.route('/ask', methods=['POST'])
def ask_model():
    data = request.get_json()
    query = data.get('query', '')
    model = data.get('model', MODEL_DEFAULT)

    if not query:
        return jsonify({"error": "Missing query"}), 400

    # Check if query is cached
    try:
        answer = get_cached_response(query, model)
        return jsonify({"answer": answer})
    except KeyError:
        # Not cached → stream response
        return Response(stream_response(query, model), mimetype="text/event-stream")

# -------------------------
# Start Flask
# -------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
