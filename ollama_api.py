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
    "https://uptura-tech.com",
    "https://gogel.thebngc.com/agents",
    "https://thebngc.com/privacy-policy",
    "https://thebngc.com/terms-conditions"
]  # Add as many landing pages as needed
TOP_K = 3  # Number of most relevant document chunks to use
CACHE = {}
prompt = "i dont know"  # default empty

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

def normalize_text(text):
    return text.lower().strip()

def generate_streaming_response(messages):
    try:
        stream = ollama.chat(model=OLLAMA_MODEL, messages=messages, stream=True)
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                yield json.dumps({"answer": chunk["message"]["content"]}) + "\n"
    except Exception as e:
        logging.error(f"Ollama streaming failed: {e}")
        yield json.dumps({"answer": "Error: Failed to generate response."}) + "\n"


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
                emb = get_embedding( normalize_text(chunk))
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

def retrieve_relevant_chunks(query_emb, top_k=TOP_K, min_sim=0.35):
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
    query_emb = get_embedding( normalize_text(query))
    relevant_docs = retrieve_relevant_chunks(query_emb)

    separator = "\n---\n"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": f"Context:\n{separator.join(relevant_docs)}"},
        {"role": "user", "content": query}
    ]


    # If no relevant docs, respond immediately
    if not relevant_docs:
        return Response(stream_response(["I’m sorry, I don’t have enough information to answer that."]),
                        mimetype="application/json")
    
    # Stream Ollama output in real time
    return Response(generate_streaming_response(messages), mimetype="application/json")


    # # Retrieve relevant document chunks
    # if not relevant_docs or all(len(doc.strip()) == 0 for doc in relevant_docs):
    # # No relevant info — return immediately
    #     prompt = "i dont know"
    #     answer = "I’m sorry, I don’t have enough information to answer that."
    # else:
    #     # Construct prompt only when we have relevant docs
    #     prompt = SYSTEM_PROMPT + "\n\n" + "\n---\n".join(relevant_docs)
    #     prompt += f"\n\nUser: {query}\nAnswer:"

    #     # Call Ollama
    #     # Call Ollama safely
    #     try:
    #         response = ollama.chat(
    #             model=OLLAMA_MODEL,
    #             messages=[{"role": "system", "content": prompt}],
    #             max_tokens=100
    #         )
    #         answer = response['message']['content']
    #     except Exception as e:
    #         logging.error(f"Ollama call failed: {e}")
    #         answer = "Error: Failed to generate response."

    # Cache result
    CACHE[key] = answer
    return Response(stream_response([answer]), mimetype="application/json")

# ----------------- Run App -----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)

