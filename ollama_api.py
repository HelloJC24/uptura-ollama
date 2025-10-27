from flask import Flask, request, jsonify
from ollama import Client

app = Flask(__name__)

# Use the host machine's IP (since verified reachable)
#OLLAMA_URL = "http://72.60.43.106:11434"
OLLAMA_URL = "http://localhost:11434"
ollama = Client(host=OLLAMA_URL)

@app.route('/ask', methods=['POST'])
def ask_model():
    data = request.get_json()
    query = data.get('query', '')
    model = data.get('model', 'mistral')

    if not query:
        return jsonify({"error": "Missing query"}), 400

    response = ollama.chat(model=model, messages=[
        {"role": "user", "content": query}
    ])

    return jsonify({"answer": response['message']['content']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
