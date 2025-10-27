from flask import Flask, request, jsonify
from ollama import Client

app = Flask(__name__)
#ollama = Client(host='http://127.0.0.1:11434')
ollama = Client(host='http://host.docker.internal:11434')

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
