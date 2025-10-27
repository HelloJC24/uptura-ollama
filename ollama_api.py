from flask import Flask, request, Response
from ollama import Client

app = Flask(__name__)
OLLAMA_URL = "http://72.60.43.106:11434"
ollama = Client(host=OLLAMA_URL)

@app.route('/ask', methods=['POST'])
def ask_model():
    data = request.get_json()
    query = data.get('query', '')
    model = data.get('model', 'mistral')

    if not query:
        return jsonify({"error": "Missing query"}), 400

    def generate():
        for chunk in ollama.chat(model=model, messages=[{"role": "user", "content": query}], stream=True):
            yield chunk['delta']  # stream chunk content as it arrives

    return Response(generate(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
