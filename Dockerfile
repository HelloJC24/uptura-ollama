FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ollama_api.py .

# Expose Flask API
EXPOSE 5000

CMD ["python", "ollama_api.py"]
