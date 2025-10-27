FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ollama_api.py .

# Expose Flask API
EXPOSE 5000

CMD ["python", "ollama_api.py"]
# FROM python:3.11-slim

# WORKDIR /app

# # Install system deps for numpy/sentence-transformers
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential git curl && \
#     rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

COPY ollama_api.py .

EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "ollama_api:app"]
