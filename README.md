# Enhanced Flask RAG API - Version 5

A robust Retrieval-Augmented Generation (RAG) API built with Flask, featuring comprehensive logging, performance optimization, configurable streaming, and modular architecture.

## 🚀 **New Features & Improvements**

### Architecture Improvements
- **Modular Design**: Separated concerns into distinct modules (`config.py`, `models.py`, `services/`, `utils.py`)
- **Service Layer**: Dedicated services for Redis, Ollama, embedding, and conversation operations
- **Configuration Management**: Environment variable support with validation
- **Error Handling**: Comprehensive error handling throughout the application

### Conversation System 🗣️
- **User Conversations**: Maintain conversation history per user with configurable limits
- **Context Awareness**: Responses include previous conversation context
- **Automatic Management**: Configurable TTL and cleanup of old conversations
- **User Identification**: Support for user-specific conversations via `user_id`
- **Memory Limit**: Configurable conversation history (default: 10 messages)

### Performance Optimizations
- **LRU Caching**: Intelligent caching with TTL and memory management
- **Batch Processing**: Efficient batch embedding generation
- **Connection Pooling**: Optimized Redis connections with health checks
- **Background Processing**: Document loading and model warmup in background threads
- **Contextual Caching**: Cache keys include user context for personalized responses

### Enhanced RAG System
- **Improved Chunking**: Configurable chunk size with overlap for better context
- **Smart Retrieval**: Similarity threshold filtering for relevant results only
- **Context Building**: Better context construction with source attribution
- **Document Processing**: Robust HTML parsing and text extraction

### Logging & Monitoring
- **Comprehensive Logging**: Function-level logging with structured format
- **Performance Metrics**: Request timing, cache hit rates, error tracking
- **Health Monitoring**: Service status tracking and health endpoints
- **Statistics API**: Detailed system statistics and performance data

### Streaming Configuration
- **Configurable Streaming**: Toggle streaming mode via config or per-request
- **Response Formatting**: Consistent JSON streaming format
- **Error Streaming**: Proper error handling in streaming responses

## 📁 Project Structure

```
upupapi/
├── config.py              # Configuration management
├── models.py              # Data models and document processing
├── utils.py               # Utility functions and helpers
├── services/              # Service layer
│   ├── __init__.py
│   ├── redis_service.py   # Redis operations
│   ├── ollama_service.py  # Ollama LLM integration
│   └── embedding_service.py # Text embedding operations
├── ollama_api.py          # Main Flask application
├── requirements.txt       # Dependencies
├── Dockerfile            # Container configuration
├── dockploy.yaml         # Deployment configuration
└── .env.example          # Environment variables template
```

## 🛠️ Setup & Installation

### 1. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your configuration
nano .env
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python ollama_api.py
```

## 🔧 Configuration

All configuration is managed through environment variables. Key settings include:

### Service Configuration
- `OLLAMA_HOST`: Ollama server URL
- `OLLAMA_MODEL`: Model name to use
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`: Redis connection details

### RAG Settings
- `CHUNK_SIZE`: Document chunk size (default: 200 words)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50 words)
- `TOP_K`: Number of relevant chunks to retrieve (default: 3)
- `MIN_SIMILARITY`: Minimum similarity threshold (default: 0.35)

### Response Settings
- `ENABLE_STREAMING`: Enable streaming responses (default: True)
- `CACHE_TTL`: Cache time-to-live in seconds (default: 3600)
- `MAX_CACHE_SIZE`: Maximum cache entries (default: 1000)

### Conversation Configuration
- `ENABLE_CONVERSATIONS`: Enable/disable conversation features (default: True)
- `MAX_CONVERSATION_HISTORY`: Maximum messages per conversation (default: 10)
- `CONVERSATION_TTL`: Conversation expiry time in seconds (default: 86400 = 24 hours)
- `REQUIRE_USER_ID`: Whether user_id is required for all requests (default: False)

## 📡 API Endpoints

### POST /ask
Ask questions to the RAG system with optional conversation history.

**Request:**
```json
{
    "query": "do you know Gogel?",
    "user_id": "user@gmail.com",  // Optional: for conversation history
    "streaming": true             // Optional: override global streaming setting
}
```

**Response (Streaming):**
```
data: {"status": "connected"}

data: {"answer": "Yes, I am aware of Gogel..."}

data: {"status": "complete"}
```

**Conversation Flow Example:**
```json
// First question
{
    "query": "do you know Gogel?",
    "user_id": "user@gmail.com"
}
// Response: "Yes, Gogel is a real estate firm..."

// Follow-up question (will include previous context)
{
    "query": "What services do they offer?",
    "user_id": "user@gmail.com" 
}
// Response: "Based on our previous discussion about Gogel, they offer..."
```

### POST /stream (Optimized for Real-time Streaming)
Dedicated streaming endpoint with better real-time performance.

**Request:**
```json
{
    "query": "Your question here"
}
```

**Response (Server-Sent Events format):**
```
data: {"status": "processing", "message": "Retrieving relevant documents..."}

data: {"status": "generating", "message": "Found 3 relevant documents. Generating answer..."}

data: {"answer": "Yes"}

data: {"answer": ","}

data: {"answer": " I"}

data: {"answer": " am"}

data: {"answer": " aware"}

data: {"status": "complete"}
```

### Testing Streaming in Postman
1. Use the `/stream` endpoint for best results
2. In Postman, look for the "Stream" toggle in the response section
3. You should see responses appear in real-time, not all at once
4. If responses appear all at once, try using curl instead:

```bash
curl -X POST http://localhost:5000/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "do you know Gogel?"}' \
  --no-buffer
```

### Performance Optimization Tips
1. **Use `.env.fast` configuration** for better performance:
   ```bash
   cp .env.fast .env
   ```

2. **Reduce chunk size and TOP_K** for faster retrieval:
   ```
   CHUNK_SIZE=150
   TOP_K=2
   MIN_SIMILARITY=0.3
   ```

3. **Lower logging level** for production:
   ```
   LOG_LEVEL=WARNING
   ```

4. **Use the `/stream` endpoint** instead of `/ask` for real-time streaming

### GET /conversations
List all active conversations.

**Response:**
```json
{
    "status": "success",
    "active_conversations": ["user@gmail.com", "another@user.com"],
    "total_count": 2,
    "service_stats": {
        "local_cache_size": 5,
        "active_conversations": 2,
        "max_history": 10,
        "conversations_enabled": true
    }
}
```

### GET /conversations/<user_id>/history
Get conversation history for a specific user.

**Parameters:**
- `limit` (optional): Maximum number of messages to return

**Response:**
```json
{
    "status": "success",
    "user_id": "user@gmail.com",
    "history": [
        {
            "role": "user",
            "content": "do you know Gogel?",
            "timestamp": "2025-10-28T10:30:00Z"
        },
        {
            "role": "assistant", 
            "content": "Yes, Gogel is a real estate firm...",
            "timestamp": "2025-10-28T10:30:15Z"
        }
    ],
    "stats": {
        "total_messages": 4,
        "user_messages": 2,
        "assistant_messages": 2,
        "created_at": "2025-10-28T10:30:00Z",
        "last_updated": "2025-10-28T10:35:00Z"
    }
}
```

### POST /conversations/<user_id>/clear
Clear conversation history for a specific user.

**Response:**
```json
{
    "status": "success",
    "message": "Conversation history cleared for user: user@gmail.com"
}
```

### POST /conversations/cleanup
Clean up all expired conversations.

**Response:**
```json
{
    "status": "success",
    "message": "Cleaned up 3 expired conversations",
    "cleaned_count": 3
}
```

### GET /stats
System statistics and performance metrics.

**Response:**
```json
{
    "status": "success",
    "timestamp": 1635789456.789,
    "stats": {
        "performance": {
            "requests": 150,
            "average_response_time": 1.25,
            "cache_hit_rate": 0.45,
            "error_rate": 0.02
        },
        "cache": {
            "size": 45,
            "max_size": 1000,
            "hit_rate": 0.45
        },
        "documents": {
            "total_chunks": 1250,
            "processed_urls": 6
        }
    }
}
```

### POST /clear-cache
Clear all cached data.

### POST /reload-documents
Reload documents from configured URLs.

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t ollama-rag-api .
```

### Run Container
```bash
docker run -p 5000:5000 --env-file .env ollama-rag-api
```

### Using Docker Compose (recommended)
```yaml
version: '3.8'
services:
  ollama-api:
    build: .
    ports:
      - "5000:5000"
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 🔍 Monitoring & Debugging

### Logging Levels
Set `LOG_LEVEL` environment variable:
- `DEBUG`: Detailed debugging information
- `INFO`: General information (default)
- `WARNING`: Warning messages only
- `ERROR`: Error messages only

### Performance Monitoring
- Monitor `/stats` endpoint for performance metrics
- Check `/health` endpoint for service availability
- Review application logs for detailed operation info

### Common Issues
1. **Ollama Connection**: Ensure Ollama server is running and accessible
2. **Redis Connection**: Verify Redis credentials and network connectivity
3. **Model Loading**: Check if the specified Ollama model exists
4. **Memory Usage**: Monitor cache size and adjust `MAX_CACHE_SIZE` if needed

## 📈 Performance Tuning

### Cache Optimization
- Adjust `CACHE_TTL` based on content update frequency
- Increase `MAX_CACHE_SIZE` for better hit rates (watch memory usage)
- Monitor cache hit rates via `/stats` endpoint

### Embedding Performance
- Use GPU-enabled sentence transformers if available
- Adjust `CHUNK_SIZE` based on your content type
- Consider using smaller embedding models for faster processing

### Concurrent Requests
- The application is thread-safe and supports concurrent requests
- Adjust Flask's threading settings for high-load scenarios
- Consider using a WSGI server like Gunicorn for production

## 🔒 Security Considerations

1. **Environment Variables**: Never commit `.env` files
2. **Redis Security**: Use strong passwords and network isolation
3. **Input Validation**: The API includes input sanitization
4. **Rate Limiting**: Consider adding rate limiting for production use
5. **HTTPS**: Use HTTPS in production environments

## 🤝 Contributing

1. Follow the modular architecture when adding features
2. Add comprehensive logging to new functions
3. Include error handling for external service calls
4. Update documentation for configuration changes
5. Test with different embedding models and chunk sizes

## 📄 License

This project is licensed under the MIT License.