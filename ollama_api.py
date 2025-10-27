"""
Enhanced Flask RAG API - VERSION 5
A robust Retrieval-Augmented Generation API with improved architecture, 
comprehensive logging, performance optimization, and configurable streaming.
"""

import threading
import time
import logging
import json
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, Response

# Import our improved modules
from config import Config
from models import document_processor, query_cache
from services.redis_service import redis_service
from services.ollama_service import ollama_service
from services.embedding_service import embedding_service
from utils import (
    RAGRetriever, ResponseGenerator, ResponseFormatter, 
    validate_query, sanitize_text, log_request_info,
    create_health_check_response, performance_monitor,
    timing_decorator
)

# Initialize logging first
Config.setup_logging()
logger = logging.getLogger(__name__)

# Validate configuration
if not Config.validate_config():
    logger.error("Configuration validation failed. Exiting.")
    exit(1)

Config.log_config()

# Initialize Flask app
app = Flask(Config.APP_NAME)

# Global RAG retriever instance
rag_retriever: Optional[RAGRetriever] = None

@timing_decorator
def initialize_services():
    """Initialize all services and warm up models"""
    logger.info("Initializing services...")
    
    # Check service availability
    services_status = {
        "redis": redis_service.is_available(),
        "ollama": ollama_service.is_available(),
        "embeddings": embedding_service.is_available()
    }
    
    logger.info(f"Service status: {services_status}")
    
    # Log warnings for unavailable services
    for service, available in services_status.items():
        if not available:
            logger.warning(f"{service.title()} service is not available")
    
    return services_status

@timing_decorator
def load_documents():
    """Load and process documents in background"""
    global rag_retriever
    
    logger.info("Starting document processing...")
    try:
        # Process all configured documents
        document_processor.process_all_documents()
        
        # Initialize RAG retriever with processed chunks
        rag_retriever = RAGRetriever(document_processor.chunks)
        
        logger.info(f"Document processing completed. RAG retriever initialized with {len(document_processor.chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error during document processing: {e}")
        # Initialize with empty chunks to prevent crashes
        rag_retriever = RAGRetriever([])

# Flask Routes

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    logger.debug("Health check requested")
    health_data = create_health_check_response()
    return jsonify(health_data), 200

@app.route("/stats", methods=["GET"])
def get_stats():
    """Get system statistics"""
    logger.debug("Statistics requested")
    
    try:
        stats = {
            "performance": performance_monitor.get_stats(),
            "cache": query_cache.get_stats(),
            "documents": document_processor.get_stats(),
            "services": {
                "redis": redis_service.get_stats(),
                "ollama": ollama_service.get_stats(),
                "embeddings": embedding_service.get_stats()
            }
        }
        
        return jsonify(ResponseFormatter.format_stats_response(stats)), 200
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return ResponseFormatter.format_error_response("Failed to retrieve statistics", 500)

@app.route("/ask", methods=["POST"])
@timing_decorator
def ask_model():
    """Main endpoint for asking questions"""
    start_time = time.time()
    user_ip = request.remote_addr
    
    try:
        # Get and validate request data
        data = request.get_json()
        if not data:
            return ResponseFormatter.format_error_response("Invalid JSON in request body")
        
        log_request_info(data, user_ip)
        
        # Extract and validate query
        raw_query = data.get("query", "")
        query = sanitize_text(raw_query)
        
        is_valid, error_message = validate_query(query)
        if not is_valid:
            return ResponseFormatter.format_error_response(error_message)
        
        # Extract streaming preference (defaults to config setting)
        use_streaming = data.get("streaming", Config.ENABLE_STREAMING)
        
        logger.info(f"Processing query: '{query[:100]}...' (streaming: {use_streaming})")
        
        # Check cache first
        cached_answer = query_cache.get(query)
        if cached_answer:
            logger.info("Returning cached answer")
            performance_monitor.record_request(time.time() - start_time, cache_hit=True)
            
            if use_streaming:
                def cached_generator():
                    yield cached_answer
                return ResponseFormatter.format_streaming_response(cached_generator())
            else:
                return ResponseFormatter.format_direct_response(cached_answer)
        
        # Check if RAG retriever is ready
        if not rag_retriever:
            logger.warning("RAG retriever not initialized")
            return ResponseFormatter.format_error_response("System is still initializing. Please try again in a moment.", 503)
        
        # Retrieve relevant documents
        relevant_chunks = rag_retriever.retrieve_relevant_chunks(query)
        
        if not relevant_chunks:
            logger.info("No relevant documents found for query")
            answer = "I'm sorry, I don't have enough information to answer that question."
            
            # Cache the result
            query_cache.set(query, answer)
            performance_monitor.record_request(time.time() - start_time, cache_hit=False)
            
            if use_streaming:
                def no_info_generator():
                    yield answer
                return ResponseFormatter.format_streaming_response(no_info_generator())
            else:
                return ResponseFormatter.format_direct_response(answer)
        
        # Create context from relevant chunks
        context = rag_retriever.create_context(relevant_chunks)
        
        logger.info(f"Found {len(relevant_chunks)} relevant chunks, context length: {len(context)} characters")
        
        # Generate answer
        if use_streaming:
            # Streaming response
            answer_generator = ResponseGenerator.generate_answer(query, context, streaming=True)
            
            # Collect full answer for caching
            def cache_and_stream():
                full_answer = ""
                for chunk in answer_generator:
                    full_answer += chunk
                    yield chunk
                
                # Cache the complete answer
                query_cache.set(query, full_answer)
                performance_monitor.record_request(time.time() - start_time, cache_hit=False)
            
            return ResponseFormatter.format_streaming_response(cache_and_stream())
        
        else:
            # Direct response
            answer = ResponseGenerator.generate_answer(query, context, streaming=False)
            
            # Cache the result
            query_cache.set(query, answer)
            performance_monitor.record_request(time.time() - start_time, cache_hit=False)
            
            return ResponseFormatter.format_direct_response(answer)
    
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        performance_monitor.record_request(time.time() - start_time, cache_hit=False, error=True)
        return ResponseFormatter.format_error_response("Internal server error", 500)

@app.route("/stream", methods=["POST"])
@timing_decorator  
def stream_model():
    """Optimized streaming endpoint with Server-Sent Events"""
    start_time = time.time()
    user_ip = request.remote_addr
    
    try:
        # Get and validate request data
        data = request.get_json()
        if not data:
            return ResponseFormatter.format_error_response("Invalid JSON in request body")
        
        log_request_info(data, user_ip)
        
        # Extract and validate query
        raw_query = data.get("query", "")
        query = sanitize_text(raw_query)
        
        is_valid, error_message = validate_query(query)
        if not is_valid:
            return ResponseFormatter.format_error_response(error_message)
        
        logger.info(f"Processing streaming query: '{query[:100]}...'")
        
        # Check if RAG retriever is ready
        if not rag_retriever:
            logger.warning("RAG retriever not initialized")
            return ResponseFormatter.format_error_response("System is still initializing. Please try again in a moment.", 503)
        
        def stream_generator():
            try:
                # Send initial status
                status_msg = json.dumps({'status': 'processing', 'message': 'Retrieving relevant documents...'})
                yield f"data: {status_msg}\n\n"
                
                # Retrieve relevant documents
                relevant_chunks = rag_retriever.retrieve_relevant_chunks(query)
                
                if not relevant_chunks:
                    no_info_msg = json.dumps({'answer': "I'm sorry, I don't have enough information to answer that question."})
                    yield f"data: {no_info_msg}\n\n"
                    complete_msg = json.dumps({'status': 'complete'})
                    yield f"data: {complete_msg}\n\n"
                    return
                
                # Send status update
                found_msg = json.dumps({'status': 'generating', 'message': f'Found {len(relevant_chunks)} relevant documents. Generating answer...'})
                yield f"data: {found_msg}\n\n"
                
                # Create context and generate answer
                context = rag_retriever.create_context(relevant_chunks)
                answer_generator = ResponseGenerator.generate_answer(query, context, streaming=True)
                
                # Stream the answer
                full_answer = ""
                for chunk in answer_generator:
                    full_answer += chunk
                    chunk_msg = json.dumps({'answer': chunk})
                    yield f"data: {chunk_msg}\n\n"
                
                # Cache and complete
                query_cache.set(query, full_answer)
                complete_msg = json.dumps({'status': 'complete'})
                yield f"data: {complete_msg}\n\n"
                
                performance_monitor.record_request(time.time() - start_time, cache_hit=False)
                
            except Exception as e:
                logger.error(f"Error in streaming: {e}")
                error_msg = json.dumps({'error': f'Streaming error: {str(e)}'})
                yield f"data: {error_msg}\n\n"
                performance_monitor.record_request(time.time() - start_time, cache_hit=False, error=True)
        
        return Response(
            stream_generator(),
            mimetype="text/plain",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control, Content-Type",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )
    
    except Exception as e:
        logger.error(f"Error processing streaming request: {e}", exc_info=True)
        performance_monitor.record_request(time.time() - start_time, cache_hit=False, error=True)
        return ResponseFormatter.format_error_response("Internal server error", 500)

@app.route("/clear-cache", methods=["POST"])
def clear_cache():
    """Clear the query cache"""
    logger.info("Cache clear requested")
    
    try:
        # Clear in-memory cache
        query_cache.clear()
        
        # Clear Redis cache if available
        if redis_service.is_available():
            cleared_count = redis_service.clear_cache("doc_chunk:*")
            logger.info(f"Cleared {cleared_count} items from Redis cache")
        
        return jsonify({"status": "success", "message": "Cache cleared successfully"}), 200
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return ResponseFormatter.format_error_response("Failed to clear cache", 500)

@app.route("/reload-documents", methods=["POST"])
def reload_documents():
    """Reload documents from configured URLs"""
    logger.info("Document reload requested")
    
    try:
        # Start document processing in background
        threading.Thread(target=load_documents, daemon=True).start()
        
        return jsonify({
            "status": "success", 
            "message": "Document reloading started in background"
        }), 202
        
    except Exception as e:
        logger.error(f"Error starting document reload: {e}")
        return ResponseFormatter.format_error_response("Failed to start document reload", 500)

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    return ResponseFormatter.format_error_response("Bad request", 400)

@app.errorhandler(404)
def not_found(error):
    return ResponseFormatter.format_error_response("Endpoint not found", 404)

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return ResponseFormatter.format_error_response("Internal server error", 500)

# Startup initialization
def initialize_app():
    """Initialize the application"""
    logger.info("Starting Flask RAG API initialization...")
    
    # Initialize services
    services_status = initialize_services()
    
    # Start document loading in background
    document_thread = threading.Thread(target=load_documents, daemon=True)
    document_thread.start()
    
    logger.info("Application initialization completed!")
    logger.info(f"API will be available at http://{Config.HOST}:{Config.PORT}")
    logger.info("Available endpoints:")
    logger.info("  POST /ask - Ask questions")
    logger.info("  GET  /health - Health check")
    logger.info("  GET  /stats - System statistics")
    logger.info("  POST /clear-cache - Clear cache")
    logger.info("  POST /reload-documents - Reload documents")

if __name__ == "__main__":
    try:
        initialize_app()
        
        # Run Flask app
        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG,
            threaded=True,
            use_reloader=False  # Disable reloader to prevent double initialization
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        exit(1)