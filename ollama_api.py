"""
Enhanced Flask RAG API - VERSION 5
A robust Retrieval-Augmented Generation API with improved architecture, 
comprehensive logging, performance optimization, and configurable streaming.
"""

import threading
import time
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, Response

# Import our improved modules
from config import Config
from models import document_processor, query_cache
from services.redis_service import redis_service
from services.ollama_service import ollama_service
from services.embedding_service import embedding_service
from services.conversation_service import conversation_service
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

# Add request timing middleware
@app.before_request
def before_request():
    """Log request start time"""
    request.start_time = time.time()
    logger.info(f"Request started: {request.method} {request.path}")

@app.after_request
def after_request(response):
    """Log request completion time"""
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        logger.info(f"Request completed: {request.method} {request.path} - {response.status_code} in {duration:.3f}s")
    return response

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
            "conversations": conversation_service.get_service_stats() if Config.ENABLE_CONVERSATIONS else {"enabled": False},
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
    """Main endpoint for asking questions with conversation support"""
    start_time = time.time()
    user_ip = request.remote_addr
    
    # Immediate logging to show request was received
    logger.info(f"=== ASK REQUEST RECEIVED at {time.strftime('%H:%M:%S.%f')[:-3]} from {user_ip} ===")
    
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
        
        # Extract user ID (optional unless required by config)
        user_id = data.get("user_id", "anonymous")
        if Config.REQUIRE_USER_ID and user_id == "anonymous":
            return ResponseFormatter.format_error_response("user_id is required")
        
        # Extract streaming preference (defaults to config setting)
        use_streaming = data.get("streaming", Config.ENABLE_STREAMING)
        
        logger.info(f"Processing query for user {user_id}: '{query[:100]}...' (streaming: {use_streaming})")
        
        # Get conversation context if conversations are enabled
        context_start = time.time()
        conversation_messages = []
        if Config.ENABLE_CONVERSATIONS and user_id != "anonymous":
            conversation_messages = conversation_service.get_context_messages(user_id, include_system_prompt=False)
            logger.debug(f"Retrieved {len(conversation_messages)} conversation messages for user {user_id}")
        context_time = time.time() - context_start
        logger.debug(f"Conversation context retrieval took {context_time:.3f}s")
        
        # Create cache key that includes user context
        cache_key = f"{user_id}:{query}" if Config.ENABLE_CONVERSATIONS else query
        
        # Check cache first
        cache_start = time.time()
        cached_answer = query_cache.get(cache_key)
        cache_time = time.time() - cache_start
        logger.debug(f"Cache check took {cache_time:.3f}s")
        
        if cached_answer:
            logger.info("Returning cached answer")
            performance_monitor.record_request(time.time() - start_time, cache_hit=True)
            
            # Add to conversation history if enabled
            if Config.ENABLE_CONVERSATIONS and user_id != "anonymous":
                conversation_service.add_message(user_id, "user", query)
                conversation_service.add_message(user_id, "assistant", cached_answer)
            
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
        retrieval_start = time.time()
        relevant_chunks = rag_retriever.retrieve_relevant_chunks(query)
        retrieval_time = time.time() - retrieval_start
        logger.info(f"Document retrieval took {retrieval_time:.3f}s, found {len(relevant_chunks)} chunks")
        
        if not relevant_chunks:
            logger.info("No relevant documents found for query")
            answer = "I'm sorry, I don't have enough information to answer that question."
            
            # Add to conversation history if enabled
            if Config.ENABLE_CONVERSATIONS and user_id != "anonymous":
                conversation_service.add_message(user_id, "user", query)
                conversation_service.add_message(user_id, "assistant", answer)
            
            # Cache the result
            query_cache.set(cache_key, answer)
            performance_monitor.record_request(time.time() - start_time, cache_hit=False)
            
            if use_streaming:
                def no_info_generator():
                    yield answer
                return ResponseFormatter.format_streaming_response(no_info_generator())
            else:
                return ResponseFormatter.format_direct_response(answer)
        
        # Create context from relevant chunks
        context_build_start = time.time()
        context = rag_retriever.create_context(relevant_chunks)
        context_build_time = time.time() - context_build_start
        
        logger.info(f"Context building took {context_build_time:.3f}s, context length: {len(context)} characters")
        
        # Generate answer with conversation context
        generation_start = time.time()
        if use_streaming:
            # Streaming response
            answer_generator = ResponseGenerator.generate_answer(
                query, context, conversation_messages, streaming=True
            )
            
            # Collect full answer for caching and conversation history
            def cache_and_stream():
                full_answer = ""
                for chunk in answer_generator:
                    full_answer += chunk
                    yield chunk
                
                # Add to conversation history if enabled
                if Config.ENABLE_CONVERSATIONS and user_id != "anonymous":
                    conversation_service.add_message(user_id, "user", query)
                    conversation_service.add_message(user_id, "assistant", full_answer)
                
                # Cache the complete answer
                query_cache.set(cache_key, full_answer)
                performance_monitor.record_request(time.time() - start_time, cache_hit=False)
            
            return ResponseFormatter.format_streaming_response(cache_and_stream())
        
        else:
            # Direct response
            answer = ResponseGenerator.generate_answer(
                query, context, conversation_messages, streaming=False
            )
            
            # Add to conversation history if enabled
            if Config.ENABLE_CONVERSATIONS and user_id != "anonymous":
                conversation_service.add_message(user_id, "user", query)
                conversation_service.add_message(user_id, "assistant", answer)
            
            # Cache the result
            query_cache.set(cache_key, answer)
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

@app.route("/conversations/<user_id>/history", methods=["GET"])
def get_conversation_history(user_id: str):
    """Get conversation history for a user"""
    logger.debug(f"Conversation history requested for user: {user_id}")
    
    try:
        if not Config.ENABLE_CONVERSATIONS:
            return ResponseFormatter.format_error_response("Conversations are disabled", 400)
        
        # Get optional limit parameter
        limit = request.args.get('limit', type=int)
        
        history = conversation_service.get_conversation_history(user_id, limit)
        stats = conversation_service.get_conversation_stats(user_id)
        
        return jsonify({
            "status": "success",
            "user_id": user_id,
            "history": history,
            "stats": stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting conversation history for user {user_id}: {e}")
        return ResponseFormatter.format_error_response("Failed to retrieve conversation history", 500)

@app.route("/conversations/<user_id>/clear", methods=["POST"])
def clear_conversation_history(user_id: str):
    """Clear conversation history for a user"""
    logger.info(f"Clearing conversation history for user: {user_id}")
    
    try:
        if not Config.ENABLE_CONVERSATIONS:
            return ResponseFormatter.format_error_response("Conversations are disabled", 400)
        
        success = conversation_service.clear_conversation(user_id)
        
        if success:
            return jsonify({
                "status": "success",
                "message": f"Conversation history cleared for user: {user_id}"
            }), 200
        else:
            return ResponseFormatter.format_error_response("Failed to clear conversation history", 500)
        
    except Exception as e:
        logger.error(f"Error clearing conversation for user {user_id}: {e}")
        return ResponseFormatter.format_error_response("Failed to clear conversation history", 500)

@app.route("/conversations", methods=["GET"])
def list_conversations():
    """List all active conversations"""
    logger.debug("Listing all active conversations")
    
    try:
        if not Config.ENABLE_CONVERSATIONS:
            return ResponseFormatter.format_error_response("Conversations are disabled", 400)
        
        active_users = conversation_service.list_active_conversations()
        service_stats = conversation_service.get_service_stats()
        
        return jsonify({
            "status": "success",
            "active_conversations": active_users,
            "total_count": len(active_users),
            "service_stats": service_stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        return ResponseFormatter.format_error_response("Failed to list conversations", 500)

@app.route("/conversations/cleanup", methods=["POST"])
def cleanup_conversations():
    """Clean up expired conversations"""
    logger.info("Manual conversation cleanup requested")
    
    try:
        if not Config.ENABLE_CONVERSATIONS:
            return ResponseFormatter.format_error_response("Conversations are disabled", 400)
        
        cleaned_count = conversation_service.cleanup_expired_conversations()
        
        return jsonify({
            "status": "success",
            "message": f"Cleaned up {cleaned_count} expired conversations",
            "cleaned_count": cleaned_count
        }), 200
        
    except Exception as e:
        logger.error(f"Error during conversation cleanup: {e}")
        return ResponseFormatter.format_error_response("Failed to cleanup conversations", 500)

@app.route("/debug/query", methods=["POST"])
def debug_query():
    """Debug endpoint to see what's happening with query processing"""
    try:
        data = request.get_json()
        if not data:
            return ResponseFormatter.format_error_response("Invalid JSON in request body")
        
        query = data.get("query", "")
        if not query:
            return ResponseFormatter.format_error_response("Query is required")
        
        logger.info(f"Debug query: {query}")
        
        # Check if RAG retriever is ready
        if not rag_retriever:
            return jsonify({
                "error": "RAG retriever not initialized",
                "total_chunks": 0
            }), 503
        
        # Get embeddings and similarity scores
        query_embedding = embedding_service.get_embedding(query)
        if not query_embedding:
            return jsonify({
                "error": "Failed to generate query embedding",
                "total_chunks": len(rag_retriever.chunks)
            }), 500
        
        # Calculate similarities for all chunks
        similarities = []
        for i, chunk in enumerate(rag_retriever.chunks):
            if chunk.embedding:
                similarity = embedding_service.calculate_similarity(query_embedding, chunk.embedding)
                similarities.append({
                    "index": i,
                    "similarity": round(similarity, 4),
                    "source_url": chunk.source_url,
                    "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Get relevant chunks using current settings
        relevant_chunks = rag_retriever.retrieve_relevant_chunks(query)
        
        debug_info = {
            "query": query,
            "query_embedding_length": len(query_embedding),
            "total_chunks": len(rag_retriever.chunks),
            "config": {
                "top_k": Config.TOP_K,
                "min_similarity": Config.MIN_SIMILARITY,
                "chunk_size": Config.CHUNK_SIZE,
                "system_prompt_preview": Config.SYSTEM_PROMPT[:200] + "..."
            },
            "top_similarities": similarities[:10],  # Top 10 similarities
            "relevant_chunks_found": len(relevant_chunks),
            "relevant_chunks": [
                {
                    "similarity": "N/A",  # We'd need to recalculate
                    "source_url": chunk.source_url,
                    "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                }
                for chunk in relevant_chunks
            ],
            "chunks_above_threshold": len([s for s in similarities if s["similarity"] >= Config.MIN_SIMILARITY]),
            "max_similarity": similarities[0]["similarity"] if similarities else 0,
            "min_similarity_threshold": Config.MIN_SIMILARITY
        }
        
        return jsonify(debug_info), 200
        
    except Exception as e:
        logger.error(f"Error in debug query: {e}", exc_info=True)
        return ResponseFormatter.format_error_response(f"Debug error: {str(e)}", 500)

@app.route("/debug/contact", methods=["POST"])
def debug_contact():
    """Debug endpoint specifically for contact-related queries"""
    try:
        data = request.get_json()
        if not data:
            return ResponseFormatter.format_error_response("Invalid JSON in request body")
        
        query = data.get("query", "contact information")
        logger.info(f"Debug contact query: {query}")
        
        # Check if RAG retriever is ready
        if not rag_retriever:
            return jsonify({
                "error": "RAG retriever not initialized",
                "total_chunks": 0
            }), 503
        
        # Get embeddings and similarity scores
        query_embedding = embedding_service.get_embedding(query)
        if not query_embedding:
            return jsonify({
                "error": "Failed to generate query embedding",
                "total_chunks": len(rag_retriever.chunks)
            }), 500
        
        # Calculate similarities for all chunks, looking specifically for contact info
        contact_keywords = ['contact', 'email', 'phone', 'address', 'call', 'write', '@', 'tel:', 'mailto:']
        contact_chunks = []
        
        for i, chunk in enumerate(rag_retriever.chunks):
            if chunk.embedding:
                similarity = embedding_service.calculate_similarity(query_embedding, chunk.embedding)
                
                # Check if chunk contains potential contact information
                has_contact_info = any(keyword in chunk.text.lower() for keyword in contact_keywords)
                
                chunk_info = {
                    "index": i,
                    "similarity": round(similarity, 4),
                    "source_url": chunk.source_url,
                    "text_preview": chunk.text[:400] + "..." if len(chunk.text) > 400 else chunk.text,
                    "has_contact_keywords": has_contact_info,
                    "above_threshold": similarity >= Config.MIN_SIMILARITY
                }
                
                if has_contact_info or similarity >= Config.MIN_SIMILARITY:
                    contact_chunks.append(chunk_info)
        
        # Sort by similarity
        contact_chunks.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Get the actual context that would be used
        context = RAGRetriever.build_context(rag_retriever.retrieve_relevant_chunks(query))
        
        debug_info = {
            "query": query,
            "config": {
                "min_similarity": Config.MIN_SIMILARITY,
                "top_k": Config.TOP_K
            },
            "total_chunks": len(rag_retriever.chunks),
            "contact_related_chunks": len(contact_chunks),
            "chunks_above_threshold": len([c for c in contact_chunks if c["above_threshold"]]),
            "top_contact_chunks": contact_chunks[:10],
            "context_used": context[:1000] + "..." if len(context) > 1000 else context,
            "system_prompt_preview": Config.SYSTEM_PROMPT[:300] + "..."
        }
        
        return jsonify(debug_info), 200
        
    except Exception as e:
        logger.error(f"Error in debug contact: {e}", exc_info=True)
        return ResponseFormatter.format_error_response(f"Debug contact error: {str(e)}", 500)

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

@app.route('/health/detailed', methods=['GET'])
def detailed_health_check():
    """Detailed health check endpoint for all services"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {}
        }
        
        # Check Redis
        try:
            redis_service.redis_client.ping()
            health_status['services']['redis'] = 'healthy'
        except Exception as e:
            health_status['services']['redis'] = f'unhealthy: {str(e)}'
            health_status['status'] = 'degraded'
        
        # Check Ollama
        try:
            ollama_available = ollama_service.is_available()
            if ollama_available:
                models = ollama_service.get_models()
                health_status['services']['ollama'] = {
                    'status': 'healthy',
                    'models_count': len(models) if models else 0,
                    'models': models[:3] if models else []  # Show first 3 models
                }
            else:
                health_status['services']['ollama'] = 'unhealthy: service not available'
                health_status['status'] = 'degraded'
        except Exception as e:
            health_status['services']['ollama'] = f'unhealthy: {str(e)}'
            health_status['status'] = 'degraded'
        
        # Check embedding service
        try:
            # Test embedding generation
            test_embedding = embedding_service.get_embedding("test")
            if test_embedding and len(test_embedding) > 0:
                health_status['services']['embeddings'] = 'healthy'
            else:
                health_status['services']['embeddings'] = 'unhealthy: no embedding generated'
                health_status['status'] = 'degraded'
        except Exception as e:
            health_status['services']['embeddings'] = f'unhealthy: {str(e)}'
            health_status['status'] = 'degraded'
        
        # Check RAG retriever
        try:
            if rag_retriever and len(rag_retriever.chunks) > 0:
                health_status['services']['rag'] = {
                    'status': 'healthy',
                    'documents_loaded': len(rag_retriever.chunks)
                }
            else:
                health_status['services']['rag'] = 'unhealthy: no documents loaded'
                health_status['status'] = 'degraded'
        except Exception as e:
            health_status['services']['rag'] = f'unhealthy: {str(e)}'
            health_status['status'] = 'degraded'
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/warmup', methods=['POST'])
def warmup_model():
    """Manually trigger model warmup"""
    try:
        logger.info("Manual warmup requested")
        start_time = time.time()
        
        if not ollama_service.is_available():
            return jsonify({
                'status': 'error',
                'message': 'Ollama service not available'
            }), 503
        
        if ollama_service.is_warmed_up:
            return jsonify({
                'status': 'already_warmed',
                'message': 'Model is already warmed up',
                'timestamp': datetime.now().isoformat()
            })
        
        # Force immediate warmup
        ollama_service.ensure_warmup()
        warmup_time = time.time() - start_time
        
        return jsonify({
            'status': 'success',
            'message': f'Model warmup completed in {warmup_time:.2f}s',
            'warmup_time': warmup_time,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Warmup error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get quick service status including warmup state"""
    try:
        return jsonify({
            'ollama_available': ollama_service.is_available(),
            'model_warmed_up': ollama_service.is_warmed_up,
            'warmup_in_progress': ollama_service.warmup_in_progress,
            'model_name': ollama_service.model_name,
            'rag_ready': rag_retriever is not None and len(rag_retriever.chunks) > 0,
            'documents_loaded': len(rag_retriever.chunks) if rag_retriever else 0,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

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
    logger.info("  POST /ask - Ask questions (with optional user_id for conversations)")
    logger.info("  POST /stream - Optimized streaming questions")
    logger.info("  GET  /health - Health check")
    logger.info("  GET  /stats - System statistics")
    logger.info("  POST /clear-cache - Clear cache")
    logger.info("  POST /reload-documents - Reload documents")
    
    if Config.ENABLE_CONVERSATIONS:
        logger.info("  GET  /conversations - List all active conversations")
        logger.info("  GET  /conversations/<user_id>/history - Get user conversation history")
        logger.info("  POST /conversations/<user_id>/clear - Clear user conversation")
        logger.info("  POST /conversations/cleanup - Clean up expired conversations")
        logger.info(f"  Conversations enabled: max {Config.MAX_CONVERSATION_HISTORY} messages, TTL {Config.CONVERSATION_TTL}s")
    else:
        logger.info("  Conversations are disabled")

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