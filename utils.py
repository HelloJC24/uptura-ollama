"""
Utility functions for the RAG API
"""
import json
import logging
import time
import threading
from typing import Dict, Any, Generator, List, Optional, Callable
from functools import wraps
from flask import Response
from config import Config
from models import DocumentChunk, QueryResult
from services.embedding_service import embedding_service
from services.ollama_service import ollama_service

logger = logging.getLogger(__name__)

class RAGRetriever:
    """Enhanced RAG retrieval system"""
    
    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        logger.info(f"RAG Retriever initialized with {len(chunks)} chunks")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = None, min_similarity: float = None) -> List[DocumentChunk]:
        """Retrieve most relevant document chunks for a query"""
        if not self.chunks:
            logger.warning("No document chunks available for retrieval")
            return []
        
        top_k = top_k or Config.TOP_K
        min_similarity = min_similarity or Config.MIN_SIMILARITY
        
        logger.info(f"Retrieving relevant chunks for query: '{query[:50]}...'")
        logger.info(f"Total available chunks: {len(self.chunks)}")
        logger.info(f"Using top_k={top_k}, min_similarity={min_similarity}")
        start_time = time.time()
        
        # Get query embedding
        query_embedding = embedding_service.get_embedding(query)
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []
        
        # Calculate similarities
        similarities = []
        for i, chunk in enumerate(self.chunks):
            if chunk.embedding:
                similarity = embedding_service.calculate_similarity(query_embedding, chunk.embedding)
                similarities.append((similarity, chunk))
                logger.debug(f"Chunk {i} from {chunk.source_url}: similarity={similarity:.3f}")
            else:
                similarities.append((0.0, chunk))
        
        # Sort by similarity and filter
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        logger.info(f"Top similarities found:")
        for i, (similarity, chunk) in enumerate(similarities[:5]):  # Log top 5
            logger.info(f"  {i+1}. Similarity: {similarity:.3f} from {chunk.source_url}")
            logger.info(f"     Text preview: {chunk.text[:100]}...")
        
        relevant_chunks = []
        for similarity, chunk in similarities[:top_k]:
            if similarity >= min_similarity:
                relevant_chunks.append(chunk)
                logger.info(f"Selected chunk with similarity {similarity:.3f} from {chunk.source_url}")
            else:
                logger.warning(f"Rejected chunk with similarity {similarity:.3f} (below threshold {min_similarity})")
        
        retrieval_time = time.time() - start_time
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks in {retrieval_time:.3f}s")
        
        if not relevant_chunks:
            logger.warning("No chunks met the similarity threshold - consider lowering MIN_SIMILARITY")
        
        return relevant_chunks
    
    def create_context(self, chunks: List[DocumentChunk]) -> str:
        """Create context string from relevant chunks"""
        if not chunks:
            logger.warning("No chunks provided for context creation")
            return ""
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source_info = f"[Source {i}: {chunk.source_url}]"
            context_parts.append(f"{source_info}\n{chunk.text}")
        
        context = "\n\n---\n\n".join(context_parts)
        logger.info(f"Created context with {len(context)} characters from {len(chunks)} chunks")
        logger.debug(f"Context preview: {context[:500]}...")
        
        return context

class ResponseGenerator:
    """Handles response generation with streaming support"""
    
    @staticmethod
    def generate_answer(query: str, context: str, conversation_messages: List[Dict[str, str]] = None, streaming: bool = None) -> str:
        """Generate answer using Ollama with conversation context"""
        streaming = streaming if streaming is not None else Config.ENABLE_STREAMING
        
        if not context.strip():
            return "I'm sorry, I don't have enough information to answer that question."
        
        # Check if this is a contact-related query and enhance the prompt
        contact_keywords = ['contact', 'email', 'phone', 'address', 'reach', 'call', 'write']
        is_contact_query = any(keyword in query.lower() for keyword in contact_keywords)
        
        # Build messages with conversation history
        messages = []
        
        # Add system prompt with RAG context
        system_content = f"""{Config.SYSTEM_PROMPT}

Context from BNGC/Gogel company documents:
{context}

IMPORTANT: You must use the above context to answer questions about BNGC/Gogel. Do not refuse to provide business information that is available in the context."""
        
        if is_contact_query:
            system_content += f"""

CONTACT INFORMATION REQUEST DETECTED:
The user is asking for contact information. You MUST provide any contact details (emails, phone numbers, addresses) that are available in the context above. This is publicly available business information from the company's own website and documents. Do not refuse to provide this information."""
        
        messages.append({"role": "system", "content": system_content})
        
        # Add conversation history (excluding system messages)
        if conversation_messages:
            for msg in conversation_messages:
                if msg.get("role") != "system":  # Skip system messages from history
                    messages.append(msg)
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        logger.info(f"Generating {'streaming' if streaming else 'non-streaming'} answer with {len(messages)} messages in context")
        
        if streaming:
            return ResponseGenerator._generate_streaming_answer(messages)
        else:
            return ResponseGenerator._generate_direct_answer(messages)
    
    @staticmethod
    def _generate_direct_answer(messages: List[Dict[str, str]]) -> str:
        """Generate direct (non-streaming) answer"""
        answer = ollama_service.chat(messages)
        return answer or "Error: Failed to generate response."
    
    @staticmethod
    def _generate_streaming_answer(messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Generate streaming answer"""
        logger.info(f"Starting streaming answer generation with {len(messages)} messages")
        try:
            chunk_count = 0
            for chunk in ollama_service.stream_chat(messages):
                chunk_count += 1
                logger.debug(f"Yielding chunk {chunk_count}: {chunk[:50]}...")
                yield chunk
            logger.info(f"Streaming completed with {chunk_count} chunks")
        except Exception as e:
            logger.error(f"Error in streaming answer generation: {e}")
            yield f"Error: {str(e)}"

class ResponseFormatter:
    """Formats API responses"""
    
    @staticmethod
    def format_streaming_response(content_generator: Generator[str, None, None]) -> Response:
        """Format streaming response for Flask with proper SSE headers"""
        def generate():
            try:
                # Send initial connection confirmation
                initial_data = "data: " + json.dumps({"status": "connected"}) + "\n\n"
                logger.debug("Sending initial SSE connection confirmation")
                yield initial_data
                
                chunk_count = 0
                for content in content_generator:
                    if content:  # Only send non-empty content
                        chunk_count += 1
                        response_chunk = {"answer": content}
                        sse_data = "data: " + json.dumps(response_chunk) + "\n\n"
                        logger.debug(f"Sending SSE chunk {chunk_count}: {content[:30]}...")
                        yield sse_data
                
                # Send completion signal
                completion_data = "data: " + json.dumps({"status": "complete"}) + "\n\n"
                logger.info(f"Sending SSE completion signal after {chunk_count} chunks")
                yield completion_data
                
            except Exception as e:
                logger.error(f"Error in streaming response: {e}")
                error_response = {"error": f"Streaming error: {str(e)}"}
                error_data = "data: " + json.dumps(error_response) + "\n\n"
                yield error_data
        
        return Response(
            generate(),
            mimetype="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable Nginx buffering
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
    
    @staticmethod
    def format_direct_response(answer: str) -> Response:
        """Format direct response for Flask"""
        def generate():
            # Send connection confirmation first
            yield "data: " + json.dumps({"status": "connected"}) + "\n\n"
            # Send the answer
            response_chunk = {"answer": answer}
            yield "data: " + json.dumps(response_chunk) + "\n\n"
            # Send completion signal
            yield "data: " + json.dumps({"status": "complete"}) + "\n\n"
        
        return Response(
            generate(),
            mimetype="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    @staticmethod
    def format_error_response(error_message: str, status_code: int = 400) -> tuple:
        """Format error response"""
        return {"error": error_message}, status_code
    
    @staticmethod
    def format_stats_response(stats: Dict[str, Any]) -> Dict[str, Any]:
        """Format statistics response"""
        return {
            "status": "success",
            "timestamp": time.time(),
            "stats": stats
        }

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self):
        self.request_count = 0
        self.total_response_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        self._lock = threading.Lock()
    
    def record_request(self, response_time: float, cache_hit: bool = False, error: bool = False):
        """Record request metrics"""
        with self._lock:
            self.request_count += 1
            self.total_response_time += response_time
            
            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            
            if error:
                self.errors += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self._lock:
            if self.request_count == 0:
                return {
                    "requests": 0,
                    "average_response_time": 0.0,
                    "cache_hit_rate": 0.0,
                    "error_rate": 0.0
                }
            
            return {
                "requests": self.request_count,
                "average_response_time": self.total_response_time / self.request_count,
                "cache_hit_rate": self.cache_hits / self.request_count,
                "error_rate": self.errors / self.request_count,
                "total_cache_hits": self.cache_hits,
                "total_cache_misses": self.cache_misses,
                "total_errors": self.errors
            }
    
    def reset_stats(self):
        """Reset all statistics"""
        with self._lock:
            self.request_count = 0
            self.total_response_time = 0.0
            self.cache_hits = 0
            self.cache_misses = 0
            self.errors = 0
        
        logger.info("Performance statistics reset")

def timing_decorator(func: Callable) -> Callable:
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.4f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f}s: {e}")
            raise
    return wrapper

def validate_query(query: str) -> tuple[bool, Optional[str]]:
    """Validate user query"""
    if not query:
        return False, "Query cannot be empty"
    
    if not query.strip():
        return False, "Query cannot be only whitespace"
    
    if len(query) > Config.MAX_QUERY_LENGTH:
        return False, f"Query too long. Maximum length is {Config.MAX_QUERY_LENGTH} characters"
    
    # Check for potentially malicious content
    suspicious_patterns = ['<script', '<?php', 'javascript:', 'eval(', 'exec(']
    query_lower = query.lower()
    
    for pattern in suspicious_patterns:
        if pattern in query_lower:
            return False, "Query contains potentially malicious content"
    
    return True, None

def sanitize_text(text: str) -> str:
    """Sanitize text input"""
    if not text:
        return ""
    
    # Remove control characters but keep newlines and tabs
    sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    # Limit length
    if len(sanitized) > Config.MAX_QUERY_LENGTH:
        sanitized = sanitized[:Config.MAX_QUERY_LENGTH]
    
    return sanitized.strip()

def log_request_info(request_data: Dict[str, Any], user_ip: str = None):
    """Log incoming request information"""
    logger.info(f"Incoming request from {user_ip or 'unknown'}: {request_data}")

def create_health_check_response() -> Dict[str, Any]:
    """Create health check response"""
    from services.redis_service import redis_service
    from services.ollama_service import ollama_service
    from services.embedding_service import embedding_service
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "redis": redis_service.is_available(),
            "ollama": ollama_service.is_available(),
            "embeddings": embedding_service.is_available()
        },
        "config": {
            "streaming_enabled": Config.ENABLE_STREAMING,
            "model": Config.OLLAMA_MODEL,
            "top_k": Config.TOP_K,
            "min_similarity": Config.MIN_SIMILARITY
        }
    }

# Global performance monitor
performance_monitor = PerformanceMonitor()