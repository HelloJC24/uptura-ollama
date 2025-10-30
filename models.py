"""
Data models for the RAG system
"""
import logging
import time
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from config import Config
from services.embedding_service import embedding_service
from services.redis_service import redis_service

logger = logging.getLogger(__name__)

@dataclass
class ConversationMessage:
    """Represents a single message in a conversation"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMessage':
        """Create from dictionary"""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
        )

@dataclass
class UserConversation:
    """Represents a user's conversation history"""
    user_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation"""
        message = ConversationMessage(role=role, content=content)
        self.messages.append(message)
        self.last_updated = datetime.now()
        
        # Keep only the last N messages
        max_history = Config.MAX_CONVERSATION_HISTORY
        if len(self.messages) > max_history:
            self.messages = self.messages[-max_history:]
    
    def get_context_messages(self) -> List[Dict[str, str]]:
        """Get messages in format suitable for LLM context"""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]
    
    def get_recent_messages(self, count: int = None) -> List[ConversationMessage]:
        """Get the most recent messages"""
        count = count or Config.MAX_CONVERSATION_HISTORY
        return self.messages[-count:] if self.messages else []
    
    def clear_history(self) -> None:
        """Clear all conversation history"""
        self.messages.clear()
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "user_id": self.user_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserConversation':
        """Create from dictionary"""
        conversation = cls(
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            last_updated=datetime.fromisoformat(data.get("last_updated", datetime.now().isoformat()))
        )
        
        # Load messages
        for msg_data in data.get("messages", []):
            message = ConversationMessage.from_dict(msg_data)
            conversation.messages.append(message)
        
        return conversation
    
    def is_expired(self) -> bool:
        """Check if conversation has expired"""
        ttl_seconds = Config.CONVERSATION_TTL
        expiry_time = self.last_updated + timedelta(seconds=ttl_seconds)
        return datetime.now() > expiry_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        user_messages = [msg for msg in self.messages if msg.role == 'user']
        assistant_messages = [msg for msg in self.messages if msg.role == 'assistant']
        
        return {
            "total_messages": len(self.messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "is_expired": self.is_expired()
        }

@dataclass
class DocumentChunk:
    """Represents a chunk of a document with its embedding"""
    text: str
    embedding: List[float]
    source_url: str
    chunk_index: int
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "text": self.text,
            "embedding": self.embedding,
            "source_url": self.source_url,
            "chunk_index": self.chunk_index,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Create from dictionary"""
        return cls(
            text=data["text"],
            embedding=data["embedding"],
            source_url=data["source_url"],
            chunk_index=data["chunk_index"],
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        )

@dataclass
class QueryResult:
    """Represents the result of a query"""
    query: str
    answer: str
    relevant_chunks: List[DocumentChunk]
    processing_time: float
    cache_hit: bool = False
    streaming: bool = False
    created_at: datetime = field(default_factory=datetime.now)

class DocumentProcessor:
    """Handles document fetching, chunking, and embedding"""
    
    def __init__(self):
        self.processed_urls: set = set()
        self.chunks: List[DocumentChunk] = []
    
    def fetch_document(self, url: str, timeout: int = 10) -> Optional[str]:
        """Fetch document content from URL"""
        logger.info(f"Fetching document from: {url}")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            start_time = time.time()
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            
            fetch_time = time.time() - start_time
            logger.info(f"Fetched {len(response.text)} characters from {url} in {fetch_time:.2f}s")
            
            return response.text
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return None
    
    def extract_text(self, html_content: str) -> str:
        """Extract clean text from HTML content"""
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator="\n", strip=True)
            
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)
            
            logger.debug(f"Extracted {len(clean_text)} characters of clean text")
            return clean_text
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""
    
    def create_chunks(self, text: str, url: str) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        chunk_size = Config.CHUNK_SIZE
        overlap = Config.CHUNK_OVERLAP
        
        logger.debug(f"Creating chunks from {len(words)} words (chunk_size={chunk_size}, overlap={overlap})")
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Skip very short chunks
            if len(chunk_text.strip()) < 50:
                continue
                
            chunks.append(chunk_text)
        
        logger.info(f"Created {len(chunks)} chunks from {url}")
        return chunks
    
    async def process_document(self, url: str) -> List[DocumentChunk]:
        """Process a single document: fetch with enhanced processing, extract, chunk, and embed"""
        logger.info(f"Processing document with enhanced fetching: {url}")
        
        if url in self.processed_urls:
            logger.info(f"Document {url} already processed, skipping")
            return []
        
        try:
            # Use enhanced document processor
            from enhanced_document_processor import enhanced_document_processor
            
            # Fetch enhanced content
            result = await enhanced_document_processor.fetch_document_content(url)
            
            if result.get('error'):
                logger.error(f"Error fetching enhanced content from {url}: {result['error']}")
                return []
            
            # Get combined content from all sources
            text = enhanced_document_processor.get_all_content(result)
            
            if not text or not text.strip():
                logger.warning(f"No text extracted from {url} with enhanced processing")
                # Fallback to static method
                html_content = self.fetch_document(url)
                if html_content:
                    text = self.extract_text(html_content)
                else:
                    return []
            
            logger.info(f"Enhanced processing extracted {len(text)} characters from {url}")
            logger.info(f"Method used: {result.get('method', 'fallback')}")
            if result.get('api_data'):
                logger.info(f"Found {len(result['api_data'])} API endpoints with data")
            
            # Create chunks
            chunk_texts = self.create_chunks(text, url)
            if not chunk_texts:
                logger.warning(f"No chunks created from {url}")
                return []
            
            # Generate embeddings
            embeddings = embedding_service.get_embeddings_batch(chunk_texts)
            
            # Create DocumentChunk objects
            document_chunks = []
            for i, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings)):
                if embedding:  # Only add chunks with valid embeddings
                    chunk = DocumentChunk(
                        text=chunk_text,
                        embedding=embedding,
                        source_url=url,
                        chunk_index=i
                    )
                    document_chunks.append(chunk)
            
            # Store in Redis with enhanced metadata
            self._store_chunks_in_redis(document_chunks, metadata=result)
            
            # Mark as processed
            self.processed_urls.add(url)
            self.chunks.extend(document_chunks)
            
        except Exception as e:
            logger.error(f"Error in enhanced document processing for {url}: {e}")
            # Fallback to standard processing
            logger.info(f"Falling back to standard processing for {url}")
            return await self._process_document_fallback(url)
        
        logger.info(f"Successfully processed {url} with {len(document_chunks)} chunks using enhanced method")
        return document_chunks
    
    async def _process_document_fallback(self, url: str) -> List[DocumentChunk]:
        """Fallback to standard document processing"""
        # Fetch document
        html_content = self.fetch_document(url)
        if not html_content:
            return []
        
        # Extract text
        text = self.extract_text(html_content)
        if not text:
            logger.warning(f"No text extracted from {url}")
            return []
        
        # Create chunks
        chunk_texts = self.create_chunks(text, url)
        if not chunk_texts:
            logger.warning(f"No chunks created from {url}")
            return []
        
        # Generate embeddings
        embeddings = embedding_service.get_embeddings_batch(chunk_texts)
        
        # Create DocumentChunk objects
        document_chunks = []
        for i, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings)):
            if embedding:  # Only add chunks with valid embeddings
                chunk = DocumentChunk(
                    text=chunk_text,
                    embedding=embedding,
                    source_url=url,
                    chunk_index=i
                )
                document_chunks.append(chunk)
        
        # Store in Redis
        self._store_chunks_in_redis(document_chunks)
        
        # Mark as processed
        self.processed_urls.add(url)
        self.chunks.extend(document_chunks)
        
        logger.info(f"Successfully processed {len(document_chunks)} chunks from {url}")
        return document_chunks
    
    def _store_chunks_in_redis(self, chunks: List[DocumentChunk], metadata: Dict[str, Any] = None) -> None:
        """Store document chunks in Redis with optional enhanced metadata"""
        if not redis_service.is_available():
            logger.warning("Redis not available, skipping chunk storage")
            return
        
        for chunk in chunks:
            key = f"doc_chunk:{chunk.source_url}:{chunk.chunk_index}"
            chunk_data = chunk.to_dict()
            
            # Add enhanced metadata if available
            if metadata:
                chunk_data['enhanced_metadata'] = {
                    'fetch_method': metadata.get('method', 'static'),
                    'has_dynamic_content': bool(metadata.get('dynamic_content')),
                    'api_endpoints_found': len(metadata.get('api_data', [])),
                    'fetch_timestamp': time.time()
                }
            
            success = redis_service.set(key, chunk_data, ttl=Config.CACHE_TTL)
            
            if not success:
                logger.warning(f"Failed to store chunk in Redis: {key}")
        
        # Store processing metadata separately
        if metadata and chunks:
            metadata_key = f"doc_metadata:{chunks[0].source_url}"
            metadata_summary = {
                'url': metadata.get('url'),
                'method': metadata.get('method', 'static'),
                'has_dynamic_content': bool(metadata.get('dynamic_content')),
                'api_endpoints': [api['url'] for api in metadata.get('api_data', [])],
                'processed_timestamp': time.time(),
                'chunk_count': len(chunks)
            }
            redis_service.set(metadata_key, metadata_summary, ttl=Config.CACHE_TTL)
    
    def load_chunks_from_redis(self) -> List[DocumentChunk]:
        """Load existing chunks from Redis"""
        if not redis_service.is_available():
            logger.warning("Redis not available, cannot load chunks")
            return []
        
        logger.info("Loading existing chunks from Redis")
        
        chunk_keys = redis_service.get_keys("doc_chunk:*")
        loaded_chunks = []
        
        for key in chunk_keys:
            chunk_data = redis_service.get(key)
            if chunk_data:
                try:
                    chunk = DocumentChunk.from_dict(chunk_data)
                    loaded_chunks.append(chunk)
                except Exception as e:
                    logger.error(f"Error loading chunk {key}: {e}")
        
        logger.info(f"Loaded {len(loaded_chunks)} chunks from Redis")
        return loaded_chunks
    
    async def process_all_documents(self, urls: List[str] = None) -> None:
        """Process all configured documents with enhanced fetching"""
        urls = urls or Config.DOCUMENT_URLS
        
        logger.info(f"Processing {len(urls)} documents with enhanced fetching")
        start_time = time.time()
        
        # Load existing chunks from Redis
        self.chunks = self.load_chunks_from_redis()
        
        # Process each URL with enhanced fetching
        for url in urls:
            try:
                new_chunks = await self.process_document(url)
                logger.info(f"Processed {url}: {len(new_chunks)} new chunks")
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
        
        total_time = time.time() - start_time
        logger.info(f"Enhanced document processing completed in {total_time:.2f}s. Total chunks: {len(self.chunks)}")
    
    def process_all_documents_sync(self, urls: List[str] = None) -> None:
        """Synchronous wrapper for document processing"""
        import asyncio
        
        try:
            # Try to run in existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new event loop in a thread
                import concurrent.futures
                import threading
                
                def run_async():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self.process_all_documents(urls))
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    future.result()
            else:
                # Run in current event loop
                loop.run_until_complete(self.process_all_documents(urls))
        except RuntimeError:
            # No event loop, create one
            asyncio.run(self.process_all_documents(urls))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get document processor statistics"""
        return {
            "total_chunks": len(self.chunks),
            "processed_urls": len(self.processed_urls),
            "urls": list(self.processed_urls),
            "chunk_sources": {url: sum(1 for chunk in self.chunks if chunk.source_url == url) 
                            for url in self.processed_urls}
        }

class LRUCache:
    """LRU Cache for query results"""
    
    def __init__(self, max_size: int = None):
        self.max_size = max_size or Config.MAX_CACHE_SIZE
        self.cache: Dict[str, Any] = {}
        self.access_order: List[str] = []
        self.ttl_map: Dict[str, datetime] = {}
    
    def _make_key(self, query: str) -> str:
        """Create cache key from query"""
        return hashlib.sha256(query.encode()).hexdigest()
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries"""
        now = datetime.now()
        expired_keys = [
            key for key, expiry in self.ttl_map.items() 
            if expiry < now
        ]
        
        for key in expired_keys:
            self._remove_key(key)
    
    def _remove_key(self, key: str) -> None:
        """Remove key from all data structures"""
        self.cache.pop(key, None)
        self.ttl_map.pop(key, None)
        if key in self.access_order:
            self.access_order.remove(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if self.access_order:
            lru_key = self.access_order[0]
            self._remove_key(lru_key)
    
    def get(self, query: str) -> Optional[str]:
        """Get cached result for query"""
        key = self._make_key(query)
        
        # Cleanup expired entries
        self._cleanup_expired()
        
        if key in self.cache:
            # Move to end (most recent)
            self.access_order.remove(key)
            self.access_order.append(key)
            
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return self.cache[key]
        
        logger.debug(f"Cache miss for query: {query[:50]}...")
        return None
    
    def set(self, query: str, result: str, ttl_seconds: int = None) -> None:
        """Cache query result"""
        key = self._make_key(query)
        ttl_seconds = ttl_seconds or Config.CACHE_TTL
        
        # Remove if already exists
        if key in self.cache:
            self._remove_key(key)
        
        # Evict if at capacity
        while len(self.cache) >= self.max_size:
            self._evict_lru()
        
        # Add new entry
        self.cache[key] = result
        self.access_order.append(key)
        self.ttl_map[key] = datetime.now() + timedelta(seconds=ttl_seconds)
        
        logger.debug(f"Cached result for query: {query[:50]}...")
    
    def clear(self) -> None:
        """Clear all cached entries"""
        self.cache.clear()
        self.access_order.clear()
        self.ttl_map.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        self._cleanup_expired()
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": getattr(self, 'hits', 0) / max(getattr(self, 'total_requests', 1), 1),
            "oldest_entry": min(self.ttl_map.values()) if self.ttl_map else None,
            "newest_entry": max(self.ttl_map.values()) if self.ttl_map else None
        }

# Global instances
document_processor = DocumentProcessor()
query_cache = LRUCache()