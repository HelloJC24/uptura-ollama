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
    
    def process_document(self, url: str) -> List[DocumentChunk]:
        """Process a single document: fetch, extract, chunk, and embed"""
        logger.info(f"Processing document: {url}")
        
        if url in self.processed_urls:
            logger.info(f"Document {url} already processed, skipping")
            return []
        
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
    
    def _store_chunks_in_redis(self, chunks: List[DocumentChunk]) -> None:
        """Store document chunks in Redis"""
        if not redis_service.is_available():
            logger.warning("Redis not available, skipping chunk storage")
            return
        
        for chunk in chunks:
            key = f"doc_chunk:{chunk.source_url}:{chunk.chunk_index}"
            success = redis_service.set(key, chunk.to_dict(), ttl=Config.CACHE_TTL)
            
            if not success:
                logger.warning(f"Failed to store chunk in Redis: {key}")
    
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
    
    def process_all_documents(self, urls: List[str] = None) -> None:
        """Process all configured documents"""
        urls = urls or Config.DOCUMENT_URLS
        
        logger.info(f"Processing {len(urls)} documents")
        start_time = time.time()
        
        # Load existing chunks from Redis
        self.chunks = self.load_chunks_from_redis()
        
        # Process each URL
        for url in urls:
            try:
                new_chunks = self.process_document(url)
                logger.info(f"Processed {url}: {len(new_chunks)} new chunks")
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
        
        total_time = time.time() - start_time
        logger.info(f"Document processing completed in {total_time:.2f}s. Total chunks: {len(self.chunks)}")
    
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