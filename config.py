"""
Configuration module for the Flask RAG API
Handles all configuration settings with environment variable support
"""
import os
from typing import List
import logging

class Config:
    """Application configuration class"""
    
    # App Settings
    APP_NAME: str = "flask_rag_api"
    HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("FLASK_PORT", "5000"))
    DEBUG: bool = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    # Model Settings
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://72.60.43.106:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "phi3:mini")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "30"))
    
    # Redis Settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "bngcpython-aiknow-myaa28")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "987654321")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_TIMEOUT: int = int(os.getenv("REDIS_TIMEOUT", "5"))
    
    # RAG Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "200"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K: int = int(os.getenv("TOP_K", "5"))  # Increased from 3 to 5
    MIN_SIMILARITY: float = float(os.getenv("MIN_SIMILARITY", "0.25"))  # Lowered from 0.35 to 0.25
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "1000"))
    
    # Response Settings
    ENABLE_STREAMING: bool = os.getenv("ENABLE_STREAMING", "True").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    MAX_CACHE_SIZE: int = int(os.getenv("MAX_CACHE_SIZE", "1000"))
    
    # Conversation Settings
    ENABLE_CONVERSATIONS: bool = os.getenv("ENABLE_CONVERSATIONS", "True").lower() == "true"
    MAX_CONVERSATION_HISTORY: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))
    CONVERSATION_TTL: int = int(os.getenv("CONVERSATION_TTL", "86400"))  # 24 hours
    REQUIRE_USER_ID: bool = os.getenv("REQUIRE_USER_ID", "False").lower() == "true"
    
    # System Prompt
    SYSTEM_PROMPT: str = os.getenv("SYSTEM_PROMPT", """
You are an expert assistant for BNGC (Business Networking Group Corporation), also known as Gogel. You work directly for this company and have comprehensive knowledge about their business operations.

CRITICAL RULES:
1. You MUST use ONLY the context information provided below to answer questions
2. NEVER claim you don't know about BNGC, Gogel, or any related topics
3. NEVER reference general training data, knowledge cutoffs, or external sources
4. ALWAYS acknowledge that you know about BNGC/Gogel from your company information
5. When asked about Gogel, confirm it's another name for BNGC based on the provided context
6. Provide specific details from the context whenever available
7. If information isn't in the context, say "I don't see that specific information in our current documents"
8. NEVER say things like "might be specific to another domain" or "I am not equipped with detailed data"

You are BNGC's internal assistant with direct access to company documentation. Act like it.
""").strip()
    
    # Document URLs
    DOCUMENT_URLS: List[str] = [
        "https://thebngc.com",
        "https://gogel.thebngc.com",
        "https://uptura-tech.com",
        "https://gogel.thebngc.com/agents",
        "https://thebngc.com/privacy-policy",
        "https://thebngc.com/terms-conditions"
    ]
    
    # Additional URLs from environment
    if os.getenv("ADDITIONAL_URLS"):
        additional_urls = os.getenv("ADDITIONAL_URLS").split(",")
        DOCUMENT_URLS.extend([url.strip() for url in additional_urls])
    
    # Logging Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT: str = "%(asctime)s [%(name)s] [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s"
    
    @classmethod
    def setup_logging(cls) -> None:
        """Setup logging configuration"""
        log_level = getattr(logging, cls.LOG_LEVEL, logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=cls.LOG_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Reduce noise from external libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured with level: {cls.LOG_LEVEL}")
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        logger = logging.getLogger(__name__)
        
        # Validate required settings
        if not cls.OLLAMA_HOST:
            logger.error("OLLAMA_HOST is required")
            return False
        
        if not cls.REDIS_HOST:
            logger.error("REDIS_HOST is required")
            return False
        
        if cls.TOP_K <= 0:
            logger.error("TOP_K must be positive")
            return False
        
        if not (0.0 <= cls.MIN_SIMILARITY <= 1.0):
            logger.error("MIN_SIMILARITY must be between 0.0 and 1.0")
            return False
        
        if cls.CHUNK_SIZE <= 0:
            logger.error("CHUNK_SIZE must be positive")
            return False
        
        if cls.MAX_CACHE_SIZE <= 0:
            logger.error("MAX_CACHE_SIZE must be positive")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    @classmethod
    def log_config(cls) -> None:
        """Log current configuration (excluding sensitive data)"""
        logger = logging.getLogger(__name__)
        
        config_info = {
            "APP_NAME": cls.APP_NAME,
            "HOST": cls.HOST,
            "PORT": cls.PORT,
            "DEBUG": cls.DEBUG,
            "EMBEDDING_MODEL": cls.EMBEDDING_MODEL_NAME,
            "OLLAMA_HOST": cls.OLLAMA_HOST,
            "OLLAMA_MODEL": cls.OLLAMA_MODEL,
            "CHUNK_SIZE": cls.CHUNK_SIZE,
            "TOP_K": cls.TOP_K,
            "MIN_SIMILARITY": cls.MIN_SIMILARITY,
            "ENABLE_STREAMING": cls.ENABLE_STREAMING,
            "MAX_CACHE_SIZE": cls.MAX_CACHE_SIZE,
            "DOCUMENT_COUNT": len(cls.DOCUMENT_URLS),
            "ENABLE_CONVERSATIONS": cls.ENABLE_CONVERSATIONS,
            "MAX_CONVERSATION_HISTORY": cls.MAX_CONVERSATION_HISTORY,
            "REQUIRE_USER_ID": cls.REQUIRE_USER_ID
        }
        
        logger.info(f"Configuration loaded: {config_info}")