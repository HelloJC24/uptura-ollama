"""
Redis service for caching and data storage
"""
import redis
import json
import logging
import time
from typing import Optional, Any, Dict
from config import Config

logger = logging.getLogger(__name__)

class RedisService:
    """Redis service for caching and document storage"""
    
    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self.is_connected = False
        self._connect()
    
    def _connect(self) -> None:
        """Establish Redis connection with retry logic"""
        logger.info(f"Connecting to Redis at {Config.REDIS_HOST}:{Config.REDIS_PORT}")
        
        try:
            self.client = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                db=Config.REDIS_DB,
                password=Config.REDIS_PASSWORD,
                socket_timeout=Config.REDIS_TIMEOUT,
                socket_connect_timeout=Config.REDIS_TIMEOUT,
                decode_responses=False,  # We'll handle encoding manually
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.client.ping()
            self.is_connected = True
            logger.info("Successfully connected to Redis")
            
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.is_connected = False
            self.client = None
        except Exception as e:
            logger.error(f"Unexpected error connecting to Redis: {e}")
            self.is_connected = False
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Redis is available"""
        if not self.client:
            return False
        
        try:
            self.client.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            self.is_connected = False
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis with error handling"""
        if not self.is_available():
            logger.warning("Redis not available for GET operation")
            return None
        
        try:
            logger.debug(f"Getting key from Redis: {key}")
            value = self.client.get(key)
            if value:
                return json.loads(value.decode('utf-8'))
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON for key {key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting key {key} from Redis: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis with TTL and error handling"""
        if not self.is_available():
            logger.warning("Redis not available for SET operation")
            return False
        
        try:
            logger.debug(f"Setting key in Redis: {key}")
            serialized_value = json.dumps(value).encode('utf-8')
            
            if ttl:
                result = self.client.setex(key, ttl, serialized_value)
            else:
                result = self.client.set(key, serialized_value)
            
            return bool(result)
        except Exception as e:
            logger.error(f"Error setting key {key} in Redis: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        if not self.is_available():
            logger.warning("Redis not available for DELETE operation")
            return False
        
        try:
            logger.debug(f"Deleting key from Redis: {key}")
            result = self.client.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Error deleting key {key} from Redis: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        if not self.is_available():
            return False
        
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Error checking existence of key {key}: {e}")
            return False
    
    def get_keys(self, pattern: str) -> list:
        """Get all keys matching pattern"""
        if not self.is_available():
            return []
        
        try:
            logger.debug(f"Getting keys with pattern: {pattern}")
            keys = self.client.keys(pattern)
            return [key.decode('utf-8') for key in keys]
        except Exception as e:
            logger.error(f"Error getting keys with pattern {pattern}: {e}")
            return []
    
    def clear_cache(self, pattern: str = "*") -> int:
        """Clear cache entries matching pattern"""
        if not self.is_available():
            logger.warning("Redis not available for cache clearing")
            return 0
        
        try:
            keys = self.get_keys(pattern)
            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics"""
        if not self.is_available():
            return {"available": False}
        
        try:
            info = self.client.info()
            return {
                "available": True,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "Unknown"),
                "uptime": info.get("uptime_in_seconds", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {"available": False, "error": str(e)}

# Global Redis service instance
redis_service = RedisService()