"""
Conversation service for managing user chat histories
"""
import logging
import time
from typing import Dict, List, Optional, Any
from models import UserConversation, ConversationMessage
from services.redis_service import redis_service
from config import Config

logger = logging.getLogger(__name__)

class ConversationService:
    """Service for managing user conversations"""
    
    def __init__(self):
        self.local_cache: Dict[str, UserConversation] = {}
        self.cache_timeout = 300  # 5 minutes local cache
        self.last_cleanup = time.time()
    
    def _get_redis_key(self, user_id: str) -> str:
        """Generate Redis key for user conversation"""
        return f"conversation:{user_id}"
    
    def _cleanup_local_cache(self) -> None:
        """Clean up expired conversations from local cache"""
        current_time = time.time()
        
        # Only cleanup every 5 minutes
        if current_time - self.last_cleanup < 300:
            return
        
        expired_users = []
        for user_id, conversation in self.local_cache.items():
            if conversation.is_expired():
                expired_users.append(user_id)
        
        for user_id in expired_users:
            del self.local_cache[user_id]
            logger.debug(f"Cleaned up expired conversation for user: {user_id}")
        
        self.last_cleanup = current_time
        if expired_users:
            logger.info(f"Cleaned up {len(expired_users)} expired conversations from local cache")
    
    def get_conversation(self, user_id: str) -> UserConversation:
        """Get or create a conversation for a user"""
        logger.debug(f"Getting conversation for user: {user_id}")
        
        # Clean up cache periodically
        self._cleanup_local_cache()
        
        # Check local cache first
        if user_id in self.local_cache:
            conversation = self.local_cache[user_id]
            if not conversation.is_expired():
                logger.debug(f"Found conversation in local cache for user: {user_id}")
                return conversation
            else:
                # Remove expired conversation
                del self.local_cache[user_id]
        
        # Try to load from Redis
        conversation = self._load_from_redis(user_id)
        if conversation:
            if not conversation.is_expired():
                self.local_cache[user_id] = conversation
                logger.debug(f"Loaded conversation from Redis for user: {user_id}")
                return conversation
            else:
                # Delete expired conversation from Redis
                self._delete_from_redis(user_id)
        
        # Create new conversation
        conversation = UserConversation(user_id=user_id)
        self.local_cache[user_id] = conversation
        logger.info(f"Created new conversation for user: {user_id}")
        return conversation
    
    def _load_from_redis(self, user_id: str) -> Optional[UserConversation]:
        """Load conversation from Redis"""
        if not redis_service.is_available():
            logger.warning("Redis not available, cannot load conversation")
            return None
        
        try:
            redis_key = self._get_redis_key(user_id)
            conversation_data = redis_service.get(redis_key)
            
            if conversation_data:
                return UserConversation.from_dict(conversation_data)
            
        except Exception as e:
            logger.error(f"Error loading conversation for user {user_id}: {e}")
        
        return None
    
    def _save_to_redis(self, conversation: UserConversation) -> bool:
        """Save conversation to Redis"""
        if not redis_service.is_available():
            logger.warning("Redis not available, cannot save conversation")
            return False
        
        try:
            redis_key = self._get_redis_key(conversation.user_id)
            success = redis_service.set(
                redis_key, 
                conversation.to_dict(), 
                ttl=Config.CONVERSATION_TTL
            )
            
            if success:
                logger.debug(f"Saved conversation to Redis for user: {conversation.user_id}")
            else:
                logger.warning(f"Failed to save conversation to Redis for user: {conversation.user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving conversation for user {conversation.user_id}: {e}")
            return False
    
    def _delete_from_redis(self, user_id: str) -> bool:
        """Delete conversation from Redis"""
        if not redis_service.is_available():
            return False
        
        try:
            redis_key = self._get_redis_key(user_id)
            return redis_service.delete(redis_key)
        except Exception as e:
            logger.error(f"Error deleting conversation for user {user_id}: {e}")
            return False
    
    def add_message(self, user_id: str, role: str, content: str) -> bool:
        """Add a message to a user's conversation"""
        logger.debug(f"Adding {role} message for user {user_id}: {content[:50]}...")
        
        try:
            conversation = self.get_conversation(user_id)
            conversation.add_message(role, content)
            
            # Update local cache
            self.local_cache[user_id] = conversation
            
            # Save to Redis
            return self._save_to_redis(conversation)
            
        except Exception as e:
            logger.error(f"Error adding message for user {user_id}: {e}")
            return False
    
    def get_context_messages(self, user_id: str, include_system_prompt: bool = True) -> List[Dict[str, str]]:
        """Get conversation messages formatted for LLM context"""
        logger.debug(f"Getting context messages for user: {user_id}")
        
        try:
            conversation = self.get_conversation(user_id)
            context_messages = conversation.get_context_messages()
            
            # Add system prompt if requested and not already present
            if include_system_prompt:
                if not context_messages or context_messages[0].get("role") != "system":
                    system_message = {"role": "system", "content": Config.SYSTEM_PROMPT}
                    context_messages.insert(0, system_message)
            
            logger.debug(f"Retrieved {len(context_messages)} context messages for user: {user_id}")
            return context_messages
            
        except Exception as e:
            logger.error(f"Error getting context messages for user {user_id}: {e}")
            return [{"role": "system", "content": Config.SYSTEM_PROMPT}] if include_system_prompt else []
    
    def clear_conversation(self, user_id: str) -> bool:
        """Clear a user's conversation history"""
        logger.info(f"Clearing conversation for user: {user_id}")
        
        try:
            # Clear from local cache
            if user_id in self.local_cache:
                del self.local_cache[user_id]
            
            # Delete from Redis
            return self._delete_from_redis(user_id)
            
        except Exception as e:
            logger.error(f"Error clearing conversation for user {user_id}: {e}")
            return False
    
    def get_conversation_history(self, user_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        logger.debug(f"Getting conversation history for user: {user_id}")
        
        try:
            conversation = self.get_conversation(user_id)
            limit = limit or Config.MAX_CONVERSATION_HISTORY
            
            recent_messages = conversation.get_recent_messages(limit)
            return [msg.to_dict() for msg in recent_messages]
            
        except Exception as e:
            logger.error(f"Error getting conversation history for user {user_id}: {e}")
            return []
    
    def get_conversation_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a user's conversation"""
        try:
            conversation = self.get_conversation(user_id)
            return conversation.get_stats()
        except Exception as e:
            logger.error(f"Error getting conversation stats for user {user_id}: {e}")
            return {"error": str(e)}
    
    def list_active_conversations(self) -> List[str]:
        """List all users with active conversations"""
        logger.debug("Listing active conversations")
        
        active_users = []
        
        # From local cache
        for user_id in self.local_cache.keys():
            active_users.append(user_id)
        
        # From Redis
        if redis_service.is_available():
            try:
                redis_keys = redis_service.get_keys("conversation:*")
                for key in redis_keys:
                    user_id = key.replace("conversation:", "")
                    if user_id not in active_users:
                        active_users.append(user_id)
            except Exception as e:
                logger.error(f"Error listing conversations from Redis: {e}")
        
        logger.info(f"Found {len(active_users)} active conversations")
        return active_users
    
    def cleanup_expired_conversations(self) -> int:
        """Clean up all expired conversations"""
        logger.info("Starting cleanup of expired conversations")
        
        cleaned_count = 0
        
        # Clean local cache
        self._cleanup_local_cache()
        
        # Clean Redis
        if redis_service.is_available():
            try:
                active_users = self.list_active_conversations()
                for user_id in active_users:
                    conversation = self._load_from_redis(user_id)
                    if conversation and conversation.is_expired():
                        if self._delete_from_redis(user_id):
                            cleaned_count += 1
                            logger.debug(f"Cleaned up expired conversation for user: {user_id}")
            except Exception as e:
                logger.error(f"Error during Redis cleanup: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} expired conversations")
        return cleaned_count
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "local_cache_size": len(self.local_cache),
            "active_conversations": len(self.list_active_conversations()),
            "redis_available": redis_service.is_available(),
            "max_history": Config.MAX_CONVERSATION_HISTORY,
            "conversation_ttl": Config.CONVERSATION_TTL,
            "conversations_enabled": Config.ENABLE_CONVERSATIONS
        }

# Global conversation service instance
conversation_service = ConversationService()