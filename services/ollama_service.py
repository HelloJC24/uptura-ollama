"""
Ollama service for LLM interactions
"""
import logging
import time
from typing import Dict, Any, Generator, Optional, List
from ollama import Client
from config import Config

logger = logging.getLogger(__name__)

class OllamaService:
    """Ollama service for LLM interactions"""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.is_connected = False
        self.model_name = Config.OLLAMA_MODEL
        self._connect()
        self._warmup()
    
    def _connect(self) -> None:
        """Initialize Ollama client"""
        logger.info(f"Connecting to Ollama at {Config.OLLAMA_HOST}")
        
        try:
            self.client = Client(host=Config.OLLAMA_HOST)
            self.is_connected = True
            logger.info("Successfully initialized Ollama client")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            self.is_connected = False
            self.client = None
    
    def _warmup(self) -> None:
        """Warm up the model with a simple query"""
        if not self.is_available():
            logger.warning("Ollama not available for warmup")
            return
        
        logger.info(f"Warming up Ollama model: {self.model_name}")
        
        try:
            start_time = time.time()
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                options={"temperature": 0.1}
            )
            
            warmup_time = time.time() - start_time
            logger.info(f"Model warmup completed in {warmup_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Model warmup failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama service is available"""
        if not self.client:
            return False
        
        try:
            # Try to list models as a health check
            models = self.client.list()
            return True
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            self.is_connected = False
            return False
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Optional[str]:
        """Send chat request to Ollama"""
        if not self.is_available():
            logger.error("Ollama not available for chat")
            return None
        
        try:
            logger.debug(f"Sending chat request to Ollama with {len(messages)} messages")
            start_time = time.time()
            
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40
                }
            )
            
            response_time = time.time() - start_time
            answer = response.get('message', {}).get('content', '')
            
            logger.info(f"Chat response received in {response_time:.2f}s, length: {len(answer)}")
            return answer
            
        except Exception as e:
            logger.error(f"Error in Ollama chat: {e}")
            return None
    
    def stream_chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Generator[str, None, None]:
        """Stream chat response from Ollama"""
        if not self.is_available():
            logger.error("Ollama not available for streaming chat")
            yield "Error: Ollama service unavailable"
            return
        
        try:
            logger.debug(f"Starting streaming chat with Ollama")
            start_time = time.time()
            total_chunks = 0
            
            stream = self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                options={
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40
                }
            )
            
            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    if content:  # Only yield non-empty content
                        total_chunks += 1
                        yield content
            
            response_time = time.time() - start_time
            logger.info(f"Streaming completed in {response_time:.2f}s, {total_chunks} chunks")
            
        except Exception as e:
            logger.error(f"Error in Ollama streaming: {e}")
            yield f"Error: {str(e)}"
    
    def get_models(self) -> List[str]:
        """Get list of available models"""
        if not self.is_available():
            return []
        
        try:
            models_response = self.client.list()
            models = [model['name'] for model in models_response.get('models', [])]
            logger.debug(f"Available models: {models}")
            return models
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []
    
    def check_model_exists(self, model_name: str = None) -> bool:
        """Check if specified model exists"""
        model_to_check = model_name or self.model_name
        available_models = self.get_models()
        
        # Check exact match or partial match
        for model in available_models:
            if model == model_to_check or model.startswith(model_to_check):
                return True
        
        logger.warning(f"Model {model_to_check} not found in available models: {available_models}")
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Ollama service statistics"""
        return {
            "available": self.is_available(),
            "host": Config.OLLAMA_HOST,
            "model": self.model_name,
            "model_exists": self.check_model_exists(),
            "available_models": self.get_models() if self.is_available() else []
        }

# Global Ollama service instance
ollama_service = OllamaService()