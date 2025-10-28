"""
Ollama service for LLM interactions
"""
import logging
import time
import threading
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
        self.is_warmed_up = False
        self.warmup_in_progress = False
        self._connect()
        # Start warmup in background thread to not block initialization
        self._async_warmup()
    
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
    
    def _async_warmup(self) -> None:
        """Start warmup in background thread"""
        if not self.is_available():
            logger.warning("Ollama not available for async warmup")
            return
        
        def warmup_thread():
            self.warmup_in_progress = True
            logger.info(f"Starting background warmup for model: {self.model_name}")
            
            try:
                start_time = time.time()
                # Use a very simple prompt for faster warmup
                response = self.client.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "Hi"}],
                    options={
                        "temperature": 0.1,
                        "num_predict": 5,  # Limit to just a few tokens
                        "num_ctx": 512     # Smaller context for warmup
                    }
                )
                
                warmup_time = time.time() - start_time
                self.is_warmed_up = True
                logger.info(f"Background model warmup completed in {warmup_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Background model warmup failed: {e}")
            finally:
                self.warmup_in_progress = False
        
        # Start warmup in daemon thread
        warmup_thread = threading.Thread(target=warmup_thread, daemon=True)
        warmup_thread.start()
    
    def ensure_warmup(self) -> None:
        """Ensure model is warmed up before processing requests"""
        if self.is_warmed_up:
            return
        
        if self.warmup_in_progress:
            logger.info("Waiting for background warmup to complete...")
            # Wait up to 10 seconds for warmup to complete
            max_wait = 10
            wait_time = 0
            while self.warmup_in_progress and wait_time < max_wait:
                time.sleep(0.5)
                wait_time += 0.5
            
            if self.is_warmed_up:
                logger.info("Background warmup completed successfully")
                return
        
        # If warmup hasn't completed or failed, do a quick inline warmup
        logger.info("Performing quick inline warmup...")
        try:
            start_time = time.time()
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "OK"}],
                options={
                    "temperature": 0.1,
                    "num_predict": 1,  # Just one token
                    "num_ctx": 256     # Minimal context
                }
            )
            warmup_time = time.time() - start_time
            self.is_warmed_up = True
            logger.info(f"Quick inline warmup completed in {warmup_time:.2f}s")
        except Exception as e:
            logger.error(f"Quick inline warmup failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama service is available"""
        if not self.client:
            return False
        
        try:
            # Try to list models as a health check
            models_response = self.client.list()
            # Just check if we got a response, don't parse it here
            return models_response is not None
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            self.is_connected = False
            return False
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Optional[str]:
        """Send chat request to Ollama"""
        if not self.is_available():
            logger.error("Ollama not available for chat")
            return None
        
        # Ensure model is warmed up before processing
        self.ensure_warmup()
        
        try:
            logger.debug(f"Sending chat request to Ollama with {len(messages)} messages")
            start_time = time.time()
            
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_ctx": 4096,        # Adequate context window
                    "num_predict": 2048,    # Limit max tokens for faster response
                    "repeat_penalty": 1.1,
                    "tfs_z": 1.0
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
        
        # Ensure model is warmed up before processing
        self.ensure_warmup()
        
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
                    "top_k": 40,
                    "num_predict": 2048,    # Limit max tokens for faster response
                    "num_ctx": 4096,        # Context window
                    "repeat_penalty": 1.1,
                    "tfs_z": 1.0
                }
            )
            
            # Yield chunks as they arrive
            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    if content:  # Only yield non-empty content
                        total_chunks += 1
                        yield content
                        
                        # Minimal delay to prevent overwhelming but keep speed
                        time.sleep(0.005)  # Reduced to 5ms delay
            
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
            logger.debug(f"Raw models response: {models_response}")
            
            # Handle different response structures
            if isinstance(models_response, dict):
                if 'models' in models_response:
                    models_list = models_response['models']
                else:
                    models_list = models_response
            else:
                models_list = models_response
            
            # Extract model names safely
            models = []
            for model in models_list:
                if isinstance(model, dict):
                    # Try different possible keys for model name
                    model_name = model.get('name') or model.get('model') or model.get('id') or str(model)
                    models.append(model_name)
                elif isinstance(model, str):
                    models.append(model)
                else:
                    logger.warning(f"Unexpected model format: {type(model)} - {model}")
                    models.append(str(model))
            
            logger.debug(f"Available models: {models}")
            return models
            
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            logger.debug(f"Exception details:", exc_info=True)
            return []
    
    def check_model_exists(self, model_name: str = None) -> bool:
        """Check if specified model exists"""
        model_to_check = model_name or self.model_name
        available_models = self.get_models()
        
        # If we couldn't get models list, try a different approach
        if not available_models:
            logger.warning("Could not retrieve models list, trying direct model check")
            try:
                # Try to use the model directly with a simple prompt
                test_response = self.client.chat(
                    model=model_to_check,
                    messages=[{"role": "user", "content": "test"}],
                    options={"max_tokens": 1}
                )
                return test_response is not None
            except Exception as e:
                logger.error(f"Direct model check failed for {model_to_check}: {e}")
                return False
        
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