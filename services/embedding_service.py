"""
Embedding service for text embeddings
"""
import logging
import time
import asyncio
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import Config

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating and working with text embeddings"""
    
    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.model_name = Config.EMBEDDING_MODEL_NAME
        self.is_loaded = False
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the embedding model"""
        logger.info(f"Loading embedding model: {self.model_name}")
        
        try:
            start_time = time.time()
            self.model = SentenceTransformer(self.model_name)
            load_time = time.time() - start_time
            
            self.is_loaded = True
            logger.info(f"Embedding model loaded successfully in {load_time:.2f}s")
            
            # Test the model with a simple embedding
            test_embedding = self.model.encode("test")
            logger.info(f"Model test successful, embedding dimension: {len(test_embedding)}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.is_loaded = False
            self.model = None
    
    def is_available(self) -> bool:
        """Check if embedding service is available"""
        return self.is_loaded and self.model is not None
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text"""
        if not self.is_available():
            logger.error("Embedding service not available")
            return None
        
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        try:
            logger.debug(f"Generating embedding for text of length: {len(text)}")
            start_time = time.time()
            
            # Normalize text
            normalized_text = self.normalize_text(text)
            embedding = self.model.encode(normalized_text)
            
            generation_time = time.time() - start_time
            logger.debug(f"Embedding generated in {generation_time:.4f}s")
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts in batches"""
        if not self.is_available():
            logger.error("Embedding service not available")
            return [None] * len(texts)
        
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}")
        
        try:
            start_time = time.time()
            
            # Normalize all texts
            normalized_texts = [self.normalize_text(text) for text in texts]
            
            # Process in batches
            embeddings = []
            for i in range(0, len(normalized_texts), batch_size):
                batch = normalized_texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, batch_size=len(batch))
                embeddings.extend([emb.tolist() for emb in batch_embeddings])
            
            generation_time = time.time() - start_time
            logger.info(f"Generated {len(embeddings)} embeddings in {generation_time:.2f}s")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [None] * len(texts)
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            if not embedding1 or not embedding2:
                return 0.0
            
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: List[float], 
                         document_embeddings: List[List[float]], 
                         top_k: int = None) -> List[int]:
        """Find indices of most similar documents to query"""
        if not query_embedding or not document_embeddings:
            return []
        
        try:
            top_k = top_k or Config.TOP_K
            
            logger.debug(f"Finding top {top_k} similar documents from {len(document_embeddings)} candidates")
            
            # Calculate similarities
            similarities = []
            for doc_emb in document_embeddings:
                if doc_emb:
                    sim = cosine_similarity([query_embedding], [doc_emb])[0][0]
                    similarities.append(sim)
                else:
                    similarities.append(0.0)
            
            # Get top K indices
            similarities_array = np.array(similarities)
            top_indices = np.argsort(similarities_array)[-top_k:][::-1]
            
            # Filter by minimum similarity threshold
            filtered_indices = []
            for idx in top_indices:
                if similarities[idx] >= Config.MIN_SIMILARITY:
                    filtered_indices.append(int(idx))
            
            logger.debug(f"Found {len(filtered_indices)} documents above similarity threshold {Config.MIN_SIMILARITY}")
            return filtered_indices
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for consistent embedding generation"""
        if not text:
            return ""
        
        # Basic normalization
        normalized = text.lower().strip()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def get_stats(self) -> dict:
        """Get embedding service statistics"""
        return {
            "available": self.is_available(),
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "embedding_dimension": self.model.get_sentence_embedding_dimension() if self.is_available() else None
        }

# Global embedding service instance
embedding_service = EmbeddingService()