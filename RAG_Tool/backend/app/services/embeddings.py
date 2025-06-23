import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Optional
import pickle
import hashlib
from functools import wraps
import logging

logger = logging.getLogger(__name__)

def cache_embeddings(expiry: int = 3600):
    def decorator(func):
        @wraps(func)
        def wrapper(self, texts: List[str], *args, **kwargs):
            cache_key = f"embedding:{hashlib.md5(str(texts).encode()).hexdigest()}"
            
            try:
                cached = db_manager.get_redis().get(cache_key)
                if cached:
                    logger.info(f"Cache hit for {len(texts)} texts")
                    return pickle.loads(cached)
            except Exception as e:
                logger.warning(f"Cache retrieval failed: {e}")

            result = func(self, texts, *args, **kwargs)
            
            try:
                db_manager.get_redis().setex(
                    cache_key, expiry, pickle.dumps(result)
                )
                logger.info(f"Cached embeddings for {len(texts)} texts")
            except Exception as e:
                logger.warning(f"Cache storage failed: {e}")
            
            return result
        return wrapper
    return decorator

class EmbeddingService:
    def __init__(self):
        self.general_model = None
        self.financial_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_general_model(self):
        if not self.general_model:
            self.general_model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=self.device
            )
            logger.info(f"Loaded general embedding model on {self.device}")
    
    def load_financial_model(self):
        if not self.financial_model:
            self.financial_model = SentenceTransformer(
                settings.FINANCIAL_EMBEDDING_MODEL,
                device=self.device
            )
            logger.info(f"Loaded financial embedding model on {self.device}")
    
    @cache_embeddings()
    def encode_general(self, texts: List[str]) -> np.ndarray:
        self.load_general_model()
        return self.general_model.encode(texts).astype(np.float32)
    
    @cache_embeddings()
    def encode_financial(self, texts: List[str]) -> np.ndarray:
        self.load_financial_model()
        return self.financial_model.encode(texts).astype(np.float32)
    
    def encode_hybrid(self, texts: List[str], financial_weight: float = 0.7) -> np.ndarray:
        general_embeddings = self.encode_general(texts)
        financial_embeddings = self.encode_financial(texts)

        hybrid_embeddings = (
            financial_weight * financial_embeddings + 
            (1 - financial_weight) * general_embeddings
        )
        
        return hybrid_embeddings.astype(np.float32)

embedding_service = EmbeddingService()