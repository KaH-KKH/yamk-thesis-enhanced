# src/utils/model_cache.py
from typing import Dict, Any, Optional, Tuple
import torch
from loguru import logger

class ModelCache:
    """Global model cache to avoid reloading"""
    _instance = None
    _models = {}
    _tokenizers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_name: str, loader) -> Tuple[Any, Any]:
        """Get model from cache or load it"""
        if model_name not in self._models:
            logger.info(f"Loading {model_name} to cache...")
            model = loader.load_model(model_name)
            tokenizer = loader.tokenizers.get(model_name)
            self._models[model_name] = model
            self._tokenizers[model_name] = tokenizer
        else:
            logger.info(f"Using cached {model_name}")
        return self._models[model_name], self._tokenizers.get(model_name)
    
    def has_model(self, model_name: str) -> bool:
        """Check if model is in cache"""
        return model_name in self._models
    
    def get_cached_model(self, model_name: str) -> Optional[Any]:
        """Get model from cache (returns None if not found)"""
        return self._models.get(model_name)
    
    def get_cached_tokenizer(self, model_name: str) -> Optional[Any]:
        """Get tokenizer from cache (returns None if not found)"""
        return self._tokenizers.get(model_name)
    
    def cache_model(self, model_name: str, model: Any, tokenizer: Any):
        """Cache model and tokenizer"""
        self._models[model_name] = model
        self._tokenizers[model_name] = tokenizer
        logger.info(f"Cached {model_name} with tokenizer")
    
    def clear_model(self, model_name: str):
        """Remove specific model from cache"""
        if model_name in self._models:
            del self._models[model_name]
            if model_name in self._tokenizers:
                del self._tokenizers[model_name]
            torch.cuda.empty_cache()
            logger.info(f"Cleared {model_name} from cache")
    
    def clear_all(self):
        """Clear all cached models"""
        self._models.clear()
        self._tokenizers.clear()
        torch.cuda.empty_cache()
        logger.info("Cleared all models from cache")
    
    def list_cached_models(self) -> list[str]:
        """List all cached model names"""
        return list(self._models.keys())
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "models_count": len(self._models),
            "tokenizers_count": len(self._tokenizers),
            "model_names": list(self._models.keys()),
            "memory_info": self._get_memory_info() if torch.cuda.is_available() else {}
        }
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information"""
        if not torch.cuda.is_available():
            return {}
        
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3
        }