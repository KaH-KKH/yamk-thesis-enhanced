# src/utils/model_cache.py
from typing import Dict, Any, Optional
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
    
    def get_model(self, model_name: str, loader):
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