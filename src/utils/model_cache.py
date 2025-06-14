# src/utils/model_cache.py
"""
Singleton model cache to manage loaded models across agents
UUSI TIEDOSTO - Luo tämä src/utils/ kansioon
"""

import torch
from loguru import logger
import gc
from typing import Dict, Any, Optional, Tuple


class ModelCache:
    """Singleton cache for loaded models"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
            cls._instance._tokenizers = {}
            cls._instance._usage_count = {}
            logger.info("Model cache initialized")
        return cls._instance
    
    def add_model(self, model_name: str, model: Any, tokenizer: Any):
        """Add a model to cache"""
        self._models[model_name] = model
        self._tokenizers[model_name] = tokenizer
        self._usage_count[model_name] = self._usage_count.get(model_name, 0) + 1
        logger.info(f"Model {model_name} added to cache (usage count: {self._usage_count[model_name]})")
    
    def get_model(self, model_name: str) -> Tuple[Optional[Any], Optional[Any]]:
        """Get model from cache"""
        if model_name in self._models:
            self._usage_count[model_name] = self._usage_count.get(model_name, 0) + 1
            return self._models[model_name], self._tokenizers.get(model_name)
        return None, None
    
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
    
    def remove_model(self, model_name: str):
        """Remove model from cache"""
        if model_name in self._models:
            del self._models[model_name]
            if model_name in self._tokenizers:
                del self._tokenizers[model_name]
            if model_name in self._usage_count:
                del self._usage_count[model_name]
            
            # Force cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Model {model_name} removed from cache")
    
    def clear_all(self):
        """Clear all cached models"""
        logger.warning("Clearing all models from cache")
        
        # Delete all references
        for model_name in list(self._models.keys()):
            self.remove_model(model_name)
        
        # Clear dictionaries
        self._models.clear()
        self._tokenizers.clear()
        self._usage_count.clear()
        
        # Aggressive cleanup
        for _ in range(3):
            gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        logger.info("Model cache cleared")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage of cached models"""
        info = {
            "cached_models": list(self._models.keys()),
            "usage_counts": self._usage_count.copy()
        }
        
        if torch.cuda.is_available():
            info["gpu_allocated_gb"] = torch.cuda.memory_allocated(0) / 1024**3
            info["gpu_reserved_gb"] = torch.cuda.memory_reserved(0) / 1024**3
        
        return info
    
    def optimize_cache(self, max_models: int = 1):
        """Remove least used models if cache is too large"""
        if len(self._models) > max_models:
            # Sort by usage count
            sorted_models = sorted(
                self._usage_count.items(), 
                key=lambda x: x[1]
            )
            
            # Remove least used models
            while len(self._models) > max_models:
                model_to_remove = sorted_models.pop(0)[0]
                logger.info(f"Removing least used model from cache: {model_to_remove}")
                self.remove_model(model_to_remove)
    
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