"""
Model loader for open-source LLMs and SLMs with flexible quantization support
PÄIVITETTY VERSIO: Lisätty SLM-tuki (Small Language Models)
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from loguru import logger
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from ..utils.model_cache import ModelCache


class ModelLoader:
    """Load and manage open-source language models (both LLMs and SLMs)"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models_config = {
            model['name']: model 
            for model in self.config['models']['available']
        }
        
        self.loaded_models = {}
        self.tokenizers = {}
        
        # KORJAUS: Lisätty vaihtoehtoiset kvantisointikonfiguraatiot
        # 8-bit config (vakaa mutta käyttää enemmän muistia)
        self.quantization_config_8bit = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_threshold=6.0,
        )
        
        # 4-bit config (käyttää vähemmän muistia)
        self.quantization_config_4bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Default to 8-bit for stability
        self.quantization_config = self.quantization_config_8bit
        
        logger.info("Model loader initialized")
    
    # KORJAUS: Lisätty GPU muistin seurantafunktio
    def _get_gpu_memory_usage(self):
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            return allocated, reserved
        return 0, 0
    
    def _estimate_model_size(self, model_id: str) -> float:
        """Estimate model size in billions of parameters based on model ID"""
        # Common patterns for model sizes
        size_patterns = {
            '1.1b': 1.1, '1b': 1.0, '1.3b': 1.3, '1.4b': 1.4,
            '2b': 2.0, '2.7b': 2.7, '3b': 3.0,
            '7b': 7.0, '8b': 8.0, '13b': 13.0,
            'tiny': 0.5, 'small': 1.0, 'base': 3.0, 'large': 7.0
        }
        
        model_id_lower = model_id.lower()
        
        # Check for size in model ID
        for pattern, size in size_patterns.items():
            if pattern in model_id_lower:
                return size
        
        # Default estimates based on known models
        if 'tinyllama' in model_id_lower:
            return 1.1
        elif 'phi-2' in model_id_lower:
            return 2.7
        elif 'pythia' in model_id_lower and '1b' in model_id_lower:
            return 1.0
        elif 'opt' in model_id_lower and '1.3b' in model_id_lower:
            return 1.3
        elif 'stablelm' in model_id_lower and '3b' in model_id_lower:
            return 3.0
        
        # Default to medium size if unknown
        return 3.0
    
    def load_model(self, model_name: str, force_reload: bool = False):
        """Load a model by name with improved memory management and SLM support"""
        
        # KORJAUS: Optimoi cache käyttöä - pidä vain 1 malli kerrallaan muistissa
        cache = ModelCache()
        if len(cache.list_cached_models()) >= 1 and model_name not in cache.list_cached_models():
            logger.info("Cache full, clearing old models before loading new one")
            cache.optimize_cache(max_models=0)  # Tyhjennä kaikki vanhat mallit
            
        # Log initial memory state
        allocated, reserved = self._get_gpu_memory_usage()
        logger.info(f"GPU memory before loading {model_name}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
        
        # Clear memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Check cache
        if not force_reload and hasattr(cache, '_models') and model_name in cache._models:
            logger.info(f"Using cached model: {model_name}")
            if hasattr(cache, '_tokenizers') and model_name in cache._tokenizers:
                self.tokenizers[model_name] = cache._tokenizers[model_name]
                self.tokenizer = cache._tokenizers[model_name]
            return cache._models[model_name]
        
        if model_name in self.loaded_models and not force_reload:
            logger.info(f"Model {model_name} already loaded")
            return self.loaded_models[model_name]
        
        if model_name not in self.models_config:
            raise ValueError(f"Model {model_name} not found in configuration")
        
        model_config = self.models_config[model_name]
        model_id = model_config['model_id']
        
        logger.info(f"Loading model: {model_name} ({model_id})")
        
        # PÄIVITYS: Tarkista quantization config-tiedostosta
        quantization = model_config.get('quantization', 'none')
        
        # Estimate model size
        estimated_size = self._estimate_model_size(model_id)
        logger.info(f"Estimated model size: {estimated_size}B parameters")
        
        try:
            # Load tokenizer first (uses less memory)
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # PÄIVITYS: Valitse latausstrategia quantization-asetuksen mukaan
            if quantization == 'none':
                # SLM-malleille ei quantizatiota
                logger.info(f"Loading {model_name} without quantization (SLM mode)")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                )
                
            elif quantization == '4bit':
                # Käytä 4-bit quantization
                logger.info(f"Loading {model_name} with 4-bit quantization")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=self.quantization_config_4bit,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    max_memory={0: "20GB", "cpu": "30GB"},
                    offload_folder="offload",
                )
                
            elif quantization == '8bit':
                # Käytä 8-bit quantization
                logger.info(f"Loading {model_name} with 8-bit quantization")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=self.quantization_config_8bit,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    max_memory={0: "22GB", "cpu": "30GB"},
                    offload_folder="offload",
                )
                
            else:
                # Automaattinen valinta mallin koon mukaan
                logger.info("Auto-selecting quantization based on model size")
                
                if estimated_size <= 2.0:
                    # Pienet mallit (≤2B) - ei quantizatiota
                    logger.info(f"Small model detected ({estimated_size}B), loading without quantization")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                    )
                    
                elif estimated_size <= 3.0:
                    # Keskikokoiset mallit (2-3B) - 8-bit tai none riippuen muistista
                    allocated, _ = self._get_gpu_memory_usage()
                    if allocated > 10.0:  # Jos muisti tiukalla
                        logger.info(f"Medium model with high memory usage, using 8-bit quantization")
                        model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            quantization_config=self.quantization_config_8bit,
                            device_map="auto",
                            trust_remote_code=True,
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True,
                            max_memory={0: "22GB", "cpu": "30GB"},
                            offload_folder="offload",
                        )
                    else:
                        logger.info(f"Medium model with available memory, loading without quantization")
                        model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            device_map="auto",
                            trust_remote_code=True,
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True,
                        )
                        
                else:
                    # Isot mallit (>3B) - käytä quantizatiota
                    allocated, _ = self._get_gpu_memory_usage()
                    if allocated > 5.0 or estimated_size >= 7.0:
                        logger.info(f"Large model or high memory usage, using 4-bit quantization")
                        current_quantization = self.quantization_config_4bit
                    else:
                        logger.info(f"Large model with available memory, using 8-bit quantization")
                        current_quantization = self.quantization_config_8bit
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=current_quantization,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        max_memory={0: "22GB", "cpu": "30GB"},
                        offload_folder="offload",
                    )
            
            # Store loaded model and tokenizer
            self.loaded_models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            self.tokenizer = tokenizer  # For compatibility
            
            # Add to cache
            cache.add_model(model_name, model, tokenizer)
            
            # Log success and memory usage
            allocated, reserved = self._get_gpu_memory_usage()
            logger.success(f"Model {model_name} loaded successfully")
            logger.info(f"GPU memory after loading: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
            
            # Log model info
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Total parameters: {total_params:,}")
            
            # Log quantization status
            if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
                logger.info(f"Quantization: {model.config.quantization_config}")
            else:
                logger.info("Quantization: None (full precision)")
            
            # Check if model is split across devices
            if hasattr(model, 'hf_device_map'):
                logger.info(f"Model device map: {model.hf_device_map}")
            
            return model
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM while loading {model_name}. Trying recovery...")
            
            # Emergency cleanup
            self._emergency_cleanup()
            
            # Retry with 4-bit quantization
            try:
                logger.info("Retrying with 4-bit quantization after OOM")
                # Force 4-bit for retry
                forced_4bit_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=forced_4bit_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    max_memory={0: "20GB", "cpu": "40GB"},  # Even more conservative
                    offload_folder="offload",
                )
                
                # If successful, store everything
                self.loaded_models[model_name] = model
                self.tokenizers[model_name] = tokenizer
                self.tokenizer = tokenizer
                cache.add_model(model_name, model, tokenizer)
                
                allocated, reserved = self._get_gpu_memory_usage()
                logger.success(f"Model {model_name} loaded with 4-bit after OOM recovery")
                logger.info(f"GPU memory after recovery: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
                
                return model
                
            except Exception as e2:
                logger.error(f"Failed to load {model_name} even with 4-bit: {str(e2)}")
                raise
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    # KORJAUS: Lisätty hätäsiivosfunktio
    def _emergency_cleanup(self):
        """Emergency cleanup when OOM occurs"""
        logger.warning("Performing emergency GPU cleanup")
        
        # Unload all models
        for model_name in list(self.loaded_models.keys()):
            self.unload_model(model_name)
        
        # Clear cache
        cache = ModelCache()
        cache.clear_all()
        
        # Force garbage collection
        import gc
        for _ in range(5):
            gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # Wait a bit
        import time
        time.sleep(3)
    
    def get_pipeline(self, model_name: str, task: str = "text-generation", **kwargs):
        """Get a pipeline for the model"""
        model = self.load_model(model_name)
        tokenizer = self.tokenizers[model_name]
        
        default_kwargs = {
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
        }
        
        default_kwargs.update(kwargs)
        
        return pipeline(
            task,
            model=model,
            tokenizer=tokenizer,
            **default_kwargs
        )
    
    def generate(self, model_name: str, prompt: str, **kwargs) -> str:
        """Generate text using the model"""
        model = self.load_model(model_name)
        
        # KORJAUS: Varmista että tokenizer on olemassa
        if model_name not in self.tokenizers:
            logger.error(f"Tokenizer for {model_name} not found in self.tokenizers")
            logger.info(f"Available tokenizers: {list(self.tokenizers.keys())}")
            # Yritä ladata uudelleen
            self.load_model(model_name)
        
        tokenizer = self.tokenizers[model_name]
        
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048, padding=True)
        # Poista token_type_ids jos se on olemassa
        inputs.pop('token_type_ids', None)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Default generation parameters
        gen_kwargs = {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        gen_kwargs.update(kwargs)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    # KORJAUS: Parannettu unload_model metodi
    def unload_model(self, model_name: str):
        """Unload a model to free memory with improved cleanup"""
        if model_name in self.loaded_models:
            # Delete model
            del self.loaded_models[model_name]
            
            # Delete tokenizer
            if model_name in self.tokenizers:
                del self.tokenizers[model_name]
            
            # KORJAUS: Poista myös cachesta
            cache = ModelCache()
            cache.remove_model(model_name)
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # KORJAUS: Raportti muistin tilasta poiston jälkeen
            allocated, reserved = self._get_gpu_memory_usage()
            logger.info(f"Model {model_name} unloaded. GPU memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    
    # Lisää myös tämä metodi joka tyhjentää kaikki paitsi tietyn mallin:
    def clear_all_except(self, keep_model: Optional[str] = None):
        """Clear all models except the specified one"""
        logger.info(f"Clearing all models except: {keep_model}")
        
        # Unload from loader
        for model_name in list(self.loaded_models.keys()):
            if model_name != keep_model:
                self.unload_model(model_name)
        
        # Clear from cache
        cache = ModelCache()
        for model_name in cache.list_cached_models():
            if model_name != keep_model:
                cache.remove_model(model_name)
        
        # Force cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model"""
        if model_name not in self.models_config:
            raise ValueError(f"Model {model_name} not found")
        
        info = self.models_config[model_name].copy()
        
        # Add size estimation
        model_id = info.get('model_id', '')
        info['estimated_size_b'] = self._estimate_model_size(model_id)
        
        if model_name in self.loaded_models:
            model = self.loaded_models[model_name]
            info.update({
                "loaded": True,
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "device": str(next(model.parameters()).device),
                "dtype": str(next(model.parameters()).dtype),
            })
        else:
            info["loaded"] = False
        
        return info
    
    def list_available_models(self) -> list[str]:
        """List all available models"""
        return list(self.models_config.keys())
    
    def add_model(self, name: str, model_id: str, quantization: str = "auto"):
        """Add a new model to configuration"""
        new_model = {
            "name": name,
            "model_id": model_id,
            "quantization": quantization
        }
        
        # Add to config
        self.models_config[name] = new_model
        self.config['models']['available'].append(new_model)
        
        # Save updated config
        config_path = Path("configs/config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Added model {name} to configuration")


# Example usage
if __name__ == "__main__":
    loader = ModelLoader()
    
    # List available models
    print("Available models:", loader.list_available_models())
    
    # Load a model
    model_name = "tinyllama"  # Try with an SLM
    model = loader.load_model(model_name)
    
    # Generate text
    prompt = "Write a user story for a login feature:"
    response = loader.generate(model_name, prompt, max_new_tokens=200)
    print(f"\nGenerated text:\n{response}")
    
    # Get model info
    info = loader.get_model_info(model_name)
    print(f"\nModel info:\n{info}")