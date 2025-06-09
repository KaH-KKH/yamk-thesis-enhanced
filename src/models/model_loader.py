"""
Model loader for open-source LLMs with 8-bit (4-bit) quantization
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


class ModelLoader:
    """Load and manage open-source language models"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models_config = {
            model['name']: model 
            for model in self.config['models']['available']
        }
        
        self.loaded_models = {}
        self.tokenizers = {}
        
        # Setup quantization config for 4-bit
        # self.quantization_config = BitsAndBytesConfig(
        #    load_in_4bit=True,
        #    bnb_4bit_quant_type="nf4",
        #    bnb_4bit_use_double_quant=True,
        #    bnb_4bit_compute_dtype=torch.bfloat16,
        #    llm_int8_enable_fp32_cpu_offload=True  # LISÄÄ TÄMÄ TÄNNE
        # )

        # Setup quantization config for 8-bit (vähemmän ongelmia)
        self.quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,  # Muuta 4bit -> 8bit
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_threshold=6.0,
        )
        
        logger.info("Model loader initialized")
    
    def load_model(self, model_name: str):
        # Tyhjennä muisti ennen uuden mallin lataamista
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        """Load a model by name"""
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded")
            return self.loaded_models[model_name]
        
        if model_name not in self.models_config:
            raise ValueError(f"Model {model_name} not found in configuration")
        
        model_config = self.models_config[model_name]
        model_id = model_config['model_id']
        
        logger.info(f"Loading model: {model_name} ({model_id})")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with 4-bit quantization
            # model = AutoModelForCausalLM.from_pretrained(
            #    model_id,
            #    quantization_config=self.quantization_config,
            #    device_map="auto",
            #    trust_remote_code=True,
            #    torch_dtype=torch.bfloat16,
            # )

            # Load model with 8-bit quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=self.quantization_config,
                device_map={"": 0},  # Muuta: pakottaa kaikki GPU:lle
                trust_remote_code=True,
                # torch_dtype=torch.bfloat16,
                torch_dtype=torch.float16,  # Muuta: bfloat16 -> float16
                low_cpu_mem_usage=True,  # Lisää tämä
                offload_folder=None,     # Lisää tämä: estää CPU offloading
            )
            
            # Store loaded model and tokenizer
            self.loaded_models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            self.tokenizer = tokenizer  # For compatibility
            
            logger.success(f"Model {model_name} loaded successfully")
            
            # Log model info
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Model device: {next(model.parameters()).device}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
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
        tokenizer = self.tokenizers[model_name]
        
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048, padding=True)
        # LISÄÄ TÄMÄ RIVI: Poista token_type_ids jos se on olemassa
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
    
    def unload_model(self, model_name: str):
        """Unload a model to free memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            del self.tokenizers[model_name]
            
            import gc
            gc.collect()  # Python garbage collection
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Varmista että cache tyhjenee
                
            logger.info(f"Model {model_name} unloaded")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model"""
        if model_name not in self.models_config:
            raise ValueError(f"Model {model_name} not found")
        
        info = self.models_config[model_name].copy()
        
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
    
    #def list_available_models(self) -> List[str]:
    def list_available_models(self) -> list[str]:
        """List all available models"""
        return list(self.models_config.keys())
    
    def add_model(self, name: str, model_id: str, quantization: str = "4bit"):
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
    model_name = "mistral"
    model = loader.load_model(model_name)
    
    # Generate text
    prompt = "Write a user story for a login feature:"
    response = loader.generate(model_name, prompt, max_new_tokens=200)
    print(f"\nGenerated text:\n{response}")
    
    # Get model info
    info = loader.get_model_info(model_name)
    print(f"\nModel info:\n{info}")