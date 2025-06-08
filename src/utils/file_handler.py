# src/utils/file_handler.py
"""
File handling utilities
"""

import json
import yaml
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

# LISÄÄ TÄMÄ LUOKKA TÄHÄN:
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
class FileHandler:
    """Handle file operations"""
    
    @staticmethod
    def read_text_file(filepath: str) -> str:
        """Read text file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def save_text_file(content: str, filepath: str):
        """Save text file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    @staticmethod
    def load_json(filepath: str) -> Dict[str, Any]:
        """Load JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_json(data: Dict[str, Any], filepath: str):
        """Save JSON file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder) # LISÄÄ: cls=NumpyEncoder
    
    @staticmethod
    def load_yaml(filepath: str) -> Dict[str, Any]:
        """Load YAML file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def save_yaml(data: Dict[str, Any], filepath: str):
        """Save YAML file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False)