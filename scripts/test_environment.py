#!/usr/bin/env python3
"""Test environment setup and GPU availability"""

import sys
import os
import torch
import importlib

def test_cuda():
    """Test CUDA/GPU availability"""
    print("\n=== GPU/CUDA Status ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
    else:
        print("Running in CPU-only mode")
    
    # Test tensor operation
    print("\n=== Testing PyTorch Operations ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(100, 100).to(device)
    y = torch.randn(100, 100).to(device)
    z = torch.matmul(x, y)
    print(f"✓ Tensor operations on {device} successful!")

def test_imports():
    """Test if all key packages can be imported"""
    print("\n=== Testing Package Imports ===")
    packages = [
        'transformers', 'pydantic', 'pydantic_ai', 
        'datasets', 'accelerate', 'evaluate',
        'nltk', 'spacy', 'pandas', 'numpy'
    ]
    
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")

def main():
    print("YAMK Thesis Environment Test")
    print("=" * 50)
    
    # Print Python info
    print(f"Python: {sys.version}")
    print(f"Virtual env: {os.environ.get('VIRTUAL_ENV', 'Not activated')}")
    
    test_cuda()
    test_imports()
    
    print("\n" + "=" * 50)
    print("Environment test completed!")

if __name__ == "__main__":
    main()
