#!/bin/bash
# Quick activation script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "yamk_env/bin/activate" ]; then
    source yamk_env/bin/activate
    echo "✓ Virtual environment activated: yamk_env"
    echo "✓ Working directory: $(pwd)"
    echo "✓ Python: $(which python)"
    
    # Set CUDA paths if available
    if command -v nvidia-smi &> /dev/null; then
        echo "✓ GPU available: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    fi
else
    echo "✗ Virtual environment not found. Run setup script first."
fi
