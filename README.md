# YAMK Thesis - LLM Evaluation Project

## Overview
Multi-agent system for evaluating open-source LLMs in test automation.

## Setup
1. Project has been set up in isolated virtual environment
2. Activate environment: `source activate.sh` or `source yamk_env/bin/activate`
3. Copy `.env.template` to `.env` and add your API keys
4. Test environment: `python scripts/test_environment.py`

## Project Structure
```
yamk_thesis_enhanced/
├── yamk_env/           # Virtual environment (not in git)
├── src/                # Source code
├── data/               # Data files
├── configs/            # Configuration files
├── results/            # Output results
├── scripts/            # Utility scripts
└── notebooks/          # Jupyter notebooks
```

## Virtual Environment
All dependencies are installed in the isolated virtual environment `yamk_env`.
No system-wide installations are made.

## GPU Support
The system automatically detects CUDA availability and configures PyTorch accordingly.
