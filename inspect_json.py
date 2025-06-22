#!/usr/bin/env python3
"""
Simple script to inspect JSON structure of result files
"""

import json
from pathlib import Path
import sys

def inspect_json_files(results_dir="results"):
    """Inspect JSON structure of result files"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory '{results_dir}' not found!")
        return
    
    # Find all JSON files
    json_files = list(results_path.glob("**/*_results.json"))
    
    if not json_files:
        print("No result JSON files found!")
        return
    
    print(f"Found {len(json_files)} result files\n")
    
    # Inspect first file
    first_file = json_files[0]
    print(f"Inspecting: {first_file}\n")
    
    try:
        with open(first_file, 'r') as f:
            data = json.load(f)
        
        # Print structure
        print("Top-level keys:")
        for key in data.keys():
            print(f"  - {key}")
        
        # Check metrics structure
        if 'metrics' in data:
            print("\nMetrics structure:")
            print_dict_structure(data['metrics'], indent=2)
        
        # Check use_case_metrics
        if 'use_case_metrics' in data:
            print("\nuse_case_metrics structure:")
            print_dict_structure(data['use_case_metrics'], indent=2)
        
        # Check test_case_metrics
        if 'test_case_metrics' in data:
            print("\ntest_case_metrics structure:")
            print_dict_structure(data['test_case_metrics'], indent=2)
        
        # Check performance
        if 'performance' in data:
            print("\nperformance structure:")
            print_dict_structure(data['performance'], indent=2)
        
        # Save pretty-printed JSON
        output_file = "sample_result_structure.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nFull JSON saved to: {output_file}")
        
    except Exception as e:
        print(f"Error reading JSON: {e}")

def print_dict_structure(d, indent=0):
    """Recursively print dictionary structure"""
    if not isinstance(d, dict):
        return
    
    for key, value in d.items():
        print(" " * indent + f"- {key}: {type(value).__name__}")
        if isinstance(value, dict) and len(value) > 0:
            print_dict_structure(value, indent + 2)
        elif isinstance(value, list) and len(value) > 0:
            print(" " * (indent + 2) + f"[{len(value)} items]")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_json_files(sys.argv[1])
    else:
        inspect_json_files()