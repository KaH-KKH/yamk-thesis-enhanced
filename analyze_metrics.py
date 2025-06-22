#!/usr/bin/env python3
"""
Analyze what metrics are actually available in result files
"""

import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

def analyze_available_metrics(results_dir="results"):
    """Analyze what metrics are available across all result files"""
    
    results_path = Path(results_dir)
    json_files = list(results_path.glob("**/*_results.json"))
    
    if not json_files:
        print("No result files found!")
        return
    
    print(f"Analyzing {len(json_files)} result files...\n")
    
    # Track what we find
    all_metrics = defaultdict(list)
    file_info = []
    missing_expected = defaultdict(int)
    
    # Expected metrics
    expected_metrics = {
        'completeness': ['metrics', 'use_case_metrics', 'custom', 'completeness'],
        'test_validity': ['metrics', 'test_case_metrics', 'syntax_validity', 'validity_rate'],
        'perplexity': ['metrics', 'use_case_metrics', 'quality', 'perplexity', 'mean_perplexity'],
        'bleu': ['metrics', 'use_case_metrics', 'traditional', 'bleu', 'bleu'],
        'generation_time': ['performance', 'total_time']
    }
    
    for json_file in json_files:
        model_name = json_file.stem.replace("_results", "")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            file_metrics = {
                'file': json_file.name,
                'model': model_name,
                'run': json_file.parent.name
            }
            
            # Check for expected metrics
            for metric_name, path in expected_metrics.items():
                value = get_nested_value(data, path)
                if value is not None:
                    file_metrics[metric_name] = value
                    all_metrics[metric_name].append(value)
                else:
                    missing_expected[metric_name] += 1
            
            # Find all numeric values recursively
            found_metrics = find_all_metrics(data)
            file_metrics.update(found_metrics)
            
            file_info.append(file_metrics)
            
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    # Summary report
    print("=" * 60)
    print("METRICS AVAILABILITY REPORT")
    print("=" * 60)
    
    print("\n1. EXPECTED METRICS STATUS:")
    print("-" * 40)
    for metric, path in expected_metrics.items():
        found_count = len(all_metrics.get(metric, []))
        missing_count = missing_expected[metric]
        total = found_count + missing_count
        if total > 0:
            availability = (found_count / total) * 100
            print(f"{metric:20} Found: {found_count}/{total} ({availability:.1f}%)")
            if found_count > 0:
                values = all_metrics[metric]
                print(f"{'':20} Range: {min(values):.3f} - {max(values):.3f}")
        
    # Create DataFrame for detailed analysis
    df = pd.DataFrame(file_info)
    
    print("\n2. ALL NUMERIC FIELDS FOUND:")
    print("-" * 40)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        non_null = df[col].notna().sum()
        if non_null > 0:
            print(f"{col:30} {non_null}/{len(df)} files")
    
    print("\n3. SAMPLE DATA STRUCTURE:")
    print("-" * 40)
    if json_files:
        with open(json_files[0], 'r') as f:
            sample = json.load(f)
        print_structure(sample)
    
    # Save detailed results
    output_file = "metrics_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    return df

def get_nested_value(data, path):
    """Get value from nested dictionary using path"""
    try:
        current = data
        for key in path:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return None

def find_all_metrics(data, prefix=""):
    """Recursively find all numeric values in nested structure"""
    metrics = {}
    
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[new_prefix] = value
            elif isinstance(value, dict):
                metrics.update(find_all_metrics(value, new_prefix))
    
    return metrics

def print_structure(data, indent=0):
    """Print data structure"""
    if isinstance(data, dict):
        for key, value in data.items():
            print(" " * indent + f"{key}: ", end="")
            if isinstance(value, dict):
                print("{")
                print_structure(value, indent + 2)
                print(" " * indent + "}")
            elif isinstance(value, list):
                print(f"[{len(value)} items]")
            else:
                print(f"{type(value).__name__}")

if __name__ == "__main__":
    analyze_available_metrics()