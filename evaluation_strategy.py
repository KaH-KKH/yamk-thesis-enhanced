# evaluation_strategy.py
"""
Helper script for running different evaluation strategies
Place this in the project root directory
"""

import subprocess
import argparse
import sys
from datetime import datetime
from pathlib import Path
import json

# Define available models
AVAILABLE_MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "gemma": "gemma_7b_it_4bit",
    "falcon": "Falcon3-7B-Base",
    "llama": "Meta-Llama-3-8B-Instruct",
    "qwen": "Qwen2-7B-Instruct"
}

def run_evaluation(models, extended_metrics=True, no_llm_eval=False):
    """Run evaluation for given models"""
    cmd = [
        sys.executable, "-m", "src.evaluators.run_evaluation",
        "--models", ",".join(models)
    ]
    
    if extended_metrics:
        cmd.append("--extended-metrics")
    
    if no_llm_eval:
        cmd.append("--no-llm-eval")
    
    print(f"\n{'='*60}")
    print(f"Running evaluation: {' vs '.join(models)}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def baseline_comparison(baseline_model, compare_models, extended_metrics=True, no_llm_eval=False):
    """Run baseline comparison against multiple models"""
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nStarting baseline comparison: {baseline_model} vs {len(compare_models)} models")
    print(f"Timestamp: {timestamp}\n")
    
    for model in compare_models:
        if model != baseline_model:
            success = run_evaluation([baseline_model, model], extended_metrics, no_llm_eval)
            results.append({
                "baseline": baseline_model,
                "compared_to": model,
                "success": success,
                "timestamp": datetime.now().isoformat()
            })
            
            # Pause between evaluations to ensure memory is cleared
            if success:
                print("\nPausing 10 seconds before next evaluation...\n")
                import time
                time.sleep(10)
    
    # Save results summary
    summary_file = f"baseline_comparison_{baseline_model}_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBaseline comparison complete!")
    print(f"Results saved to: {summary_file}")
    
    return results

def multi_model_comparison(models, extended_metrics=True, no_llm_eval=False):
    """Run comparison of multiple models (2-3 recommended)"""
    if len(models) > 3:
        print("WARNING: Comparing more than 3 models at once may cause memory issues!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    return run_evaluation(models, extended_metrics, no_llm_eval)

def five_model_strategy():
    """Run the 5-model evaluation strategy"""
    print("\n5-Model Evaluation Strategy")
    print("="*60)
    
    # Phase 1: Baseline comparisons
    print("\nPhase 1: Baseline comparisons (Mistral vs all)")
    baseline_results = baseline_comparison(
        "mistral", 
        ["gemma", "falcon", "llama", "qwen"],
        extended_metrics=True
    )
    
    # Analyze results to find top performers
    print("\nPhase 1 complete. Please review results and select top 3 models.")
    print("Then run Phase 2 with: python evaluation_strategy.py --top3 mistral,model2,model3")

def main():
    parser = argparse.ArgumentParser(description="Evaluation strategy helper")
    parser.add_argument("--baseline", choices=AVAILABLE_MODELS.keys(), 
                       help="Run baseline comparison for specified model")
    parser.add_argument("--compare-with", nargs="+", choices=AVAILABLE_MODELS.keys(),
                       help="Models to compare against baseline")
    parser.add_argument("--multi", nargs="+", choices=AVAILABLE_MODELS.keys(),
                       help="Run multi-model comparison")
    parser.add_argument("--five-model", action="store_true",
                       help="Run 5-model evaluation strategy")
    parser.add_argument("--no-extended", action="store_true",
                       help="Disable extended metrics")
    parser.add_argument("--no-llm-eval", action="store_true",
                       help="Disable LLM evaluation")
    
    args = parser.parse_args()
    
    if args.baseline:
        if args.compare_with:
            baseline_comparison(
                args.baseline, 
                args.compare_with,
                not args.no_extended,
                args.no_llm_eval
            )
        else:
            # Compare against all other models
            all_models = list(AVAILABLE_MODELS.keys())
            all_models.remove(args.baseline)
            baseline_comparison(
                args.baseline,
                all_models,
                not args.no_extended,
                args.no_llm_eval
            )
    
    elif args.multi:
        multi_model_comparison(
            args.multi,
            not args.no_extended,
            args.no_llm_eval
        )
    
    elif args.five_model:
        five_model_strategy()
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Baseline comparison - Mistral vs all")
        print("  python evaluation_strategy.py --baseline mistral")
        print("\n  # Baseline comparison - Gemma vs specific models")
        print("  python evaluation_strategy.py --baseline gemma --compare-with mistral falcon")
        print("\n  # Multi-model comparison (top 3)")
        print("  python evaluation_strategy.py --multi mistral gemma llama")
        print("\n  # Run 5-model strategy")
        print("  python evaluation_strategy.py --five-model")
        print("\n  # Run without LLM evaluation (faster)")
        print("  python evaluation_strategy.py --baseline mistral --no-llm-eval")

if __name__ == "__main__":
    main()