"""
Main evaluation runner for comparing LLM performance
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# Metrics
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
import evaluate

# Performance monitoring
import psutil
import GPUtil
from memory_profiler import memory_usage

from ..agents.uc_agent import UCAgent
from ..agents.rf_agent import RFAgent
from ..utils.file_handler import FileHandler
from ..utils.metrics import MetricsCalculator


class EvaluationRunner:
    """Run comprehensive evaluation of LLMs"""
    
    def __init__(self, models: List[str], config_path: str = "configs/config.yaml"):
        self.models = models
        self.config = FileHandler.load_yaml(config_path)
        self.results_dir = Path(self.config["paths"]["results_dir"])
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bertscore = evaluate.load("bertscore")
        
        # Create results directory
        self.run_dir = self.results_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Evaluation runner initialized for models: {models}")
    
    async def evaluate_model(self, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model"""
        logger.info(f"Starting evaluation for model: {model_name}")
        
        results = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "use_case_generation": {},
            "test_case_generation": {},
            "metrics": {},
            "performance": {}
        }
        
        try:
            # Phase 1: Use Case Generation
            uc_results = await self._evaluate_use_case_generation(model_name)
            results["use_case_generation"] = uc_results
            
            # Phase 2: Test Case Generation
            rf_results = await self._evaluate_test_case_generation(model_name)
            results["test_case_generation"] = rf_results
            
            # Phase 3: Calculate Metrics
            metrics = await self._calculate_metrics(model_name)
            results["metrics"] = metrics
            
            # Phase 4: Performance Analysis
            performance = self._analyze_performance(uc_results, rf_results)
            results["performance"] = performance
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    async def _evaluate_use_case_generation(self, model_name: str) -> Dict[str, Any]:
        """Evaluate use case generation"""
        logger.info(f"Evaluating use case generation for {model_name}")
        
        # Monitor performance
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Track GPU if available
        gpu_start = None
        if GPUtil.getGPUs():
            gpu_start = GPUtil.getGPUs()[0].memoryUsed
        
        # Initialize UC Agent
        uc_agent = UCAgent(model_name)
        
        # Generate use cases
        input_dir = self.config["paths"]["requirements_dir"]
        output_dir = self.config["paths"]["user_stories_dir"]
        
        generation_report = await uc_agent.batch_generate(input_dir, output_dir)
        
        # Calculate performance metrics
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        performance_metrics = {
            "total_time": end_time - start_time,
            "memory_used": end_memory - start_memory,
            "avg_time_per_file": (end_time - start_time) / generation_report["total_files"]
        }
        
        if gpu_start is not None and GPUtil.getGPUs():
            gpu_end = GPUtil.getGPUs()[0].memoryUsed
            performance_metrics["gpu_memory_used"] = gpu_end - gpu_start
        
        return {
            "generation_report": generation_report,
            "performance": performance_metrics
        }
    
    async def _evaluate_test_case_generation(self, model_name: str) -> Dict[str, Any]:
        """Evaluate test case generation"""
        logger.info(f"Evaluating test case generation for {model_name}")
        
        # Similar performance monitoring
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Initialize RF Agent
        rf_agent = RFAgent(model_name)
        
        # Generate test cases from use cases
        input_dir = Path(self.config["paths"]["user_stories_dir"]) / model_name
        output_dir = self.config["paths"]["test_cases_dir"]
        
        generation_report = await rf_agent.batch_generate(str(input_dir), output_dir)
        
        # Test execution validation
        test_results = await self._validate_test_execution(model_name)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return {
            "generation_report": generation_report,
            "test_validation": test_results,
            "performance": {
                "total_time": end_time - start_time,
                "memory_used": end_memory - start_memory
            }
        }
    
    async def _calculate_metrics(self, model_name: str) -> Dict[str, Any]:
        """Calculate evaluation metrics"""
        logger.info(f"Calculating metrics for {model_name}")
        
        metrics = {
            "use_case_metrics": {},
            "test_case_metrics": {}
        }
        
        # Load generated content
        uc_dir = Path(self.config["paths"]["user_stories_dir"]) / model_name
        tc_dir = Path(self.config["paths"]["test_cases_dir"]) / model_name
        
        # Calculate use case metrics
        if uc_dir.exists():
            uc_files = list(uc_dir.glob("*.txt"))
            if uc_files:
                # Load reference if available
                references = self._load_references("use_cases")
                candidates = [FileHandler.read_text_file(str(f)) for f in uc_files]
                
                if references:
                    # BLEU Score
                    bleu = corpus_bleu(candidates, [references])
                    metrics["use_case_metrics"]["bleu"] = bleu.score
                    
                    # ROUGE Scores
                    rouge_scores = []
                    for cand, ref in zip(candidates, references):
                        scores = self.rouge_scorer.score(ref, cand)
                        rouge_scores.append({
                            "rouge1": scores["rouge1"].fmeasure,
                            "rouge2": scores["rouge2"].fmeasure,
                            "rougeL": scores["rougeL"].fmeasure
                        })
                    
                    # Average ROUGE scores
                    avg_rouge = {
                        k: sum(s[k] for s in rouge_scores) / len(rouge_scores)
                        for k in ["rouge1", "rouge2", "rougeL"]
                    }
                    metrics["use_case_metrics"]["rouge"] = avg_rouge
                    
                    # BERTScore
                    P, R, F1 = bert_score(candidates, references, lang="en", device="cuda" if torch.cuda.is_available() else "cpu")
                    metrics["use_case_metrics"]["bertscore"] = {
                        "precision": P.mean().item(),
                        "recall": R.mean().item(),
                        "f1": F1.mean().item()
                    }
                
                # Custom metrics
                custom_metrics = self._calculate_custom_metrics(candidates, "use_case")
                metrics["use_case_metrics"]["custom"] = custom_metrics
        
        # Calculate test case metrics
        if tc_dir.exists():
            tc_files = list(tc_dir.glob("*.robot"))
            if tc_files:
                candidates = [FileHandler.read_text_file(str(f)) for f in tc_files]
                
                # Syntax validation
                syntax_results = []
                for tc_file in tc_files:
                    is_valid = self._validate_robot_syntax(str(tc_file))
                    syntax_results.append(is_valid)
                
                metrics["test_case_metrics"]["syntax_validity"] = {
                    "valid_count": sum(syntax_results),
                    "total_count": len(syntax_results),
                    "validity_rate": sum(syntax_results) / len(syntax_results) if syntax_results else 0
                }
                
                # Keyword coverage
                keyword_coverage = self._analyze_keyword_coverage(candidates)
                metrics["test_case_metrics"]["keyword_coverage"] = keyword_coverage
        
        return metrics
    
    def _calculate_custom_metrics(self, texts: List[str], content_type: str) -> Dict[str, Any]:
        """Calculate custom metrics for generated content"""
        metrics = {}
        
        if content_type == "use_case":
            # Completeness check
            required_sections = ["ACTORS", "PRECONDITIONS", "MAIN FLOW", "POSTCONDITIONS"]
            completeness_scores = []
            
            for text in texts:
                score = sum(1 for section in required_sections if section in text.upper())
                completeness_scores.append(score / len(required_sections))
            
            metrics["completeness"] = sum(completeness_scores) / len(completeness_scores)
            
            # Structure quality
            metrics["avg_length"] = sum(len(text.split()) for text in texts) / len(texts)
            metrics["avg_steps"] = self._count_steps(texts)
        
        return metrics
    
    def _count_steps(self, texts: List[str]) -> float:
        """Count average number of steps in use cases"""
        step_counts = []
        for text in texts:
            # Count numbered steps
            import re
            steps = re.findall(r'\d+\.', text)
            step_counts.append(len(steps))
        return sum(step_counts) / len(step_counts) if step_counts else 0
    
    def _validate_robot_syntax(self, robot_file: str) -> bool:
        """Validate Robot Framework syntax"""
        try:
            from robot.parsing import get_model
            model = get_model(robot_file)
            return model is not None
        except Exception:
            return False
    
    def _analyze_keyword_coverage(self, test_cases: List[str]) -> Dict[str, Any]:
        """Analyze Browser library keyword usage"""
        browser_keywords = [
            "New Browser", "New Page", "Go To", "Click", "Type Text",
            "Get Text", "Wait For Elements State", "Take Screenshot"
        ]
        
        keyword_counts = {kw: 0 for kw in browser_keywords}
        total_keywords = 0
        
        for tc in test_cases:
            for keyword in browser_keywords:
                count = tc.count(keyword)
                keyword_counts[keyword] += count
                total_keywords += count
        
        return {
            "keyword_counts": keyword_counts,
            "total_keywords": total_keywords,
            "unique_keywords_used": sum(1 for count in keyword_counts.values() if count > 0),
            "coverage_rate": sum(1 for count in keyword_counts.values() if count > 0) / len(browser_keywords)
        }
    
    async def _validate_test_execution(self, model_name: str) -> Dict[str, Any]:
        """Validate if generated tests can execute"""
        tc_dir = Path(self.config["paths"]["test_cases_dir"]) / model_name
        results = {
            "executable": 0,
            "failed": 0,
            "errors": []
        }
        
        if tc_dir.exists():
            tc_files = list(tc_dir.glob("*.robot"))
            for tc_file in tc_files:
                try:
                    # Dry run to check syntax
                    import subprocess
                    result = subprocess.run(
                        ["robot", "--dryrun", str(tc_file)],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        results["executable"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append({
                            "file": tc_file.name,
                            "error": result.stderr
                        })
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append({
                        "file": tc_file.name,
                        "error": str(e)
                    })
        
        results["executability_rate"] = results["executable"] / (results["executable"] + results["failed"]) if (results["executable"] + results["failed"]) > 0 else 0
        
        return results
    
    def _analyze_performance(self, uc_results: Dict, rf_results: Dict) -> Dict[str, Any]:
        """Analyze overall performance metrics"""
        performance = {
            "use_case_generation": uc_results.get("performance", {}),
            "test_case_generation": rf_results.get("performance", {}),
            "total_time": (
                uc_results.get("performance", {}).get("total_time", 0) +
                rf_results.get("performance", {}).get("total_time", 0)
            ),
            "total_memory": (
                uc_results.get("performance", {}).get("memory_used", 0) +
                rf_results.get("performance", {}).get("memory_used", 0)
            )
        }
        
        # Calculate tokens per second if available
        if "generation_report" in uc_results:
            total_files = uc_results["generation_report"]["total_files"]
            total_time = uc_results["performance"]["total_time"]
            if total_time > 0:
                performance["files_per_second"] = total_files / total_time
        
        return performance
    
    def _load_references(self, content_type: str) -> List[str]:
        """Load reference texts for comparison"""
        # In a real scenario, you would load human-written references
        # For now, return empty list
        return []
    
    async def compare_models(self) -> Dict[str, Any]:
        """Run evaluation for all models and compare"""
        logger.info(f"Starting comparison of {len(self.models)} models")
        
        all_results = {}
        
        for model in self.models:
            logger.info(f"Evaluating model: {model}")
            results = await self.evaluate_model(model)
            all_results[model] = results
            
            # Save individual model results
            model_file = self.run_dir / f"{model}_results.json"
            FileHandler.save_json(results, str(model_file))
        
        # Generate comparison report
        comparison = self._generate_comparison_report(all_results)
        
        # Save comparison
        comparison_file = self.run_dir / "comparison_report.json"
        FileHandler.save_json(comparison, str(comparison_file))
        
        # Generate visualizations
        self._generate_visualizations(all_results, comparison)
        
        # Generate summary report
        summary = self._generate_summary_report(comparison)
        summary_file = self.run_dir / "summary_report.md"
        FileHandler.save_text_file(summary, str(summary_file))
        
        logger.success(f"Evaluation complete! Results saved to: {self.run_dir}")
        
        return comparison
    
    def _generate_comparison_report(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison report across models"""
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "models": list(all_results.keys()),
            "summary": {},
            "detailed_metrics": {},
            "performance_comparison": {},
            "recommendations": []
        }
        
        # Extract metrics for comparison
        for metric_type in ["use_case_metrics", "test_case_metrics"]:
            comparison["detailed_metrics"][metric_type] = {}
            
            for model, results in all_results.items():
                if "metrics" in results and metric_type in results["metrics"]:
                    comparison["detailed_metrics"][metric_type][model] = results["metrics"][metric_type]
        
        # Performance comparison
        for model, results in all_results.items():
            if "performance" in results:
                comparison["performance_comparison"][model] = results["performance"]
        
        # Generate summary
        best_scores = {}
        for metric_type, model_metrics in comparison["detailed_metrics"].items():
            for model, metrics in model_metrics.items():
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if metric_name not in best_scores:
                            best_scores[metric_name] = {"model": model, "value": value}
                        elif value > best_scores[metric_name]["value"]:
                            best_scores[metric_name] = {"model": model, "value": value}
        
        comparison["summary"]["best_scores"] = best_scores
        
        # Generate recommendations
        if best_scores:
            overall_best = {}
            for metric, info in best_scores.items():
                model = info["model"]
                overall_best[model] = overall_best.get(model, 0) + 1
            
            best_model = max(overall_best, key=overall_best.get)
            comparison["recommendations"].append(
                f"Based on the evaluation, {best_model} shows the best overall performance across metrics."
            )
        
        return comparison
    
    def _generate_visualizations(self, all_results: Dict[str, Any], comparison: Dict[str, Any]):
        """Generate visualization plots"""
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Metrics Comparison Bar Chart
        self._plot_metrics_comparison(comparison)
        
        # 2. Performance Timeline
        self._plot_performance_timeline(all_results)
        
        # 3. Memory Usage Comparison
        self._plot_memory_usage(comparison)
        
        # 4. Test Executability Rates
        self._plot_executability_rates(all_results)
    
    def _plot_metrics_comparison(self, comparison: Dict[str, Any]):
        """Plot metrics comparison across models"""
        metrics_data = []
        
        for metric_type, model_data in comparison["detailed_metrics"].items():
            for model, metrics in model_data.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if isinstance(value, dict) and "f1" in value:
                            value = value["f1"]
                        elif isinstance(value, dict):
                            continue
                        
                        if isinstance(value, (int, float)):
                            metrics_data.append({
                                "Model": model,
                                "Metric": f"{metric_type}_{metric_name}",
                                "Value": value
                            })
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            
            plt.figure(figsize=(14, 8))
            pivot_df = df.pivot(index="Metric", columns="Model", values="Value")
            pivot_df.plot(kind="bar", rot=45)
            plt.title("Metrics Comparison Across Models")
            plt.ylabel("Score")
            plt.tight_layout()
            plt.savefig(self.run_dir / "metrics_comparison.png", dpi=300)
            plt.close()
    
    def _plot_performance_timeline(self, all_results: Dict[str, Any]):
        """Plot performance timeline"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        for model, results in all_results.items():
            if "performance" in results:
                perf = results["performance"]
                
                # Time comparison
                uc_time = perf.get("use_case_generation", {}).get("total_time", 0)
                tc_time = perf.get("test_case_generation", {}).get("total_time", 0)
                
                ax1.bar([f"{model}_UC", f"{model}_TC"], [uc_time, tc_time])
                
                # Memory comparison
                uc_mem = perf.get("use_case_generation", {}).get("memory_used", 0)
                tc_mem = perf.get("test_case_generation", {}).get("memory_used", 0)
                
                ax2.bar([f"{model}_UC", f"{model}_TC"], [uc_mem, tc_mem])
        
        ax1.set_title("Generation Time Comparison")
        ax1.set_ylabel("Time (seconds)")
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.set_title("Memory Usage Comparison")
        ax2.set_ylabel("Memory (MB)")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / "performance_timeline.png", dpi=300)
        plt.close()
    
    def _plot_memory_usage(self, comparison: Dict[str, Any]):
        """Plot memory usage comparison"""
        models = []
        memory_values = []
        
        for model, perf in comparison["performance_comparison"].items():
            models.append(model)
            memory_values.append(perf.get("total_memory", 0))
        
        plt.figure(figsize=(10, 6))
        plt.bar(models, memory_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.title("Total Memory Usage by Model")
        plt.ylabel("Memory (MB)")
        plt.xlabel("Model")
        
        for i, v in enumerate(memory_values):
            plt.text(i, v + 5, f"{v:.1f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(self.run_dir / "memory_usage.png", dpi=300)
        plt.close()
    
    def _plot_executability_rates(self, all_results: Dict[str, Any]):
        """Plot test executability rates"""
        models = []
        rates = []
        
        for model, results in all_results.items():
            if "test_case_generation" in results and "test_validation" in results["test_case_generation"]:
                rate = results["test_case_generation"]["test_validation"].get("executability_rate", 0)
                models.append(model)
                rates.append(rate * 100)
        
        if models:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(models, rates, color=['#2ECC71', '#3498DB', '#9B59B6', '#E74C3C'])
            
            # Add percentage labels
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom')
            
            plt.title("Test Case Executability Rates")
            plt.ylabel("Executability Rate (%)")
            plt.xlabel("Model")
            plt.ylim(0, 110)
            
            plt.tight_layout()
            plt.savefig(self.run_dir / "executability_rates.png", dpi=300)
            plt.close()
    
    def _generate_summary_report(self, comparison: Dict[str, Any]) -> str:
        """Generate markdown summary report"""
        lines = [
            f"# LLM Evaluation Summary Report",
            f"**Generated:** {comparison['timestamp']}",
            "",
            "## Models Evaluated",
            *[f"- {model}" for model in comparison['models']],
            "",
            "## Key Findings",
            ""
        ]
        
        # Best scores summary
        if comparison["summary"]["best_scores"]:
            lines.extend([
                "### Best Performing Models by Metric",
                ""
            ])
            
            for metric, info in comparison["summary"]["best_scores"].items():
                lines.append(f"- **{metric}**: {info['model']} (Score: {info['value']:.4f})")
            
            lines.append("")
        
        # Performance summary
        if comparison["performance_comparison"]:
            lines.extend([
                "### Performance Summary",
                "",
                "| Model | Total Time (s) | Total Memory (MB) |",
                "|-------|----------------|-------------------|"
            ])
            
            for model, perf in comparison["performance_comparison"].items():
                time = perf.get("total_time", 0)
                memory = perf.get("total_memory", 0)
                lines.append(f"| {model} | {time:.2f} | {memory:.2f} |")
            
            lines.append("")
        
        # Recommendations
        if comparison["recommendations"]:
            lines.extend([
                "## Recommendations",
                "",
                *comparison["recommendations"],
                ""
            ])
        
        # Visualizations
        lines.extend([
            "## Visualizations",
            "",
            "The following visualizations have been generated:",
            "- `metrics_comparison.png`: Comparative metrics across all models",
            "- `performance_timeline.png`: Time and memory usage breakdown",
            "- `memory_usage.png`: Total memory consumption comparison",
            "- `executability_rates.png`: Test case validity rates",
            ""
        ])
        
        return "\n".join(lines)


async def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM evaluation")
    parser.add_argument("--models", required=True, help="Comma-separated list of models")
    parser.add_argument("--config", default="configs/config.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    models = [m.strip() for m in args.models.split(",")]
    
    # Run evaluation
    runner = EvaluationRunner(models, args.config)
    comparison = await runner.compare_models()
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {runner.run_dir}")
    
    # Print summary
    if comparison["recommendations"]:
        print("\nRecommendations:")
        for rec in comparison["recommendations"]:
            print(f"  - {rec}")


if __name__ == "__main__":
    asyncio.run(main())