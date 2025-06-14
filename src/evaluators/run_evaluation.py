# src/evaluators/run_evaluation.py
"""
Enhanced evaluation runner with all metrics modules integrated
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
import torch
import numpy as np
from .llm_evaluator import LLMEvaluator
import nltk
from nltk.translate.meteor_score import meteor_score
from .dryrun_analyzer import DryrunAnalyzer

# Import all metric modules
from .metrics import (
    QualityMetrics,
    PerformanceMetrics,
    RobotFrameworkMetrics,
    ComparativeAnalysis,
    UserExperienceMetrics
)

# Import other modules
from .ab_testing import ABTestRunner  # KORJAUS: Varmistettu import
from .realtime_monitor import RealtimeMonitor

# Import agents
from ..agents.uc_agent import UCAgent
from ..agents.rf_agent import RFAgent
from ..utils.file_handler import FileHandler

# Standard metrics (for backward compatibility)
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import evaluate


class EnhancedEvaluationRunner:
    """Enhanced evaluation runner with all metrics integrated"""
    
    def __init__(self, models: List[str], config_path: str = "configs/config.yaml", 
                 enable_extended_metrics: bool = True, enable_monitoring: bool = False,
                 enable_llm_evaluation: bool = True):
        """
        Initialize enhanced evaluation runner
        
        Args:
            models: List of model names to evaluate
            config_path: Path to configuration file
            enable_extended_metrics: Enable all extended metrics
            enable_monitoring: Enable realtime monitoring
        """
        self.models = models
        self.config_path = config_path  # <-- LISÄÄ TÄMÄ RIVI!
        self.config = FileHandler.load_yaml(config_path)
        self.results_dir = Path(self.config["paths"]["results_dir"])
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.enable_extended_metrics = enable_extended_metrics
        self.enable_monitoring = enable_monitoring
        self.enable_llm_evaluation = enable_llm_evaluation
        
        # Initialize standard metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bertscore = evaluate.load("bertscore")
        
        # Initialize extended metric modules
        if self.enable_extended_metrics:
            self.quality_metrics = QualityMetrics()
            self.performance_metrics = PerformanceMetrics()
            self.rf_metrics = RobotFrameworkMetrics()
            self.ux_metrics = UserExperienceMetrics()
            self.comparative_analysis = ComparativeAnalysis()
        
        # Initialize monitoring
        if self.enable_monitoring:
            self.monitor = RealtimeMonitor(config_path)
            if self.config.get("monitoring", {}).get("wandb", {}).get("enabled", False):
                self.monitor.start_wandb_logging(
                    project_name=self.config["monitoring"]["wandb"].get("project", "yamk-thesis")
                )
        
        # Initialize LLM evaluator
        if self.enable_llm_evaluation:
            # Use a different model as evaluator to avoid bias
            evaluator_model = "mistral" if "mistral" not in models else "gemma_7b_it_4bit"
            self.llm_evaluator = LLMEvaluator(evaluator_model, config_path)
            logger.info(f"LLM evaluator initialized with model: {evaluator_model}")

        # Create results directory
        self.run_dir = self.results_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Enhanced evaluation runner initialized for models: {models}")
        logger.info(f"Extended metrics: {self.enable_extended_metrics}")
        logger.info(f"Monitoring: {self.enable_monitoring}")
    
    async def evaluate_model(self, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model with all metrics"""
        logger.info(f"Starting enhanced evaluation for model: {model_name}")

        # Tyhjennä muisti ennen evaluointia
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        results = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "use_case_generation": {},
            "test_case_generation": {},
            "metrics": {
                "standard": {},
                "extended": {}
            },
            "performance": {}
        }
        
        try:
            # Phase 1: Use Case Generation with extended monitoring
            if self.enable_extended_metrics:
                self.performance_metrics.start_monitoring()
            
            uc_results = await self._evaluate_use_case_generation(model_name)
            results["use_case_generation"] = uc_results

            # Tyhjennä muisti vaiheiden välillä
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if self.enable_extended_metrics:
                perf_data = self.performance_metrics.stop_monitoring()
                results["performance"]["use_case_generation_detailed"] = perf_data
            
            # Phase 2: Test Case Generation
            if self.enable_extended_metrics:
                self.performance_metrics.start_monitoring()
                
            rf_results = await self._evaluate_test_case_generation(model_name)
            results["test_case_generation"] = rf_results

            # Tyhjennä muisti vaiheiden välillä
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if self.enable_extended_metrics:
                perf_data = self.performance_metrics.stop_monitoring()
                results["performance"]["test_case_generation_detailed"] = perf_data
            
            # Phase 3: Calculate Standard Metrics
            standard_metrics = await self._calculate_standard_metrics(model_name)
            results["metrics"]["standard"] = standard_metrics
            
            # Phase 4: Calculate Extended Metrics
            if self.enable_extended_metrics:
                extended_metrics = await self._calculate_extended_metrics(model_name)
                results["metrics"]["extended"] = extended_metrics
            
            # Phase 5: Performance Analysis
            performance = self._analyze_performance(uc_results, rf_results)
            results["performance"].update(performance)
            
            # Log to monitoring if enabled
            if self.enable_monitoring:
                self._log_to_monitor(model_name, results)

            # Lisää LLM evaluation ennen results palautusta:
            if self.enable_llm_evaluation:
                llm_evaluation = await self._perform_llm_evaluation(model_name)
                results["llm_evaluation"] = llm_evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    # Lisää uusi metodi evaluate_model metodin jälkeen:
    async def _perform_llm_evaluation(self, model_name: str) -> Dict[str, Any]:
        """Perform LLM-based evaluation for a model"""
        logger.info(f"Performing LLM evaluation for {model_name}")
        
        llm_results = {
            "use_case_evaluations": [],
            "test_case_evaluations": [],
            "summary": {}
        }
        
        # Evaluate use cases
        uc_dir = Path(self.config["paths"]["user_stories_dir"]) / model_name
        if uc_dir.exists():
            uc_files = list(uc_dir.glob("*.txt"))
            for uc_file in uc_files[:5]:  # Limit to 5 files for efficiency
                content = FileHandler.read_text_file(str(uc_file))
                evaluation = await self.llm_evaluator.evaluate_use_case(content, model_name)
                llm_results["use_case_evaluations"].append({
                    "file": uc_file.name,
                    "evaluation": evaluation
                })
        
        # Evaluate test cases
        tc_dir = Path(self.config["paths"]["test_cases_dir"]) / model_name
        if tc_dir.exists():
            tc_files = list(tc_dir.glob("*.robot"))
            for tc_file in tc_files[:5]:  # Limit to 5 files
                content = FileHandler.read_text_file(str(tc_file))
                evaluation = await self.llm_evaluator.evaluate_test_case(content, model_name)
                llm_results["test_case_evaluations"].append({
                    "file": tc_file.name,
                    "evaluation": evaluation
                })
        
        # Calculate summary statistics
        if llm_results["use_case_evaluations"]:
            uc_scores = [e["evaluation"].get("overall_score", 0) 
                        for e in llm_results["use_case_evaluations"]]
            llm_results["summary"]["avg_use_case_score"] = np.mean(uc_scores)
        
        if llm_results["test_case_evaluations"]:
            tc_scores = [e["evaluation"].get("overall_score", 0) 
                        for e in llm_results["test_case_evaluations"]]
            llm_results["summary"]["avg_test_case_score"] = np.mean(tc_scores)
        
        return llm_results
    
    async def _evaluate_use_case_generation(self, model_name: str) -> Dict[str, Any]:
        """Evaluate use case generation with extended metrics"""
        logger.info(f"Evaluating use case generation for {model_name}")
        
        # Standard monitoring
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Initialize UC Agent
        uc_agent = UCAgent(model_name)
        
        # Generate use cases
        input_dir = self.config["paths"]["requirements_dir"]
        output_dir = self.config["paths"]["user_stories_dir"]
        
        generation_results = await uc_agent.batch_generate(input_dir, output_dir)
        
        # Calculate performance metrics
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        # Prepare results
        results = {
            "generation_report": {
                "total_files": len(generation_results),
                "successful": sum(1 for r in generation_results if r["status"] == "success"),
                "failed": sum(1 for r in generation_results if r["status"] == "failed"),
                "results": generation_results
            },
            "performance": {
                "total_time": end_time - start_time,
                "memory_used": end_memory - start_memory,
                "avg_time_per_file": (end_time - start_time) / len(generation_results) if generation_results else 0
            }
        }
        
        return results
    
    async def _evaluate_test_case_generation(self, model_name: str) -> Dict[str, Any]:
        """Evaluate test case generation"""
        logger.info(f"Evaluating test case generation for {model_name}")
        
        # Standard monitoring
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Initialize RF Agent
        rf_agent = RFAgent(model_name)
        
        # Generate test cases from use cases
        input_dir = Path(self.config["paths"]["user_stories_dir"]) / model_name
        output_dir = self.config["paths"]["test_cases_dir"]
        
        generation_report = await rf_agent.batch_generate(str(input_dir), output_dir)
        
        # Test execution validation
        test_results = await self._validate_test_execution(model_name)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        return {
            "generation_report": generation_report,
            "test_validation": test_results,
            "performance": {
                "total_time": end_time - start_time,
                "memory_used": end_memory - start_memory
            }
        }
    
    async def _calculate_standard_metrics(self, model_name: str) -> Dict[str, Any]:
        """Calculate standard evaluation metrics"""
        logger.info(f"Calculating standard metrics for {model_name}")
        
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
                candidates = [FileHandler.read_text_file(str(f)) for f in uc_files]
                
                # Custom metrics
                custom_metrics = self._calculate_custom_metrics(candidates, "use_case")
                metrics["use_case_metrics"]["custom"] = custom_metrics
                
                # If we have references, calculate standard NLP metrics
                references = self._load_references("use_cases")
                if references and len(references) == len(candidates):
                    # BLEU
                    bleu = corpus_bleu(candidates, [references])
                    metrics["use_case_metrics"]["bleu"] = bleu.score
                    
                    # ROUGE
                    rouge_scores = []
                    for cand, ref in zip(candidates, references):
                        scores = self.rouge_scorer.score(ref, cand)
                        rouge_scores.append({
                            "rouge1": scores["rouge1"].fmeasure,
                            "rouge2": scores["rouge2"].fmeasure,
                            "rougeL": scores["rougeL"].fmeasure
                        })
                    
                    avg_rouge = {
                        k: np.mean([s[k] for s in rouge_scores])
                        for k in ["rouge1", "rouge2", "rougeL"]
                    }
                    metrics["use_case_metrics"]["rouge"] = avg_rouge
                    
                    # BERTScore
                    if torch.cuda.is_available():
                        P, R, F1 = bert_score(candidates, references, lang="en", device="cuda")
                    else:
                        P, R, F1 = bert_score(candidates, references, lang="en", device="cpu")
                    
                    metrics["use_case_metrics"]["bertscore"] = {
                        "precision": P.mean().item(),
                        "recall": R.mean().item(),
                        "f1": F1.mean().item()
                    }

                    # _calculate_standard_metrics metodissa, lisää BLEU/ROUGE/BERTScore jälkeen:
                    # METEOR
                    if references and len(references) == len(candidates):
                        # Ensure NLTK data is downloaded
                        try:
                            nltk.download('wordnet', quiet=True)
                            nltk.download('omw-1.4', quiet=True)
                        except:
                            pass
                        
                        meteor_scores = []
                        for cand, ref in zip(candidates, references):
                            # METEOR expects tokenized input
                            score = meteor_score([ref.split()], cand.split())
                            meteor_scores.append(score)
                        
                        metrics["use_case_metrics"]["meteor"] = {
                            "mean": np.mean(meteor_scores),
                            "min": np.min(meteor_scores),
                            "max": np.max(meteor_scores),
                            "std": np.std(meteor_scores)
                        }
        
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
    
    async def _calculate_extended_metrics(self, model_name: str) -> Dict[str, Any]:
        """Calculate all extended metrics"""
        logger.info(f"Calculating extended metrics for {model_name}")
        
        extended_metrics = {}
        
        # Load generated content
        uc_dir = Path(self.config["paths"]["user_stories_dir"]) / model_name
        tc_dir = Path(self.config["paths"]["test_cases_dir"]) / model_name
        
        # Quality metrics for use cases
        if uc_dir.exists():
            uc_files = list(uc_dir.glob("*.txt"))
            if uc_files:
                uc_texts = [FileHandler.read_text_file(str(f)) for f in uc_files]
                
                # Quality metrics
                quality_results = self.quality_metrics.calculate_all_metrics(uc_texts)
                extended_metrics["quality"] = quality_results
                
                # UX metrics
                ux_results = self.ux_metrics.calculate_all_metrics(uc_texts, "use_case")
                extended_metrics["user_experience"] = ux_results
        
        # Robot Framework specific metrics
        if tc_dir.exists():
            tc_files = list(tc_dir.glob("*.robot"))
            if tc_files:
                rf_results = self.rf_metrics.calculate_all_metrics(tc_files)
                extended_metrics["robot_framework"] = rf_results
                
                # UX metrics for test cases
                tc_texts = [FileHandler.read_text_file(str(f)) for f in tc_files]
                tc_ux_results = self.ux_metrics.calculate_all_metrics(tc_texts, "test_case")
                extended_metrics["test_case_ux"] = tc_ux_results
        
        # System information
        extended_metrics["system_info"] = self.performance_metrics.get_system_info()
        
        return extended_metrics
    
    async def compare_models(self) -> Dict[str, Any]:
        """Run evaluation for all models and compare with extended analysis"""
        logger.info(f"Starting enhanced comparison of {len(self.models)} models")
        
        all_results = {}
        
        # Evaluate each model
        for model in self.models:
            # Tyhjennä muisti ennen uutta mallia
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"GPU memory cleared before evaluating: {model}")

            logger.info(f"Evaluating model: {model}")

            try:
                results = await self.evaluate_model(model)
                all_results[model] = results
                
                # Save individual model results
                model_file = self.run_dir / f"{model}_results.json"
                FileHandler.save_json(results, str(model_file))
                
                # Tyhjennä muisti mallin jälkeen
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info(f"GPU memory cleared after evaluating: {model}")
            
            except Exception as e:
                logger.error(f"Error evaluating model {model}: {str(e)}")
                all_results[model] = {"error": str(e)}
                
                # Tyhjennä muisti virheen jälkeenkin
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info(f"GPU memory cleared after error in: {model}")
        
        # Generate standard comparison report
        comparison = self._generate_comparison_report(all_results)
        
        # Add extended comparative analysis
        if self.enable_extended_metrics:
            comparative_results = self.comparative_analysis.analyze_models(all_results)
            comparison["extended_analysis"] = comparative_results
        
        # *** TÄHÄN KOHTAAN DRYRUN-ANALYYSI ***
        # Run dryrun analysis
        logger.info("Running Robot Framework dryrun analysis...")
        dryrun_analyzer = DryrunAnalyzer(self.config_path)
        dryrun_results = await dryrun_analyzer.analyze_all_models()
        
        # Generate dryrun report
        dryrun_report_dir = self.run_dir / "dryrun_analysis"
        dryrun_report = dryrun_analyzer.generate_report(dryrun_results, dryrun_report_dir)
        
        # Add to comparison
        comparison["dryrun_analysis"] = dryrun_results

        # Save comparison
        comparison_file = self.run_dir / "comparison_report.json"
        FileHandler.save_json(comparison, str(comparison_file))
        
        # Generate visualizations
        self._generate_enhanced_visualizations(all_results, comparison)
        
        # Generate comprehensive report
        summary = self._generate_comprehensive_report(comparison)
        summary_file = self.run_dir / "comprehensive_report.md"
        FileHandler.save_text_file(summary, str(summary_file))
        
        # Run A/B tests if exactly 2 models
        if len(self.models) == 2 and self.enable_extended_metrics:
            logger.info("Running A/B test between models")
            ab_runner = ABTestRunner()
            try:
                ab_results = await ab_runner.run_ab_test(
                    model_a=self.models[0],
                    model_b=self.models[1],
                    test_data_path=self.config["paths"]["requirements_dir"],
                    output_dir=str(self.run_dir / "ab_test")
                )
                comparison["ab_test_results"] = ab_results
            except Exception as e:
                logger.error(f"A/B test failed: {str(e)}")
                comparison["ab_test_error"] = str(e)
        
        logger.success(f"Enhanced evaluation complete! Results saved to: {self.run_dir}")
        
        return comparison
    
    def _generate_comparison_report(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate standard comparison report"""
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "models": list(all_results.keys()),
            "summary": {},
            "detailed_metrics": {
                "standard": {},
                "extended": {}
            },
            "performance_comparison": {},
            "recommendations": []
        }
        
        # Extract metrics for comparison
        for model, results in all_results.items():
            if "metrics" in results:
                # Standard metrics
                if "standard" in results["metrics"]:
                    for metric_type, metrics in results["metrics"]["standard"].items():
                        if metric_type not in comparison["detailed_metrics"]["standard"]:
                            comparison["detailed_metrics"]["standard"][metric_type] = {}
                        comparison["detailed_metrics"]["standard"][metric_type][model] = metrics
                
                # Extended metrics
                if "extended" in results["metrics"] and self.enable_extended_metrics:
                    for metric_type, metrics in results["metrics"]["extended"].items():
                        if metric_type not in comparison["detailed_metrics"]["extended"]:
                            comparison["detailed_metrics"]["extended"][metric_type] = {}
                        comparison["detailed_metrics"]["extended"][metric_type][model] = metrics
            
            # Performance comparison
            if "performance" in results:
                comparison["performance_comparison"][model] = results["performance"]
        
        # Generate summary
        best_scores = self._find_best_scores(comparison["detailed_metrics"])
        comparison["summary"]["best_scores"] = best_scores
        
        # Generate recommendations
        comparison["recommendations"] = self._generate_recommendations(best_scores, all_results)
        
        return comparison
    
    def _find_best_scores(self, detailed_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Find best scores across all metrics"""
        best_scores = {}
        
        # Process standard metrics
        for metric_type, model_metrics in detailed_metrics.get("standard", {}).items():
            for model, metrics in model_metrics.items():
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        key = f"{metric_type}_{metric_name}"
                        if key not in best_scores or value > best_scores[key]["value"]:
                            best_scores[key] = {"model": model, "value": value}
        
        # Process extended metrics if available
        if "extended" in detailed_metrics and self.enable_extended_metrics:
            # Quality metrics
            quality_metrics = detailed_metrics["extended"].get("quality", {})
            for model, metrics in quality_metrics.items():
                if "perplexity" in metrics:
                    # Lower perplexity is better
                    perp_value = metrics["perplexity"].get("mean_perplexity", float('inf'))
                    key = "quality_perplexity"
                    if key not in best_scores or perp_value < best_scores[key]["value"]:
                        best_scores[key] = {"model": model, "value": perp_value}
        
        return best_scores
    
    def _generate_recommendations(self, best_scores: Dict[str, Any], all_results: Dict[str, Any]) -> List[str]:
        """Generate enhanced recommendations"""
        recommendations = []
        
        if not best_scores:
            recommendations.append("No significant differences found between models.")
            return recommendations
        
        # Count wins per model
        model_wins = {}
        for metric, info in best_scores.items():
            model = info["model"]
            model_wins[model] = model_wins.get(model, 0) + 1
        
        # Overall best model
        if model_wins:
            best_model = max(model_wins, key=model_wins.get)
            recommendations.append(
                f"Overall recommendation: **{best_model}** (best in {model_wins[best_model]} metrics)"
            )
        
        # Specific recommendations based on use case
        # Quality vs Performance trade-off
        quality_winner = None
        performance_winner = None
        
        for metric, info in best_scores.items():
            if "completeness" in metric or "validity" in metric:
                quality_winner = info["model"]
            elif "time" in metric or "memory" in metric:
                if "time" in metric:
                    # For time, lower is better, so we need to find the actual winner
                    time_scores = {}
                    for model, results in all_results.items():
                        if "performance" in results:
                            time_scores[model] = results["performance"].get("total_time", float('inf'))
                    if time_scores:
                        performance_winner = min(time_scores, key=time_scores.get)
        
        if quality_winner and performance_winner and quality_winner != performance_winner:
            recommendations.append(
                f"Trade-off detected: {quality_winner} has better quality, "
                f"while {performance_winner} has better performance."
            )
        
        # Extended metric recommendations
        if self.enable_extended_metrics:
            recommendations.append("\n**Extended Metrics Insights:**")
            
            # Check UX metrics
            for model, results in all_results.items():
                if "metrics" in results and "extended" in results["metrics"]:
                    ux_data = results["metrics"]["extended"].get("user_experience", {})
                    if ux_data:
                        readability = ux_data.get("readability", {}).get("overall_readability", "Unknown")
                        if readability == "Good - Easy to understand":
                            recommendations.append(f"- {model} produces highly readable content")
        
        return recommendations
    
    def _generate_enhanced_visualizations(self, all_results: Dict[str, Any], comparison: Dict[str, Any]):
        """Generate enhanced visualizations with extended metrics"""
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Standard metrics comparison
        self._plot_standard_metrics_comparison(comparison)
        
        # 2. Extended metrics visualizations
        if self.enable_extended_metrics:
            self._plot_extended_metrics(all_results)
            self._plot_quality_metrics_heatmap(all_results)
            self._plot_ux_metrics_radar(all_results)
        
        # 3. Performance analysis
        self._plot_performance_analysis(all_results)
        
        # 4. Comprehensive dashboard
        self._create_comprehensive_dashboard(all_results, comparison)
    
    def _plot_extended_metrics(self, all_results: Dict[str, Any]):
        """Plot extended metrics comparison"""
        if not self.enable_extended_metrics:
            return
        
        # Prepare data
        metrics_data = []
        
        for model, results in all_results.items():
            if "metrics" in results and "extended" in results["metrics"]:
                extended = results["metrics"]["extended"]
                
                # Quality metrics
                if "quality" in extended:
                    quality = extended["quality"]
                    if "perplexity" in quality:
                        metrics_data.append({
                            "Model": model,
                            "Metric": "Perplexity",
                            "Value": quality["perplexity"].get("mean_perplexity", 0)
                        })
                    if "diversity" in quality:
                        metrics_data.append({
                            "Model": model,
                            "Metric": "Diversity (Distinct-2)",
                            "Value": quality["diversity"].get("distinct_2", 0)
                        })
                    if "coherence" in quality:
                        metrics_data.append({
                            "Model": model,
                            "Metric": "Coherence",
                            "Value": quality["coherence"].get("mean_coherence", 0)
                        })
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            
            plt.figure(figsize=(12, 6))
            pivot_df = df.pivot(index="Metric", columns="Model", values="Value")
            ax = pivot_df.plot(kind="bar", rot=0)
            
            plt.title("Extended Quality Metrics Comparison")
            plt.ylabel("Score")
            plt.legend(title="Model")
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f')
            
            plt.tight_layout()
            plt.savefig(self.run_dir / "extended_metrics_comparison.png", dpi=300)
            plt.close()
    
    def _plot_quality_metrics_heatmap(self, all_results: Dict[str, Any]):
        """Create heatmap of quality metrics"""
        if not self.enable_extended_metrics:
            return
        
        # Prepare data matrix
        models = list(all_results.keys())
        metrics = ["Perplexity", "Coherence", "Diversity", "Readability", "Clarity", "Completeness"]
        
        data_matrix = []
        
        for model in models:
            row = []
            results = all_results[model]
            
            if "metrics" in results:
                # Get extended metrics
                extended = results["metrics"].get("extended", {})
                
                # Perplexity (inverse for visualization - lower is better)
                perp = extended.get("quality", {}).get("perplexity", {}).get("mean_perplexity", 100)
                row.append(1 / perp if perp > 0 else 0)
                
                # Coherence
                row.append(extended.get("quality", {}).get("coherence", {}).get("mean_coherence", 0))
                
                # Diversity
                row.append(extended.get("quality", {}).get("diversity", {}).get("distinct_2", 0))
                
                # Readability (from UX metrics)
                ux = extended.get("user_experience", {})
                readability = ux.get("readability", {}).get("flesch_reading_ease", {}).get("score", 0)
                row.append(readability / 100)  # Normalize to 0-1
                
                # Clarity
                row.append(ux.get("clarity", {}).get("overall_clarity", 0))
                
                # Completeness (from standard metrics)
                completeness = results["metrics"].get("standard", {}).get("use_case_metrics", {}).get("custom", {}).get("completeness", 0)
                row.append(completeness)
            else:
                row = [0] * len(metrics)
            
            data_matrix.append(row)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            data_matrix,
            xticklabels=metrics,
            yticklabels=models,
            annot=True,
            fmt='.3f',
            cmap='YlGnBu',
            cbar_kws={'label': 'Score'}
        )
        
        plt.title("Quality Metrics Heatmap")
        plt.tight_layout()
        plt.savefig(self.run_dir / "quality_metrics_heatmap.png", dpi=300)
        plt.close()
    
    def _plot_ux_metrics_radar(self, all_results: Dict[str, Any]):
        """Create radar chart for UX metrics"""
        if not self.enable_extended_metrics:
            return
        
        import plotly.graph_objects as go
        
        # Prepare data
        categories = ['Readability', 'Clarity', 'Actionability', 'Completeness', 'Usability']
        
        fig = go.Figure()
        
        for model, results in all_results.items():
            if "metrics" in results and "extended" in results["metrics"]:
                ux = results["metrics"]["extended"].get("user_experience", {})
                
                values = [
                    ux.get("readability", {}).get("flesch_reading_ease", {}).get("score", 0) / 100,
                    ux.get("clarity", {}).get("overall_clarity", 0),
                    ux.get("actionability", {}).get("overall_actionability", 0),
                    ux.get("completeness", {}).get("completeness_score", 0),
                    ux.get("usability", {}).get("user_friendliness", 0)
                ]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=model
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="User Experience Metrics Comparison"
        )
        
        fig.write_html(str(self.run_dir / "ux_metrics_radar.html"))
    
    def _create_comprehensive_dashboard(self, all_results: Dict[str, Any], comparison: Dict[str, Any]):
        """Create comprehensive dashboard as HTML"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Evaluation Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
                .metric-card {{ 
                    display: inline-block; 
                    background-color: #fff; 
                    border: 1px solid #ddd; 
                    border-radius: 5px; 
                    padding: 15px; 
                    margin: 10px;
                    width: 200px;
                }}
                .winner {{ background-color: #d4edda; }}
                .section {{ margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>LLM Evaluation Dashboard</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Models evaluated: {', '.join(self.models)}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <div class="metrics-container">
        """
        
        # Add summary metrics
        if "summary" in comparison and "best_scores" in comparison["summary"]:
            for metric, info in comparison["summary"]["best_scores"].items():
                html_content += f"""
                    <div class="metric-card winner">
                        <h3>{metric.replace('_', ' ').title()}</h3>
                        <p><strong>Winner:</strong> {info['model']}</p>
                        <p><strong>Score:</strong> {info['value']:.4f}</p>
                    </div>
                """
        
        html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>Detailed Results</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>UC Success Rate</th>
                        <th>TC Success Rate</th>
                        <th>Total Time (s)</th>
                        <th>Memory (MB)</th>
                    </tr>
        """
        
        # Add detailed results
        for model, results in all_results.items():
            uc_success = results.get("use_case_generation", {}).get("generation_report", {}).get("successful", 0)
            uc_total = results.get("use_case_generation", {}).get("generation_report", {}).get("total_files", 1)
            tc_success = results.get("test_case_generation", {}).get("generation_report", {}).get("successful", 0)
            tc_total = results.get("test_case_generation", {}).get("generation_report", {}).get("total_files", 1)
            
            perf = results.get("performance", {})
            
            html_content += f"""
                    <tr>
                        <td>{model}</td>
                        <td>{uc_success}/{uc_total} ({uc_success/uc_total*100:.1f}%)</td>
                        <td>{tc_success}/{tc_total} ({tc_success/tc_total*100:.1f}%)</td>
                        <td>{perf.get('total_time', 0):.2f}</td>
                        <td>{perf.get('total_memory', 0):.2f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
        """
        
        # Add recommendations
        for rec in comparison.get("recommendations", []):
            html_content += f"<li>{rec}</li>"
        
        html_content += """
                </ul>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <p>See generated PNG files in the results directory.</p>
            </div>
        </body>
        </html>
        """
        
        # Save dashboard
        dashboard_file = self.run_dir / "evaluation_dashboard.html"
        FileHandler.save_text_file(html_content, str(dashboard_file))
        logger.info(f"Dashboard saved to: {dashboard_file}")
    
    def _generate_comprehensive_report(self, comparison: Dict[str, Any]) -> str:
        """Generate comprehensive markdown report"""
        lines = [
            f"# Comprehensive LLM Evaluation Report",
            f"**Generated:** {comparison['timestamp']}",
            f"**Models:** {', '.join(comparison['models'])}",
            "",
            "## Executive Summary",
            ""
        ]

        # Lisää LLM-pohjainen arviointi yhteenvetoon
        if "llm_based_comparison" in comparison:
            llm_comp = comparison["llm_based_comparison"]
            lines.extend([
                "### LLM-based Evaluation Results",
                ""
            ])
            
            if "ranking" in llm_comp:
                lines.append("**Model Ranking (by LLM evaluation):**")
                for rank_info in llm_comp["ranking"]:
                    lines.append(f"{rank_info['rank']}. {rank_info['model']} "
                            f"(Score: {rank_info['score']}/100) - {rank_info['reason']}")
                lines.append("")
        
        # Yhdistä numeeriset ja LLM-pohjaiset tulokset
        lines.extend([
            "## Combined Evaluation Results",
            "",
            "| Model | BLEU | ROUGE-L | BERTScore | METEOR | LLM Score | Overall |",
            "|-------|------|---------|-----------|---------|-----------|---------|"
        ])
        
        # Päivitä _generate_comprehensive_report metodia:
        # Lisää Dryrun Analysis osio:
        if "dryrun_analysis" in comparison:
            lines.extend([
                "",
                "## Robot Framework Dryrun Analysis",
                "",
                f"**Overall Success Rate:** {comparison['dryrun_analysis']['summary']['overall_success_rate']:.1%}",
                "",
                "| Model | Success Rate | Failed Tests | Most Common Error |",
                "|-------|--------------|--------------|-------------------|"
            ])
            
            for model, results in comparison['dryrun_analysis']['by_model'].items():
                # Get most common error type
                error_patterns = comparison['dryrun_analysis']['error_analysis']['model_error_patterns'].get(model, {})
                most_common_error = max(error_patterns.items(), key=lambda x: x[1])[0] if error_patterns else "None"
                
                lines.append(
                    f"| {model} | {results['success_rate']:.1%} | "
                    f"{results['failed']}/{results['total_files']} | {most_common_error} |"
                )

        # Best model overall
        if "recommendations" in comparison and comparison["recommendations"]:
            lines.append(comparison["recommendations"][0])
            lines.append("")
        
        # Key findings
        lines.extend([
            "### Key Findings",
            ""
        ])
        
        if "summary" in comparison and "best_scores" in comparison["summary"]:
            for metric, info in comparison["summary"]["best_scores"].items():
                lines.append(f"- **{metric.replace('_', ' ').title()}**: {info['model']} (Score: {info['value']:.4f})")
        
        lines.append("")
        
        # Performance summary
        if "performance_comparison" in comparison:
            lines.extend([
                "## Performance Analysis",
                "",
                "| Model | Total Time (s) | Memory (MB) | Files/Second |",
                "|-------|----------------|-------------|--------------|"
            ])
            
            for model, perf in comparison["performance_comparison"].items():
                time = perf.get("total_time", 0)
                memory = perf.get("total_memory", 0)
                files_per_sec = perf.get("files_per_second", 0)
                lines.append(f"| {model} | {time:.2f} | {memory:.2f} | {files_per_sec:.2f} |")
        
        lines.append("")
        
        # Extended metrics summary
        if self.enable_extended_metrics and "extended_analysis" in comparison:
            lines.extend([
                "## Extended Analysis",
                ""
            ])
            
            extended = comparison["extended_analysis"]
            
            # Consistency
            if "consistency" in extended:
                lines.extend([
                    "### Consistency Analysis",
                    ""
                ])
                for model, consistency in extended["consistency"].items():
                    lines.append(f"**{model}:**")
                    lines.append(f"- Output Stability: {consistency.get('output_stability', 0):.2%}")
                    lines.append(f"- Structure Variance: {consistency.get('structure_variance', 0):.2%}")
                    lines.append("")
            
            # Trade-offs
            if "trade_offs" in extended:
                lines.extend([
                    "### Quality vs Performance Trade-offs",
                    ""
                ])
                for model, trade_off in extended["trade_offs"].items():
                    lines.append(f"**{model}:**")
                    lines.append(f"- Quality Score: {trade_off.get('quality_score', 0):.2%}")
                    lines.append(f"- Speed-Quality Ratio: {trade_off.get('speed_quality_ratio', 0):.3f}")
                    lines.append(f"- Efficiency Score: {trade_off.get('efficiency_score', 0):.3f}")
                    lines.append("")
            
            # Rankings
            if "ranking" in extended:
                lines.extend([
                    "### Model Rankings",
                    ""
                ])
                for rank_info in extended["ranking"]:
                    lines.append(f"{rank_info['rank']}. **{rank_info['model']}** (Score: {rank_info['composite_score']:.3f})")
                lines.append("")
        
        # A/B test results
        if "ab_test_results" in comparison:
            lines.extend([
                "## A/B Test Results",
                ""
            ])
            
            ab_results = comparison["ab_test_results"]
            if "report" in ab_results:
                lines.append(ab_results["report"])
        
        # Detailed recommendations
        lines.extend([
            "## Detailed Recommendations",
            ""
        ])
        
        for rec in comparison.get("recommendations", []):
            lines.append(f"- {rec}")
        
        lines.extend([
            "",
            "## Artifacts Generated",
            "",
            "The following files have been generated:",
            "- Individual model results (JSON)",
            "- Comparison report (JSON)",
            "- Metric comparison charts (PNG)",
            "- Quality metrics heatmap (PNG)",
            "- UX metrics radar chart (HTML)",
            "- Comprehensive dashboard (HTML)",
            ""
        ])
        
        return "\n".join(lines)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    
    def _log_to_monitor(self, model_name: str, results: Dict[str, Any]):
        """Log results to monitoring system"""
        if not self.enable_monitoring:
            return
        
        # Hae system info performance_metrics:stä
        system_info = self.performance_metrics.get_system_info() if self.enable_extended_metrics else {}
        
        # Prepare metrics for monitoring
        monitor_metrics = {
            "timestamp": datetime.now(),
            "models": {
                model_name: {
                    "status": "completed",
                    "timestamp": datetime.now()
                }
            },
            "system": system_info  # Lisää tämä
        }
       
        # Add key metrics
        if "metrics" in results:
            standard = results["metrics"].get("standard", {})
            if "use_case_metrics" in standard:
                uc_metrics = standard["use_case_metrics"]
                monitor_metrics["models"][model_name]["completeness"] = uc_metrics.get("custom", {}).get("completeness", 0)
            
            if "test_case_metrics" in standard:
                tc_metrics = standard["test_case_metrics"]
                monitor_metrics["models"][model_name]["test_validity"] = tc_metrics.get("syntax_validity", {}).get("validity_rate", 0)
        
        # Add performance metrics
        if "performance" in results:
            perf = results["performance"]
            monitor_metrics["models"][model_name]["generation_time"] = perf.get("total_time", 0)
            monitor_metrics["models"][model_name]["memory_usage"] = perf.get("total_memory", 0)
        
        # Log to monitoring system
        self.monitor.log_to_wandb(monitor_metrics)
    
    # Keep original methods for backward compatibility
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
        
        if "generation_report" in uc_results:
            total_files = uc_results["generation_report"]["total_files"]
            total_time = uc_results["performance"]["total_time"]
            if total_time > 0:
                performance["files_per_second"] = total_files / total_time
        
        return performance
    
    def _load_references(self, content_type: str) -> List[str]:
        """Load reference texts for comparison"""
        # Placeholder for loading human-written references
        # In production, implement actual reference loading
        return []
    
    # Standard visualization methods
    def _plot_standard_metrics_comparison(self, comparison: Dict[str, Any]):
        """Plot standard metrics comparison"""
        metrics_data = []
        
        for metric_type, model_data in comparison["detailed_metrics"].get("standard", {}).items():
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
            plt.title("Standard Metrics Comparison")
            plt.ylabel("Score")
            plt.tight_layout()
            plt.savefig(self.run_dir / "standard_metrics_comparison.png", dpi=300)
            plt.close()
    
    def _plot_performance_analysis(self, all_results: Dict[str, Any]):
        """Plot performance analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        models = []
        times = []
        memories = []
        
        for model, results in all_results.items():
            if "performance" in results:
                models.append(model)
                times.append(results["performance"].get("total_time", 0))
                memories.append(results["performance"].get("total_memory", 0))
        
        # Time comparison
        ax1.bar(models, times)
        ax1.set_title("Total Generation Time")
        ax1.set_ylabel("Time (seconds)")
        ax1.set_xlabel("Model")
        
        # Memory comparison
        ax2.bar(models, memories)
        ax2.set_title("Total Memory Usage")
        ax2.set_ylabel("Memory (MB)")
        ax2.set_xlabel("Model")
        
        plt.tight_layout()
        plt.savefig(self.run_dir / "performance_analysis.png", dpi=300)
        plt.close()


async def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced LLM evaluation")
    parser.add_argument("--models", required=True, help="Comma-separated list of models")
    parser.add_argument("--config", default="configs/config.yaml", help="Configuration file")
    parser.add_argument("--extended-metrics", action="store_true", help="Enable extended metrics")
    parser.add_argument("--monitoring", action="store_true", help="Enable realtime monitoring")
    parser.add_argument("--ab-test", action="store_true", help="Run A/B test (for 2 models)")
    
    args = parser.parse_args()
    
    models = [m.strip() for m in args.models.split(",")]
    
    # Run evaluation
    runner = EnhancedEvaluationRunner(
        models, 
        args.config,
        enable_extended_metrics=args.extended_metrics,
        enable_monitoring=args.monitoring
    )
    
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