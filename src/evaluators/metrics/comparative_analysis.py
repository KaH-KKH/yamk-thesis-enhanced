# src/evaluators/metrics/comparative_analysis.py
"""
Comparative analysis metrics for model evaluation
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
import pandas as pd
from loguru import logger


class ComparativeAnalysis:
    """Perform comparative analysis between models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def analyze_models(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive comparative analysis of multiple models"""
        
        analysis = {
            "consistency": self._analyze_consistency(model_results),
            "trade_offs": self._analyze_trade_offs(model_results),
            "scalability": self._analyze_scalability(model_results),
            "robustness": self._analyze_robustness(model_results),
            "statistical_significance": self._statistical_tests(model_results),
            "ranking": self._rank_models(model_results),
            "recommendations": self._generate_recommendations(model_results)
        }
        
        return analysis
    
    def _calculate_safe_efficiency(self, quality_score, total_time, total_memory):
        """Calculate efficiency score safely avoiding log errors"""
        try:
            # Varmista että arvot ovat positiivisia ja järkeviä
            quality_score = max(0.0, quality_score) if not np.isnan(quality_score) else 0.0
            total_time = max(0.1, total_time) if not np.isnan(total_time) else 0.1  # Min 0.1s
            total_memory = max(0.1, total_memory) if not np.isnan(total_memory) else 0.1  # Min 0.1MB
            
            # Laske turvallisesti
            time_factor = np.log1p(total_time)
            memory_factor = np.log1p(total_memory)
            
            if time_factor <= 0 or memory_factor <= 0:
                return 0.0
            
            efficiency = quality_score / (time_factor * memory_factor)
            
            # Tarkista että tulos on validi
            if np.isnan(efficiency) or np.isinf(efficiency):
                return 0.0
            
            return float(efficiency)
            
        except Exception:
            return 0.0
    
    def _analyze_consistency(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze consistency of model outputs"""
        consistency_metrics = {}
        
        for model_name, results in model_results.items():
            # Extract relevant metrics for consistency analysis
            if "metrics" in results:
                metrics = results["metrics"]
                
                # Calculate variance across different runs (if available)
                # For now, analyze consistency within the generated outputs
                use_case_metrics = metrics.get("use_case_metrics", {})
                
                if "custom" in use_case_metrics:
                    consistency_metrics[model_name] = {
                        "completeness_consistency": use_case_metrics["custom"].get("completeness", 0),
                        "structure_variance": self._calculate_structure_variance(results),
                        "output_stability": self._calculate_output_stability(results)
                    }
        
        return consistency_metrics
    
    def _calculate_structure_variance(self, results: Dict[str, Any]) -> float:
        """Calculate variance in output structure"""
        # Analyze variance in generated content structure
        if "use_case_generation" in results:
            generation_report = results["use_case_generation"].get("generation_report", {})
            success_rate = generation_report.get("successful", 0) / max(generation_report.get("total_files", 1), 1)
            return 1.0 - success_rate  # Lower is better
        return 0.0
    
    def _calculate_output_stability(self, results: Dict[str, Any]) -> float:
        """Calculate stability of outputs"""
        # This would ideally compare multiple runs
        # For now, use success rate as proxy
        stability_score = 1.0
        
        for phase in ["use_case_generation", "test_case_generation"]:
            if phase in results:
                report = results[phase].get("generation_report", {})
                if report:
                    success_rate = report.get("successful", 0) / max(report.get("total_files", 1), 1)
                    stability_score *= success_rate
        
        return stability_score
    
    def _analyze_trade_offs(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trade-offs between quality and performance"""
        trade_offs = {}
        
        for model_name, results in model_results.items():
            if "metrics" in results and "performance" in results:
                # Extract quality score (average of all quality metrics)
                quality_metrics = []
                
                # Use case metrics
                uc_metrics = results["metrics"].get("use_case_metrics", {})
                if "custom" in uc_metrics:
                    quality_metrics.append(uc_metrics["custom"].get("completeness", 0))
                
                # Test case metrics  
                tc_metrics = results["metrics"].get("test_case_metrics", {})
                if "syntax_validity" in tc_metrics:
                    quality_metrics.append(tc_metrics["syntax_validity"].get("validity_rate", 0))
                
                quality_score = np.mean(quality_metrics) if quality_metrics else 0
                
                # Performance metrics
                perf = results["performance"]
                total_time = perf.get("total_time", 1)
                total_memory = perf.get("total_memory", 1)
                
                trade_offs[model_name] = {
                    "quality_score": quality_score,
                    "speed_quality_ratio": quality_score / max(total_time, 0.1),  # Higher is better
                    "memory_quality_ratio": quality_score / max(total_memory, 1),  # Higher is better
                    # "efficiency_score": quality_score / (np.log1p(total_time) * np.log1p(total_memory))
                    "efficiency_score": self._calculate_safe_efficiency(quality_score, total_time, total_memory)
                }
        
        return trade_offs
    
    def _analyze_scalability(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how models scale with input size"""
        scalability = {}
        
        for model_name, results in model_results.items():
            if "performance" in results:
                perf = results["performance"]
                
                # Calculate scalability metrics
                uc_perf = perf.get("use_case_generation", {})
                tc_perf = perf.get("test_case_generation", {})
                
                # Files per second as scalability indicator
                uc_files_per_sec = 1 / uc_perf.get("avg_time_per_file", 1) if "avg_time_per_file" in uc_perf else 0
                
                scalability[model_name] = {
                    "files_per_second": perf.get("files_per_second", uc_files_per_sec),
                    "memory_efficiency": self._calculate_memory_efficiency(perf),
                    "time_complexity": self._estimate_time_complexity(results)
                }
        
        return scalability
    
    def _calculate_memory_efficiency(self, performance: Dict[str, Any]) -> float:
        """Calculate memory efficiency score"""
        total_memory = performance.get("total_memory", 0)
        
        # Consider both generation phases
        uc_memory = performance.get("use_case_generation", {}).get("memory_used", 0)
        tc_memory = performance.get("test_case_generation", {}).get("memory_used", 0)
        
        # Efficiency: inverse of memory per operation
        operations = 2  # UC generation + TC generation
        memory_per_op = (uc_memory + tc_memory) / operations
        
        return 1000 / max(memory_per_op, 1)  # Higher is better
    
    def _estimate_time_complexity(self, results: Dict[str, Any]) -> str:
        """Estimate time complexity based on performance data"""
        # Simplified estimation based on available data
        if "performance" in results:
            perf = results["performance"]
            total_time = perf.get("total_time", 0)
            
            # This is a simplified heuristic
            if total_time < 60:
                return "O(n) - Linear"
            elif total_time < 300:
                return "O(n log n) - Linearithmic"
            else:
                return "O(n²) - Quadratic"
        
        return "Unknown"
    
    def _analyze_robustness(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze model robustness and error handling"""
        robustness = {}
        
        for model_name, results in model_results.items():
            error_count = 0
            total_attempts = 0
            
            # Check generation reports for errors
            for phase in ["use_case_generation", "test_case_generation"]:
                if phase in results:
                    report = results[phase].get("generation_report", {})
                    if report:
                        total_attempts += report.get("total_files", 0)
                        error_count += report.get("failed", 0)
            
            # Calculate error rate
            error_rate = error_count / max(total_attempts, 1)
            
            # Check for error details
            has_error_info = "error" in results
            
            robustness[model_name] = {
                "error_rate": error_rate,
                "success_rate": 1 - error_rate,
                "graceful_failure": has_error_info,  # Model provides error details
                "reliability_score": (1 - error_rate) * 100
            }
        
        return robustness
    
    def _statistical_tests(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical significance tests between models"""
        statistical_results = {}
        
        # Extract comparable metrics from all models
        model_names = list(model_results.keys())
        
        if len(model_names) >= 2:
            # Prepare data for comparison
            metrics_data = {}
            
            for model_name, results in model_results.items():
                if "metrics" in results:
                    # Extract key metrics
                    uc_completeness = results["metrics"].get("use_case_metrics", {}).get("custom", {}).get("completeness", 0)
                    tc_validity = results["metrics"].get("test_case_metrics", {}).get("syntax_validity", {}).get("validity_rate", 0)
                    
                    metrics_data[model_name] = {
                        "completeness": uc_completeness,
                        "validity": tc_validity,
                        "combined": (uc_completeness + tc_validity) / 2
                    }
            
            # Perform pairwise comparisons
            for i, model1 in enumerate(model_names):
                for model2 in model_names[i+1:]:
                    comparison_key = f"{model1}_vs_{model2}"
                    
                    # For demonstration, we'll use the combined metric
                    # In practice, you'd have multiple samples per model
                    score1 = metrics_data[model1]["combined"]
                    score2 = metrics_data[model2]["combined"]
                    
                    # Calculate effect size (Cohen's d)
                    effect_size = abs(score1 - score2) / max(np.std([score1, score2]), 0.1)
                    
                    statistical_results[comparison_key] = {
                        "model1": model1,
                        "model2": model2,
                        "score_difference": score1 - score2,
                        "effect_size": effect_size,
                        "effect_interpretation": self._interpret_effect_size(effect_size),
                        "winner": model1 if score1 > score2 else model2
                    }
        
        return statistical_results
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return "Negligible"
        elif d < 0.5:
            return "Small"
        elif d < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def _rank_models(self, model_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank models based on multiple criteria"""
        rankings = []
        
        # Prepare scoring matrix
        scores = {}
        
        for model_name, results in model_results.items():
            model_scores = {}
            
            # Quality scores
            if "metrics" in results:
                uc_metrics = results["metrics"].get("use_case_metrics", {})
                tc_metrics = results["metrics"].get("test_case_metrics", {})
                
                model_scores["completeness"] = uc_metrics.get("custom", {}).get("completeness", 0)
                model_scores["validity"] = tc_metrics.get("syntax_validity", {}).get("validity_rate", 0)
            
            # Performance scores (inverse for time/memory - lower is better)
            if "performance" in results:
                perf = results["performance"]
                model_scores["speed"] = 1 / max(perf.get("total_time", 1), 0.1)
                model_scores["memory"] = 1 / max(perf.get("total_memory", 1), 1)
            
            scores[model_name] = model_scores
        
        # Normalize scores
        df = pd.DataFrame(scores).T
        df_normalized = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x)
        
        # Calculate composite score (equal weights for simplicity)
        for model_name in model_results.keys():
            if model_name in df_normalized.index:
                composite_score = df_normalized.loc[model_name].mean()
                
                rankings.append({
                    "model": model_name,
                    "composite_score": composite_score,
                    "scores": df_normalized.loc[model_name].to_dict(),
                    "rank": 0  # Will be set after sorting
                })
        
        # Sort by composite score
        rankings.sort(key=lambda x: x["composite_score"], reverse=True)
        
        # Assign ranks
        for i, item in enumerate(rankings):
            item["rank"] = i + 1
        
        return rankings
    
    def _generate_recommendations(self, model_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Get rankings
        rankings = self._rank_models(model_results)
        
        if rankings:
            best_model = rankings[0]["model"]
            recommendations.append(f"Recommended model: {best_model} (highest composite score)")
            
            # Specific recommendations based on use case
            for ranking in rankings:
                model = ranking["model"]
                scores = ranking["scores"]
                
                strengths = []
                weaknesses = []
                
                for metric, score in scores.items():
                    if score > 0.7:
                        strengths.append(metric)
                    elif score < 0.3:
                        weaknesses.append(metric)
                
                if strengths:
                    recommendations.append(
                        f"{model} excels at: {', '.join(strengths)}"
                    )
                
                if weaknesses:
                    recommendations.append(
                        f"{model} needs improvement in: {', '.join(weaknesses)}"
                    )
        
        # Trade-off recommendations
        trade_offs = self._analyze_trade_offs(model_results)
        best_efficiency = max(trade_offs.items(), key=lambda x: x[1].get("efficiency_score", 0))
        recommendations.append(
            f"Best quality/performance balance: {best_efficiency[0]}"
        )
        
        return recommendations