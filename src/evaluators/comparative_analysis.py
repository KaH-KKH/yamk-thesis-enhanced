# src/evaluators/comparative_analysis.py
"""
Comparative analysis module for model evaluation
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from scipy import stats
from loguru import logger
import pandas as pd
from pathlib import Path


class ComparativeAnalysis:
    """Perform comparative analysis between models"""
    
    def __init__(self):
        self.significance_level = 0.05
    
    def analyze_models(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive comparative analysis"""
        analysis = {
            'statistical_comparison': self._statistical_comparison(all_results),
            'consistency_analysis': self._analyze_consistency(all_results),
            'trade_off_analysis': self._analyze_trade_offs(all_results),
            'scalability_analysis': self._analyze_scalability(all_results),
            'robustness_analysis': self._analyze_robustness(all_results),
            'ranking': self._rank_models(all_results),
            'recommendations': []
        }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _statistical_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Statistical comparison between models"""
        comparison = {}
        
        # Extract metrics for each model
        model_metrics = {}
        for model, data in results.items():
            if 'metrics' in data:
                model_metrics[model] = self._flatten_metrics(data['metrics'])
        
        if len(model_metrics) < 2:
            return {'error': 'Need at least 2 models for comparison'}
        
        # Pairwise comparisons
        models = list(model_metrics.keys())
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model1, model2 = models[i], models[j]
                comparison[f"{model1}_vs_{model2}"] = self._compare_two_models(
                    model_metrics[model1], 
                    model_metrics[model2]
                )
        
        # ANOVA for multiple models
        if len(models) > 2:
            comparison['anova'] = self._perform_anova(model_metrics)
        
        return comparison
    
    def _compare_two_models(self, metrics1: Dict, metrics2: Dict) -> Dict[str, Any]:
        """Compare two models statistically"""
        results = {}
        
        common_metrics = set(metrics1.keys()) & set(metrics2.keys())
        
        for metric in common_metrics:
            val1 = metrics1[metric]
            val2 = metrics2[metric]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Simple difference
                diff = val2 - val1
                pct_change = (diff / val1 * 100) if val1 != 0 else 0
                
                results[metric] = {
                    'model1_value': val1,
                    'model2_value': val2,
                    'difference': diff,
                    'percent_change': pct_change,
                    'better_model': 'model1' if val1 > val2 else 'model2'
                }
        
        # Overall winner (based on majority of metrics)
        better_counts = {'model1': 0, 'model2': 0}
        for metric_result in results.values():
            if 'better_model' in metric_result:
                better_counts[metric_result['better_model']] += 1
        
        results['overall_winner'] = max(better_counts, key=better_counts.get)
        results['win_ratio'] = better_counts[results['overall_winner']] / len(results)
        
        return results
    
    def _perform_anova(self, model_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """Perform ANOVA test across multiple models"""
        anova_results = {}
        
        # Get common metrics
        all_metrics = set()
        for metrics in model_metrics.values():
            all_metrics.update(metrics.keys())
        
        for metric in all_metrics:
            values = []
            for model, metrics in model_metrics.items():
                if metric in metrics and isinstance(metrics[metric], (int, float)):
                    values.append(metrics[metric])
            
            if len(values) >= len(model_metrics):
                # Perform one-way ANOVA
                f_stat, p_value = stats.f_oneway(*[[v] for v in values])
                
                anova_results[metric] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < self.significance_level
                }
        
        return anova_results
    
    def _analyze_consistency(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consistency of model performance"""
        consistency = {}
        
        for model, data in results.items():
            if 'performance' in data:
                perf = data['performance']
                
                # Calculate variance in performance metrics
                time_variance = self._calculate_variance_metric(perf, 'time')
                memory_variance = self._calculate_variance_metric(perf, 'memory')
                
                consistency[model] = {
                    'time_consistency': 1 / (1 + time_variance) if time_variance else 1,
                    'memory_consistency': 1 / (1 + memory_variance) if memory_variance else 1,
                    'overall_consistency': (1 / (1 + time_variance) + 1 / (1 + memory_variance)) / 2
                }
        
        return consistency
    
    def _analyze_trade_offs(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trade-offs between speed and quality"""
        trade_offs = {}
        
        for model, data in results.items():
            if 'metrics' in data and 'performance' in data:
                # Extract quality score (average of all quality metrics)
                quality_metrics = self._flatten_metrics(data['metrics'])
                quality_scores = [v for v in quality_metrics.values() if isinstance(v, (int, float))]
                avg_quality = np.mean(quality_scores) if quality_scores else 0
                
                # Extract speed (inverse of time)
                total_time = data['performance'].get('total_time', 1)
                speed = 1 / total_time if total_time > 0 else 0
                
                # Calculate trade-off score
                trade_offs[model] = {
                    'quality_score': avg_quality,
                    'speed_score': speed,
                    'efficiency_ratio': avg_quality * speed,  # Higher is better
                    'quality_per_second': avg_quality / total_time if total_time > 0 else 0
                }
        
        # Identify Pareto frontier
        self._identify_pareto_frontier(trade_offs)
        
        return trade_offs
    
    def _identify_pareto_frontier(self, trade_offs: Dict[str, Dict]) -> None:
        """Identify models on the Pareto frontier"""
        models = list(trade_offs.keys())
        
        for model in models:
            dominated = False
            for other in models:
                if model != other:
                    # Check if 'other' dominates 'model'
                    if (trade_offs[other]['quality_score'] >= trade_offs[model]['quality_score'] and
                        trade_offs[other]['speed_score'] >= trade_offs[model]['speed_score'] and
                        (trade_offs[other]['quality_score'] > trade_offs[model]['quality_score'] or
                         trade_offs[other]['speed_score'] > trade_offs[model]['speed_score'])):
                        dominated = True
                        break
            
            trade_offs[model]['on_pareto_frontier'] = not dominated
    
    def _analyze_scalability(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how models scale with input size"""
        scalability = {}
        
        for model, data in results.items():
            if 'performance' in data:
                perf = data['performance']
                
                # Extract file processing metrics if available
                if 'use_case_generation' in perf:
                    uc_perf = perf['use_case_generation']
                    if 'total_time' in uc_perf and 'generation_report' in data.get('use_case_generation', {}):
                        total_files = data['use_case_generation']['generation_report'].get('total_files', 1)
                        time_per_file = uc_perf['total_time'] / total_files if total_files > 0 else 0
                        
                        scalability[model] = {
                            'time_per_file': time_per_file,
                            'estimated_time_10x': time_per_file * total_files * 10,
                            'scalability_factor': 1 / (1 + time_per_file)  # Lower time per file is better
                        }
        
        return scalability
    
    def _analyze_robustness(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze robustness and error rates"""
        robustness = {}
        
        for model, data in results.items():
            error_count = 0
            total_operations = 0
            
            # Check use case generation
            if 'use_case_generation' in data:
                report = data['use_case_generation'].get('generation_report', {})
                total_operations += report.get('total_files', 0)
                error_count += report.get('failed', 0)
            
            # Check test case generation
            if 'test_case_generation' in data:
                report = data['test_case_generation'].get('generation_report', {})
                total_operations += report.get('total_files', 0)
                error_count += report.get('failed', 0)
            
            # Calculate robustness metrics
            if total_operations > 0:
                error_rate = error_count / total_operations
                robustness[model] = {
                    'error_rate': error_rate,
                    'success_rate': 1 - error_rate,
                    'total_operations': total_operations,
                    'failed_operations': error_count,
                    'robustness_score': 1 - error_rate
                }
            else:
                robustness[model] = {
                    'error_rate': 0,
                    'success_rate': 1,
                    'robustness_score': 1
                }
        
        return robustness
    
    def _rank_models(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank models based on multiple criteria"""
        scores = {}
        
        # Define weights for different aspects
        weights = {
            'quality': 0.40,
            'performance': 0.20,
            'consistency': 0.15,
            'robustness': 0.15,
            'scalability': 0.10
        }
        
        for model, data in results.items():
            score = 0
            
            # Quality score
            if 'metrics' in data:
                quality_metrics = self._flatten_metrics(data['metrics'])
                quality_values = [v for v in quality_metrics.values() if isinstance(v, (int, float)) and 0 <= v <= 1]
                if quality_values:
                    score += np.mean(quality_values) * weights['quality']
            
            # Performance score (inverse of time, normalized)
            if 'performance' in data:
                total_time = data['performance'].get('total_time', float('inf'))
                all_times = [d['performance'].get('total_time', float('inf')) for d in results.values() if 'performance' in d]
                min_time = min(all_times) if all_times else 1
                performance_score = min_time / total_time if total_time > 0 else 0
                score += performance_score * weights['performance']
            
            # Add other scores similarly...
            
            scores[model] = score
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                'rank': i + 1,
                'model': model,
                'score': score,
                'percentage': score * 100
            }
            for i, (model, score) in enumerate(ranked)
        ]
    
    def _flatten_metrics(self, metrics: Dict, prefix: str = '') -> Dict[str, float]:
        """Flatten nested metrics dictionary"""
        flat = {}
        
        for key, value in metrics.items():
            new_key = f"{prefix}_{key}" if prefix else key
            
            if isinstance(value, dict):
                flat.update(self._flatten_metrics(value, new_key))
            elif isinstance(value, (int, float)):
                flat[new_key] = value
        
        return flat
    
    def _calculate_variance_metric(self, perf: Dict, metric_type: str) -> float:
        """Calculate variance for a specific metric type"""
        values = []
        
        for key, value in perf.items():
            if metric_type in key and isinstance(value, dict):
                for k, v in value.items():
                    if metric_type in k and isinstance(v, (int, float)):
                        values.append(v)
        
        return np.var(values) if values else 0
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Check ranking
        if 'ranking' in analysis and analysis['ranking']:
            top_model = analysis['ranking'][0]['model']
            recommendations.append(f"Overall best model: {top_model} with score {analysis['ranking'][0]['percentage']:.1f}%")
        
        # Check trade-offs
        if 'trade_off_analysis' in analysis:
            pareto_models = [m for m, d in analysis['trade_off_analysis'].items() if d.get('on_pareto_frontier', False)]
            if pareto_models:
                recommendations.append(f"Models on Pareto frontier (best trade-offs): {', '.join(pareto_models)}")
        
        # Check consistency
        if 'consistency_analysis' in analysis:
            most_consistent = max(analysis['consistency_analysis'].items(), key=lambda x: x[1].get('overall_consistency', 0))
            recommendations.append(f"Most consistent model: {most_consistent[0]}")
        
        # Check robustness
        if 'robustness_analysis' in analysis:
            most_robust = max(analysis['robustness_analysis'].items(), key=lambda x: x[1].get('robustness_score', 0))
            recommendations.append(f"Most robust model: {most_robust[0]} with {most_robust[1]['success_rate']*100:.1f}% success rate")
        
        return recommendations