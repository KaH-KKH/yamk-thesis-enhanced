# src/evaluators/ab_testing.py
"""
A/B Testing module for statistical model comparison
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
import json
from pathlib import Path
from datetime import datetime


@dataclass
class ABTestResult:
    """Results from an A/B test"""
    model_a: str
    model_b: str
    metric: str
    a_mean: float
    b_mean: float
    a_std: float
    b_std: float
    a_samples: int
    b_samples: int
    t_statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    significant: bool
    winner: Optional[str]
    improvement: float


class ABTestRunner:
    """Run A/B tests between models"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    async def run_ab_test(
        self, 
        model_a: str, 
        model_b: str,
        test_data: List[str],
        metrics_to_test: List[str] = None,
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """Run comprehensive A/B test between two models"""
        
        logger.info(f"Starting A/B test: {model_a} vs {model_b}")
        
        # Import here to avoid circular dependency
        from ..agents.uc_agent import UCAgent
        from ..utils.metrics import MetricsCalculator
        
        results = {
            'model_a': model_a,
            'model_b': model_b,
            'test_size': len(test_data),
            'iterations': num_iterations,
            'timestamp': datetime.now().isoformat(),
            'metric_results': {},
            'overall_winner': None,
            'summary': {}
        }
        
        # Collect samples for each model
        a_samples = []
        b_samples = []
        
        for iteration in range(num_iterations):
            logger.info(f"A/B test iteration {iteration + 1}/{num_iterations}")
            
            # Test model A
            agent_a = UCAgent(model_a)
            a_metrics = await self._test_model(agent_a, test_data, iteration)
            a_samples.append(a_metrics)
            
            # Test model B
            agent_b = UCAgent(model_b)
            b_metrics = await self._test_model(agent_b, test_data, iteration)
            b_samples.append(b_metrics)
        
        # Analyze results for each metric
        all_metrics = set()
        for sample in a_samples + b_samples:
            all_metrics.update(sample.keys())
        
        if metrics_to_test:
            all_metrics = all_metrics.intersection(set(metrics_to_test))
        
        for metric in all_metrics:
            a_values = [s.get(metric, 0) for s in a_samples]
            b_values = [s.get(metric, 0) for s in b_samples]
            
            if a_values and b_values:
                test_result = self._perform_statistical_test(
                    a_values, b_values, model_a, model_b, metric
                )
                results['metric_results'][metric] = test_result
        
        # Determine overall winner
        results['overall_winner'] = self._determine_overall_winner(results['metric_results'])
        results['summary'] = self._generate_summary(results)
        
        return results
    
    async def _test_model(self, agent, test_data: List[str], iteration: int) -> Dict[str, float]:
        """Test a model and collect metrics"""
        metrics = {}
        
        # Generate use cases
        generation_times = []
        quality_scores = []
        
        for i, requirement in enumerate(test_data):
            try:
                start_time = datetime.now()
                
                # Create temp file for requirement
                temp_file = Path(f"temp_req_{iteration}_{i}.txt")
                temp_file.write_text(requirement)
                
                # Generate use case
                use_case = await agent.generate_use_case(str(temp_file))
                
                # Calculate generation time
                gen_time = (datetime.now() - start_time).total_seconds()
                generation_times.append(gen_time)
                
                # Calculate quality score (simplified)
                text = agent._format_use_case_text(use_case)
                quality = self._calculate_simple_quality(text)
                quality_scores.append(quality)
                
                # Clean up
                temp_file.unlink()
                
            except Exception as e:
                logger.error(f"Error testing model: {e}")
                generation_times.append(float('inf'))
                quality_scores.append(0)
        
        # Aggregate metrics
        metrics['avg_generation_time'] = np.mean([t for t in generation_times if t != float('inf')])
        metrics['avg_quality_score'] = np.mean(quality_scores)
        metrics['success_rate'] = sum(1 for t in generation_times if t != float('inf')) / len(test_data)
        
        return metrics
    
    def _calculate_simple_quality(self, text: str) -> float:
        """Simple quality metric for testing"""
        score = 0
        
        # Check for required sections
        required = ['actors', 'preconditions', 'main flow', 'postconditions']
        for section in required:
            if section in text.lower():
                score += 0.25
        
        return score
    
    def _perform_statistical_test(
        self, 
        a_values: List[float], 
        b_values: List[float],
        model_a: str,
        model_b: str,
        metric: str
    ) -> ABTestResult:
        """Perform statistical test between two samples"""
        
        # Calculate basic statistics
        a_mean = np.mean(a_values)
        b_mean = np.mean(b_values)
        a_std = np.std(a_values, ddof=1)
        b_std = np.std(b_values, ddof=1)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(a_values, b_values, equal_var=False)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(a_values) - 1) * a_std**2 + (len(b_values) - 1) * b_std**2) / 
                            (len(a_values) + len(b_values) - 2))
        effect_size = (b_mean - a_mean) / pooled_std if pooled_std > 0 else 0
        
        # Calculate confidence interval for the difference
        se_diff = np.sqrt(a_std**2/len(a_values) + b_std**2/len(b_values))
        t_critical = stats.t.ppf(1 - self.alpha/2, len(a_values) + len(b_values) - 2)
        ci_lower = (b_mean - a_mean) - t_critical * se_diff
        ci_upper = (b_mean - a_mean) + t_critical * se_diff
        
        # Determine significance and winner
        significant = p_value < self.alpha
        
        # For metrics where higher is better
        if significant:
            winner = model_b if b_mean > a_mean else model_a
        else:
            winner = None
        
        # Calculate improvement
        improvement = ((b_mean - a_mean) / a_mean * 100) if a_mean != 0 else 0
        
        return ABTestResult(
            model_a=model_a,
            model_b=model_b,
            metric=metric,
            a_mean=a_mean,
            b_mean=b_mean,
            a_std=a_std,
            b_std=b_std,
            a_samples=len(a_values),
            b_samples=len(b_values),
            t_statistic=t_stat,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            significant=significant,
            winner=winner,
            improvement=improvement
        )
    
    def _determine_overall_winner(self, metric_results: Dict[str, ABTestResult]) -> Optional[str]:
        """Determine overall winner based on all metrics"""
        if not metric_results:
            return None
        
        wins = {}
        
        for result in metric_results.values():
            if result.winner:
                wins[result.winner] = wins.get(result.winner, 0) + 1
        
        if not wins:
            return None
        
        # Return model with most wins
        return max(wins, key=wins.get)
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of A/B test results"""
        summary = {
            'total_metrics_tested': len(results['metric_results']),
            'significant_differences': 0,
            'model_a_wins': 0,
            'model_b_wins': 0,
            'ties': 0,
            'average_effect_size': 0,
            'recommendations': []
        }
        
        effect_sizes = []
        
        for metric, result in results['metric_results'].items():
            if result.significant:
                summary['significant_differences'] += 1
                
                if result.winner == results['model_a']:
                    summary['model_a_wins'] += 1
                else:
                    summary['model_b_wins'] += 1
            else:
                summary['ties'] += 1
            
            effect_sizes.append(abs(result.effect_size))
        
        summary['average_effect_size'] = np.mean(effect_sizes) if effect_sizes else 0
        
        # Generate recommendations
        if summary['model_a_wins'] > summary['model_b_wins']:
            summary['recommendations'].append(
                f"{results['model_a']} performs significantly better on {summary['model_a_wins']} metrics"
            )
        elif summary['model_b_wins'] > summary['model_a_wins']:
            summary['recommendations'].append(
                f"{results['model_b']} performs significantly better on {summary['model_b_wins']} metrics"
            )
        else:
            summary['recommendations'].append(
                "No clear winner - models perform similarly"
            )
        
        # Effect size interpretation
        avg_effect = summary['average_effect_size']
        if avg_effect < 0.2:
            effect_desc = "negligible"
        elif avg_effect < 0.5:
            effect_desc = "small"
        elif avg_effect < 0.8:
            effect_desc = "medium"
        else:
            effect_desc = "large"
        
        summary['recommendations'].append(
            f"Average effect size is {effect_desc} ({avg_effect:.3f})"
        )
        
        return summary
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed A/B test report"""
        lines = [
            f"# A/B Test Report: {results['model_a']} vs {results['model_b']}",
            f"**Date**: {results['timestamp']}",
            f"**Test Size**: {results['test_size']} samples",
            f"**Iterations**: {results['iterations']}",
            f"**Confidence Level**: {self.confidence_level * 100}%",
            "",
            "## Summary",
            f"**Overall Winner**: {results['overall_winner'] or 'No clear winner'}",
            ""
        ]
        
        # Add summary statistics
        summary = results['summary']
        lines.extend([
            f"- Metrics tested: {summary['total_metrics_tested']}",
            f"- Significant differences: {summary['significant_differences']}",
            f"- {results['model_a']} wins: {summary['model_a_wins']}",
            f"- {results['model_b']} wins: {summary['model_b_wins']}",
            f"- Ties: {summary['ties']}",
            f"- Average effect size: {summary['average_effect_size']:.3f}",
            ""
        ])
        
        # Detailed results
        lines.append("## Detailed Results")
        
        for metric, result in results['metric_results'].items():
            lines.extend([
                f"\n### {metric}",
                f"- **{result.model_a}**: {result.a_mean:.4f} (±{result.a_std:.4f})",
                f"- **{result.model_b}**: {result.b_mean:.4f} (±{result.b_std:.4f})",
                f"- **Difference**: {result.b_mean - result.a_mean:.4f} ({result.improvement:+.1f}%)",
                f"- **p-value**: {result.p_value:.4f}",
                f"- **Effect size**: {result.effect_size:.3f}",
                f"- **95% CI**: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]",
                f"- **Significant**: {'Yes' if result.significant else 'No'}",
                f"- **Winner**: {result.winner or 'None'}"
            ])
        
        # Recommendations
        lines.extend([
            "",
            "## Recommendations",
            *[f"- {rec}" for rec in summary['recommendations']]
        ])
        
        return "\n".join(lines)