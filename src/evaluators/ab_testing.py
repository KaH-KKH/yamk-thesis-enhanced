# src/evaluators/ab_testing.py
"""
A/B Testing framework for comparing LLM models
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from scipy import stats
from statsmodels.stats.power import tt_solve_power
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .metrics import QualityMetrics, PerformanceMetrics, UserExperienceMetrics
from ..agents.uc_agent import UCAgent
from ..agents.rf_agent import RFAgent
from ..utils.file_handler import FileHandler


@dataclass
class ABTestResult:
    """Results from an A/B test"""
    model_a: str
    model_b: str
    metric: str
    sample_size: int
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    t_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    power: float
    winner: Optional[str]
    significant: bool
    

class ABTestRunner:
    """Run A/B tests between models"""
    
    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        """
        Initialize A/B test runner
        
        Args:
            alpha: Significance level (default 0.05)
            power: Desired statistical power (default 0.8)
        """
        self.alpha = alpha
        self.power = power
        self.quality_metrics = QualityMetrics()
        self.performance_metrics = PerformanceMetrics()
        self.ux_metrics = UserExperienceMetrics()
        
    async def run_ab_test(
        self,
        model_a: str,
        model_b: str,
        test_data_path: str,
        metrics_to_test: List[str] = None,
        sample_size: int = None,
        output_dir: str = "results/ab_tests"
    ) -> Dict[str, Any]:
        """
        Run comprehensive A/B test between two models
        
        Args:
            model_a: Name of first model
            model_b: Name of second model
            test_data_path: Path to test data
            metrics_to_test: List of metrics to test (default: all)
            sample_size: Number of samples per model (default: calculate optimal)
            output_dir: Directory to save results
        """
        logger.info(f"Starting A/B test: {model_a} vs {model_b}")
        
        # Create output directory
        output_path = Path(output_dir) / f"{model_a}_vs_{model_b}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate optimal sample size if not provided
        if sample_size is None:
            sample_size = self._calculate_sample_size()
            logger.info(f"Calculated optimal sample size: {sample_size}")
        
        # Prepare test data
        test_files = self._prepare_test_data(test_data_path, sample_size)
        
        # Run tests for both models
        results_a = await self._run_model_tests(model_a, test_files)
        results_b = await self._run_model_tests(model_b, test_files)
        
        # Determine which metrics to test
        if metrics_to_test is None:
            metrics_to_test = self._get_default_metrics()
        
        # Perform statistical tests
        test_results = {}
        for metric in metrics_to_test:
            result = self._perform_statistical_test(
                model_a, model_b,
                results_a.get(metric, []),
                results_b.get(metric, []),
                metric
            )
            test_results[metric] = result
        
        # Generate visualizations
        self._generate_visualizations(test_results, output_path)
        
        # Generate report
        report = self._generate_report(test_results, output_path)
        
        # Save results
        self._save_results(test_results, report, output_path)
        
        return {
            "test_results": test_results,
            "report": report,
            "output_dir": str(output_path)
        }
    
    def _calculate_sample_size(self, effect_size: float = 0.5) -> int:
        """Calculate optimal sample size for desired power"""
        # Using t-test power analysis
        sample_size = tt_solve_power(
            effect_size=effect_size,
            alpha=self.alpha,
            power=self.power,
            ratio=1.0,  # Equal sample sizes
            alternative='two-sided'
        )
        
        return max(int(np.ceil(sample_size)), 30)  # Minimum 30 samples
    
    def _prepare_test_data(self, test_data_path: str, sample_size: int) -> List[str]:
        """Prepare test data files"""
        path = Path(test_data_path)
        
        if path.is_file():
            # Single file - replicate or split
            return [str(path)] * min(sample_size, 10)  # Limit replication
        elif path.is_dir():
            # Directory - sample files
            files = list(path.glob("*.txt"))
            if len(files) >= sample_size:
                return [str(f) for f in np.random.choice(files, sample_size, replace=False)]
            else:
                # Replicate files if needed
                return [str(f) for f in np.random.choice(files, sample_size, replace=True)]
        else:
            raise ValueError(f"Invalid test data path: {test_data_path}")
    
    async def _run_model_tests(self, model_name: str, test_files: List[str]) -> Dict[str, List[float]]:
        """Run tests for a single model and collect metrics"""
        logger.info(f"Running tests for model: {model_name}")
        
        metrics_data = {
            # Quality metrics
            "completeness": [],
            "perplexity": [],
            "coherence": [],
            "diversity": [],
            
            # Performance metrics
            "generation_time": [],
            "memory_usage": [],
            "tokens_per_second": [],
            
            # UX metrics
            "readability": [],
            "clarity": [],
            "actionability": [],
            
            # Test case metrics
            "test_validity": [],
            "test_executability": []
        }
        
        # Initialize agents
        uc_agent = UCAgent(model_name)
        rf_agent = RFAgent(model_name)
        
        for test_file in test_files:
            try:
                # Monitor performance
                perf_monitor = self.performance_metrics
                perf_monitor.start_monitoring()
                
                # Generate use case
                start_time = datetime.now()
                use_case = await uc_agent.generate_use_case(test_file)
                generation_time = (datetime.now() - start_time).total_seconds()
                
                # Stop monitoring
                perf_data = perf_monitor.stop_monitoring()
                
                # Get generated text
                use_case_text = uc_agent._format_use_case_text(use_case)
                
                # Calculate metrics
                # Performance
                metrics_data["generation_time"].append(generation_time)
                metrics_data["memory_usage"].append(perf_data.get("memory", {}).get("peak_mb", 0))
                
                # Quality
                quality_results = self.quality_metrics.calculate_all_metrics([use_case_text])
                metrics_data["perplexity"].append(
                    quality_results.get("perplexity", {}).get("mean_perplexity", float('inf'))
                )
                metrics_data["coherence"].append(
                    quality_results.get("coherence", {}).get("mean_coherence", 0)
                )
                metrics_data["diversity"].append(
                    quality_results.get("diversity", {}).get("distinct_2", 0)
                )
                
                # UX metrics
                ux_results = self.ux_metrics.calculate_all_metrics([use_case_text], "use_case")
                metrics_data["readability"].append(
                    100 - ux_results.get("readability", {}).get("flesch_kincaid_grade", {}).get("score", 12)
                )
                metrics_data["clarity"].append(
                    ux_results.get("clarity", {}).get("overall_clarity", 0)
                )
                metrics_data["actionability"].append(
                    ux_results.get("actionability", {}).get("overall_actionability", 0)
                )
                
                # Completeness
                completeness = self._calculate_completeness(use_case_text)
                metrics_data["completeness"].append(completeness)
                
                # Generate test case for additional metrics
                # Save temporary use case file
                temp_uc_file = f"/tmp/uc_{model_name}_{datetime.now().timestamp()}.txt"
                FileHandler.save_text_file(use_case_text, temp_uc_file)
                
                # Generate test case
                test_case = await rf_agent.generate_test_case(temp_uc_file)
                
                # Test case metrics
                validity = self._check_test_validity(test_case)
                metrics_data["test_validity"].append(validity)
                
                # Clean up temp file
                Path(temp_uc_file).unlink(missing_ok=True)
                
            except Exception as e:
                logger.error(f"Error testing {model_name} on {test_file}: {e}")
                # Add null values for failed tests
                for metric in metrics_data:
                    metrics_data[metric].append(np.nan)
        
        return metrics_data
    
    def _calculate_completeness(self, use_case_text: str) -> float:
        """Calculate completeness score for use case"""
        required_sections = ['actors', 'preconditions', 'main flow', 'postconditions']
        found_sections = sum(1 for section in required_sections if section in use_case_text.lower())
        return found_sections / len(required_sections)
    
    def _check_test_validity(self, test_case: str) -> float:
        """Check validity of generated test case"""
        required_patterns = [
            r'\*\*\* Settings \*\*\*',
            r'\*\*\* Test Cases \*\*\*',
            r'Documentation',
            'Click', 'Type Text'  # Basic keywords
        ]
        
        found_patterns = sum(1 for pattern in required_patterns 
                           if (pattern in test_case if isinstance(pattern, str) 
                               else bool(re.search(pattern, test_case))))
        
        return found_patterns / len(required_patterns)
    
    def _get_default_metrics(self) -> List[str]:
        """Get default metrics for A/B testing"""
        return [
            "completeness",
            "readability", 
            "clarity",
            "generation_time",
            "memory_usage",
            "test_validity"
        ]
    
    def _perform_statistical_test(
        self,
        model_a: str,
        model_b: str,
        data_a: List[float],
        data_b: List[float],
        metric: str
    ) -> ABTestResult:
        """Perform statistical test for a single metric"""
        # Remove NaN values
        data_a = [x for x in data_a if not np.isnan(x)]
        data_b = [x for x in data_b if not np.isnan(x)]
        
        if not data_a or not data_b:
            logger.warning(f"Insufficient data for metric {metric}")
            return self._create_null_result(model_a, model_b, metric)
        
        # Calculate statistics
        mean_a = np.mean(data_a)
        mean_b = np.mean(data_b)
        std_a = np.std(data_a, ddof=1)
        std_b = np.std(data_b, ddof=1)
        n_a = len(data_a)
        n_b = len(data_b)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(data_a, data_b, equal_var=False)
        
        # Calculate confidence interval for difference
        se_diff = np.sqrt((std_a**2 / n_a) + (std_b**2 / n_b))
        df = self._calculate_welch_df(std_a, std_b, n_a, n_b)
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        ci_lower = (mean_a - mean_b) - t_critical * se_diff
        ci_upper = (mean_a - mean_b) + t_critical * se_diff
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
        effect_size = abs(mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
        
        # Calculate achieved power
        achieved_power = tt_solve_power(
            effect_size=effect_size,
            nobs1=n_a,
            alpha=self.alpha,
            ratio=n_b/n_a,
            alternative='two-sided'
        )
        
        # Determine winner
        significant = p_value < self.alpha
        if significant:
            # For time/memory metrics, lower is better
            if metric in ["generation_time", "memory_usage", "perplexity"]:
                winner = model_a if mean_a < mean_b else model_b
            else:
                winner = model_a if mean_a > mean_b else model_b
        else:
            winner = None
        
        return ABTestResult(
            model_a=model_a,
            model_b=model_b,
            metric=metric,
            sample_size=min(n_a, n_b),
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            t_statistic=t_stat,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            power=achieved_power,
            winner=winner,
            significant=significant
        )
    
    def _calculate_welch_df(self, std_a: float, std_b: float, n_a: int, n_b: int) -> float:
        """Calculate Welch's degrees of freedom"""
        s1_sq = std_a ** 2
        s2_sq = std_b ** 2
        
        numerator = (s1_sq/n_a + s2_sq/n_b) ** 2
        denominator = (s1_sq/n_a)**2 / (n_a - 1) + (s2_sq/n_b)**2 / (n_b - 1)
        
        return numerator / denominator
    
    def _create_null_result(self, model_a: str, model_b: str, metric: str) -> ABTestResult:
        """Create null result for insufficient data"""
        return ABTestResult(
            model_a=model_a,
            model_b=model_b,
            metric=metric,
            sample_size=0,
            mean_a=np.nan,
            mean_b=np.nan,
            std_a=np.nan,
            std_b=np.nan,
            t_statistic=np.nan,
            p_value=1.0,
            confidence_interval=(np.nan, np.nan),
            effect_size=0,
            power=0,
            winner=None,
            significant=False
        )
    
    def _generate_visualizations(self, test_results: Dict[str, ABTestResult], output_path: Path):
        """Generate visualizations for A/B test results"""
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Metric comparison plot
        self._plot_metric_comparison(test_results, output_path)
        
        # 2. Effect size plot
        self._plot_effect_sizes(test_results, output_path)
        
        # 3. P-value distribution
        self._plot_p_values(test_results, output_path)
        
        # 4. Power analysis
        self._plot_power_analysis(test_results, output_path)
    
    def _plot_metric_comparison(self, test_results: Dict[str, ABTestResult], output_path: Path):
        """Plot metric comparisons"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (metric, result) in enumerate(test_results.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Prepare data
            models = [result.model_a, result.model_b]
            means = [result.mean_a, result.mean_b]
            stds = [result.std_a, result.std_b]
            
            # Create bar plot
            bars = ax.bar(models, means, yerr=stds, capsize=10)
            
            # Color winner
            if result.winner:
                winner_idx = 0 if result.winner == result.model_a else 1
                bars[winner_idx].set_color('green')
                bars[1-winner_idx].set_color('lightcoral')
            
            # Add significance marker
            if result.significant:
                ax.text(0.5, max(means) * 1.1, '***', ha='center', transform=ax.transData)
            
            ax.set_title(f"{metric}\n(p={result.p_value:.3f})")
            ax.set_ylabel("Score")
        
        # Hide unused subplots
        for i in range(len(test_results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / "metric_comparison.png", dpi=300)
        plt.close()
    
    def _plot_effect_sizes(self, test_results: Dict[str, ABTestResult], output_path: Path):
        """Plot effect sizes with confidence intervals"""
        metrics = list(test_results.keys())
        effect_sizes = [result.effect_size for result in test_results.values()]
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(metrics, effect_sizes)
        
        # Color by magnitude
        for bar, effect in zip(bars, effect_sizes):
            if effect > 0.8:
                bar.set_color('darkgreen')
            elif effect > 0.5:
                bar.set_color('green')
            elif effect > 0.2:
                bar.set_color('yellow')
            else:
                bar.set_color('lightgray')
        
        # Add reference lines
        plt.axvline(x=0.2, color='red', linestyle='--', alpha=0.5, label='Small effect')
        plt.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
        plt.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Large effect')
        
        plt.xlabel("Effect Size (Cohen's d)")
        plt.title("Effect Sizes by Metric")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / "effect_sizes.png", dpi=300)
        plt.close()
    
    def _plot_p_values(self, test_results: Dict[str, ABTestResult], output_path: Path):
        """Plot p-value distribution"""
        p_values = [result.p_value for result in test_results.values()]
        metrics = list(test_results.keys())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, p_values)
        
        # Color by significance
        for bar, p in zip(bars, p_values):
            if p < 0.01:
                bar.set_color('darkgreen')
            elif p < 0.05:
                bar.set_color('green')
            else:
                bar.set_color('lightcoral')
        
        # Add significance threshold line
        plt.axhline(y=self.alpha, color='red', linestyle='--', label=f'α = {self.alpha}')
        
        plt.xticks(rotation=45)
        plt.ylabel("p-value")
        plt.title("Statistical Significance by Metric")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / "p_values.png", dpi=300)
        plt.close()
    
    def _plot_power_analysis(self, test_results: Dict[str, ABTestResult], output_path: Path):
        """Plot statistical power achieved"""
        metrics = list(test_results.keys())
        powers = [result.power for result in test_results.values()]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, powers)
        
        # Color by power level
        for bar, power in zip(bars, powers):
            if power >= 0.8:
                bar.set_color('green')
            elif power >= 0.5:
                bar.set_color('yellow')
            else:
                bar.set_color('lightcoral')
        
        # Add desired power line
        plt.axhline(y=self.power, color='blue', linestyle='--', label=f'Desired power = {self.power}')
        
        plt.xticks(rotation=45)
        plt.ylabel("Statistical Power")
        plt.ylim(0, 1)
        plt.title("Achieved Statistical Power by Metric")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / "power_analysis.png", dpi=300)
        plt.close()
    
    def _generate_report(self, test_results: Dict[str, ABTestResult], output_path: Path) -> str:
        """Generate comprehensive A/B test report"""
        lines = [
            "# A/B Test Report",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            ""
        ]
        
        # Count significant results
        significant_count = sum(1 for r in test_results.values() if r.significant)
        total_tests = len(test_results)
        
        lines.extend([
            f"- Total metrics tested: {total_tests}",
            f"- Statistically significant differences: {significant_count}",
            f"- Significance level (α): {self.alpha}",
            f"- Desired power: {self.power}",
            ""
        ])
        
        # Overall winner
        model_wins = {}
        for result in test_results.values():
            if result.winner:
                model_wins[result.winner] = model_wins.get(result.winner, 0) + 1
        
        if model_wins:
            overall_winner = max(model_wins, key=model_wins.get)
            lines.extend([
                "## Overall Winner",
                f"**{overall_winner}** won in {model_wins[overall_winner]} out of {significant_count} significant tests",
                ""
            ])
        
        # Detailed results
        lines.extend([
            "## Detailed Results",
            ""
        ])
        
        for metric, result in test_results.items():
            lines.extend([
                f"### {metric}",
                f"- {result.model_a}: {result.mean_a:.3f} (±{result.std_a:.3f})",
                f"- {result.model_b}: {result.mean_b:.3f} (±{result.std_b:.3f})",
                f"- Difference: {result.mean_a - result.mean_b:.3f}",
                f"- Effect size: {result.effect_size:.3f} ({self._interpret_effect_size(result.effect_size)})",
                f"- p-value: {result.p_value:.4f}",
                f"- Power: {result.power:.3f}",
                f"- **Result:** {'Significant' if result.significant else 'Not significant'}",
            ])
            
            if result.winner:
                lines.append(f"- **Winner:** {result.winner}")
            
            lines.append("")
        
        # Recommendations
        lines.extend([
            "## Recommendations",
            ""
        ])
        
        if significant_count == 0:
            lines.append("- No significant differences found between models")
            lines.append("- Consider increasing sample size or effect size")
        else:
            lines.append(f"- {overall_winner} shows superior performance in key metrics")
            
            # Specific recommendations
            for metric, result in test_results.items():
                if result.significant and result.effect_size > 0.5:
                    lines.append(f"- Strong improvement in {metric} (effect size: {result.effect_size:.2f})")
        
        # Power analysis
        low_power_metrics = [m for m, r in test_results.items() if r.power < self.power]
        if low_power_metrics:
            lines.extend([
                "",
                "## Power Analysis",
                f"The following metrics had low statistical power (<{self.power}):",
                *[f"- {m}" for m in low_power_metrics],
                "Consider increasing sample size for these metrics."
            ])
        
        return "\n".join(lines)
    
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
    
    def _save_results(self, test_results: Dict[str, ABTestResult], report: str, output_path: Path):
        """Save all results"""
        # Save raw results as JSON
        results_dict = {
            metric: {
                "model_a": result.model_a,
                "model_b": result.model_b,
                "mean_a": result.mean_a,
                "mean_b": result.mean_b,
                "std_a": result.std_a,
                "std_b": result.std_b,
                "p_value": result.p_value,
                "effect_size": result.effect_size,
                "power": result.power,
                "winner": result.winner,
                "significant": result.significant
            }
            for metric, result in test_results.items()
        }
        
        FileHandler.save_json(results_dict, str(output_path / "ab_test_results.json"))
        
        # Save report
        FileHandler.save_text_file(report, str(output_path / "ab_test_report.md"))
        
        logger.success(f"A/B test results saved to {output_path}")


# Convenience function for CLI usage
async def run_ab_test_cli():
    """Run A/B test from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run A/B test between models")
    parser.add_argument("--model-a", required=True, help="First model name")
    parser.add_argument("--model-b", required=True, help="Second model name")
    parser.add_argument("--test-data", required=True, help="Path to test data")
    parser.add_argument("--metrics", nargs="+", help="Metrics to test")
    parser.add_argument("--sample-size", type=int, help="Sample size per model")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--power", type=float, default=0.8, help="Desired statistical power")
    parser.add_argument("--output", default="results/ab_tests", help="Output directory")
    
    args = parser.parse_args()
    
    # Run test
    runner = ABTestRunner(alpha=args.alpha, power=args.power)
    results = await runner.run_ab_test(
        model_a=args.model_a,
        model_b=args.model_b,
        test_data_path=args.test_data,
        metrics_to_test=args.metrics,
        sample_size=args.sample_size,
        output_dir=args.output
    )
    
    print(f"\nA/B test complete!")
    print(f"Results saved to: {results['output_dir']}")
    
    # Print summary
    significant_tests = sum(1 for r in results['test_results'].values() if r.significant)
    print(f"\nSignificant differences found: {significant_tests}")
    
    for metric, result in results['test_results'].items():
        if result.significant:
            print(f"- {metric}: {result.winner} wins (p={result.p_value:.4f})")


if __name__ == "__main__":
    asyncio.run(run_ab_test_cli())