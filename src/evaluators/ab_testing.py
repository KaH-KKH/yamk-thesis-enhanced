# src/evaluators/ab_testing.py
"""
A/B testing implementation for comparing two LLM models
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu
from statsmodels.stats.power import tt_ind_solve_power  # KORJAUS: tt_ind_solve_power kahden otoksen vertailuun
from pathlib import Path
import json
from datetime import datetime
from loguru import logger

from ..agents.uc_agent import UCAgent
from ..agents.rf_agent import RFAgent
from ..utils.file_handler import FileHandler


class ABTestRunner:
    """Run A/B tests between two models"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = FileHandler.load_yaml(config_path)
        self.alpha = 0.05  # significance level
        self.power = 0.8   # desired power
        self.effect_size = 0.5  # medium effect size
        
    async def run_ab_test(self, model_a: str, model_b: str, 
                         test_data_path: str, output_dir: str) -> Dict[str, Any]:
        """Run A/B test between two models"""
        
        logger.info(f"Starting A/B test: {model_a} vs {model_b}")
        
        # Calculate required sample size
        sample_size = self._calculate_sample_size()
        logger.info(f"Required sample size per group: {sample_size}")
        
        # Prepare test data
        test_files = self._prepare_test_data(test_data_path, sample_size)
        
        # Run evaluation for both models
        results_a = await self._evaluate_model(model_a, test_files)
        results_b = await self._evaluate_model(model_b, test_files)
        
        # Perform statistical tests
        statistical_results = self._perform_statistical_tests(results_a, results_b)
        
        # Calculate effect sizes
        effect_sizes = self._calculate_effect_sizes(results_a, results_b)
        
        # Generate report
        report = self._generate_ab_report(
            model_a, model_b, results_a, results_b, 
            statistical_results, effect_sizes, sample_size
        )
        
        # Save results
        self._save_ab_results(output_dir, report)
        
        return {
            "model_a": model_a,
            "model_b": model_b,
            "sample_size": sample_size,
            "statistical_results": statistical_results,
            "effect_sizes": effect_sizes,
            "report": report
        }
    
    def _calculate_sample_size(self) -> int:
        """Calculate required sample size for A/B test"""
        try:
            # KORJAUS: Yksinkertainen approach ilman ratio parametria
            from statsmodels.stats.power import TTestPower
            
            power_analysis = TTestPower()
            sample_size = power_analysis.solve_power(
                effect_size=self.effect_size,
                power=self.power,
                alpha=self.alpha,
                nobs=None,  # This is what we solve for
                alternative='two-sided'
            )
            
            return max(int(np.ceil(sample_size)), 5)  # Minimum 5 samples
            
        except Exception as e:
            logger.warning(f"Error calculating sample size: {e}. Using default.")
            return 10  # fallback
    
    def _prepare_test_data(self, data_path: str, sample_size: int) -> List[str]:
        """Prepare test data files"""
        data_dir = Path(data_path)
        all_files = list(data_dir.glob("*.txt"))
        
        if len(all_files) < sample_size:
            logger.warning(f"Only {len(all_files)} files available, less than required {sample_size}")
            return [str(f) for f in all_files]
        
        # Randomly sample files
        import random
        selected_files = random.sample(all_files, sample_size)
        return [str(f) for f in selected_files]
    
    async def _evaluate_model(self, model_name: str, test_files: List[str]) -> Dict[str, List[float]]:
        """Evaluate a model on test files"""
        logger.info(f"Evaluating model: {model_name}")
        
        metrics = {
            "generation_time": [],
            "memory_usage": [],
            "use_case_quality": [],
            "test_case_quality": [],
            "success_rate": []
        }
        
        for test_file in test_files:
            try:
                # Evaluate use case generation
                uc_start = datetime.now()
                uc_agent = UCAgent(model_name)
                
                # Create temporary output dir
                temp_output = f"temp_ab_test/{model_name}"
                Path(temp_output).mkdir(parents=True, exist_ok=True)
                
                # Generate use case
                use_case = await uc_agent.generate_use_case(test_file)
                uc_time = (datetime.now() - uc_start).total_seconds()
                
                # Evaluate test case generation
                tc_start = datetime.now()
                rf_agent = RFAgent(model_name)
                
                # Save use case temporarily and generate test case
                uc_file = f"{temp_output}/temp_use_case.json"
                FileHandler.save_json(use_case.model_dump(), uc_file)
                
                robot_content = await rf_agent.generate_test_case(uc_file)
                tc_time = (datetime.now() - tc_start).total_seconds()
                
                # Calculate metrics
                total_time = uc_time + tc_time
                metrics["generation_time"].append(total_time)
                
                # Simple quality metrics
                uc_quality = self._evaluate_use_case_quality(use_case.model_dump())
                tc_quality = self._evaluate_test_case_quality(robot_content)
                
                metrics["use_case_quality"].append(uc_quality)
                metrics["test_case_quality"].append(tc_quality)
                metrics["success_rate"].append(1.0)  # Success
                
                # Memory usage (simplified)
                import psutil
                memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                metrics["memory_usage"].append(memory)
                
            except Exception as e:
                logger.error(f"Error evaluating {test_file} with {model_name}: {e}")
                # Add failure metrics
                metrics["generation_time"].append(float('inf'))
                metrics["memory_usage"].append(0)
                metrics["use_case_quality"].append(0)
                metrics["test_case_quality"].append(0)
                metrics["success_rate"].append(0.0)
        
        return metrics
    
    def _evaluate_use_case_quality(self, use_case: Dict[str, Any]) -> float:
        """Simple use case quality score"""
        score = 0.0
        
        # Check completeness
        required_fields = ['title', 'actors', 'preconditions', 'main_flow', 'postconditions']
        for field in required_fields:
            if field in use_case and use_case[field]:
                score += 0.2
        
        return score
    
    def _evaluate_test_case_quality(self, robot_content: str) -> float:
        """Simple test case quality score"""
        score = 0.0
        
        # Check for required sections
        if "*** Settings ***" in robot_content:
            score += 0.2
        if "*** Test Cases ***" in robot_content:
            score += 0.2
        if "*** Keywords ***" in robot_content:
            score += 0.2
        if "Browser" in robot_content:
            score += 0.2
        if "Click" in robot_content or "Type Text" in robot_content:
            score += 0.2
        
        return score
    
    def _perform_statistical_tests(self, results_a: Dict[str, List[float]], 
                                 results_b: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform statistical tests between groups"""
        statistical_results = {}
        
        for metric in results_a.keys():
            try:
                data_a = np.array(results_a[metric])
                data_b = np.array(results_b[metric])
                
                # Remove infinite values
                data_a = data_a[np.isfinite(data_a)]
                data_b = data_b[np.isfinite(data_b)]
                
                if len(data_a) == 0 or len(data_b) == 0:
                    continue
                
                # Shapiro-Wilk test for normality
                _, p_norm_a = stats.shapiro(data_a) if len(data_a) > 3 else (None, 0.05)
                _, p_norm_b = stats.shapiro(data_b) if len(data_b) > 3 else (None, 0.05)
                
                # Choose appropriate test
                if p_norm_a > 0.05 and p_norm_b > 0.05:
                    # Normal distribution - use t-test
                    t_stat, p_value = ttest_ind(data_a, data_b)
                    test_used = "t-test"
                else:
                    # Non-normal distribution - use Mann-Whitney U test
                    u_stat, p_value = mannwhitneyu(data_a, data_b, alternative='two-sided')
                    test_used = "mann-whitney"
                    t_stat = u_stat
                
                statistical_results[metric] = {
                    "test_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < self.alpha,
                    "test_used": test_used,
                    "mean_a": float(np.mean(data_a)),
                    "mean_b": float(np.mean(data_b)),
                    "std_a": float(np.std(data_a)),
                    "std_b": float(np.std(data_b)),
                    "n_a": len(data_a),
                    "n_b": len(data_b)
                }
                
            except Exception as e:
                logger.error(f"Error in statistical test for {metric}: {e}")
                statistical_results[metric] = {
                    "error": str(e),
                    "significant": False
                }
        
        return statistical_results
    
    def _calculate_effect_sizes(self, results_a: Dict[str, List[float]], 
                              results_b: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate effect sizes (Cohen's d)"""
        effect_sizes = {}
        
        for metric in results_a.keys():
            try:
                data_a = np.array(results_a[metric])
                data_b = np.array(results_b[metric])
                
                # Remove infinite values
                data_a = data_a[np.isfinite(data_a)]
                data_b = data_b[np.isfinite(data_b)]
                
                if len(data_a) == 0 or len(data_b) == 0:
                    continue
                
                # Cohen's d
                pooled_std = np.sqrt(((len(data_a) - 1) * np.var(data_a) + 
                                    (len(data_b) - 1) * np.var(data_b)) / 
                                   (len(data_a) + len(data_b) - 2))
                
                if pooled_std > 0:
                    cohens_d = (np.mean(data_a) - np.mean(data_b)) / pooled_std
                    effect_sizes[metric] = float(cohens_d)
                
            except Exception as e:
                logger.error(f"Error calculating effect size for {metric}: {e}")
        
        return effect_sizes
    
    def _generate_ab_report(self, model_a: str, model_b: str,
                           results_a: Dict[str, List[float]], 
                           results_b: Dict[str, List[float]],
                           statistical_results: Dict[str, Any],
                           effect_sizes: Dict[str, float],
                           sample_size: int) -> str:
        """Generate A/B test report"""
        
        report_lines = [
            f"# A/B Test Report: {model_a} vs {model_b}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Sample Size:** {sample_size} per group",
            f"**Significance Level:** {self.alpha}",
            "",
            "## Summary",
            ""
        ]
        
        # Winner determination
        significant_wins_a = 0
        significant_wins_b = 0
        
        for metric, stats_result in statistical_results.items():
            if stats_result.get("significant", False):
                if stats_result["mean_a"] > stats_result["mean_b"]:
                    if metric != "generation_time":  # Lower is better for time
                        significant_wins_a += 1
                    else:
                        significant_wins_b += 1
                else:
                    if metric != "generation_time":
                        significant_wins_b += 1
                    else:
                        significant_wins_a += 1
        
        if significant_wins_a > significant_wins_b:
            report_lines.append(f"**Winner: {model_a}** ({significant_wins_a} significant advantages)")
        elif significant_wins_b > significant_wins_a:
            report_lines.append(f"**Winner: {model_b}** ({significant_wins_b} significant advantages)")
        else:
            report_lines.append("**Result: No clear winner** (equivalent performance)")
        
        report_lines.extend([
            "",
            "## Detailed Results",
            "",
            "| Metric | Model A | Model B | P-value | Significant | Effect Size | Winner |",
            "|--------|---------|---------|---------|-------------|-------------|--------|"
        ])
        
        for metric, stats_result in statistical_results.items():
            if "error" in stats_result:
                continue
                
            mean_a = stats_result["mean_a"]
            mean_b = stats_result["mean_b"]
            p_value = stats_result["p_value"]
            significant = "Yes" if stats_result["significant"] else "No"
            effect_size = effect_sizes.get(metric, 0)
            
            # Determine winner for this metric
            if stats_result["significant"]:
                if metric == "generation_time":  # Lower is better
                    winner = model_a if mean_a < mean_b else model_b
                else:  # Higher is better
                    winner = model_a if mean_a > mean_b else model_b
            else:
                winner = "Tie"
            
            report_lines.append(
                f"| {metric} | {mean_a:.3f} | {mean_b:.3f} | {p_value:.4f} | "
                f"{significant} | {effect_size:.3f} | {winner} |"
            )
        
        report_lines.extend([
            "",
            "## Effect Size Interpretation",
            "- Small effect: |d| ≈ 0.2",
            "- Medium effect: |d| ≈ 0.5", 
            "- Large effect: |d| ≈ 0.8",
            "",
            "## Recommendations",
            ""
        ])
        
        # Add recommendations
        if significant_wins_a > significant_wins_b:
            report_lines.append(f"- **Use {model_a}** for production based on superior performance")
        elif significant_wins_b > significant_wins_a:
            report_lines.append(f"- **Use {model_b}** for production based on superior performance")
        else:
            report_lines.append("- Both models perform similarly; choose based on other factors (cost, speed, etc.)")
        
        report_lines.append("- Consider running additional tests with larger sample size for inconclusive metrics")
        
        return "\n".join(report_lines)
    
    def _save_ab_results(self, output_dir: str, report: str):
        """Save A/B test results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save report
        report_file = output_path / "ab_test_report.md"
        FileHandler.save_text_file(report, str(report_file))
        
        logger.info(f"A/B test report saved to: {report_file}")


# Example usage
async def main():
    """Example A/B test"""
    ab_runner = ABTestRunner()
    
    results = await ab_runner.run_ab_test(
        model_a="mistral",
        model_b="gemma_7b_it_4bit",
        test_data_path="data/requirements",
        output_dir="results/ab_test"
    )
    
    print("A/B Test completed!")
    print(f"Results: {results}")


if __name__ == "__main__":
    asyncio.run(main())