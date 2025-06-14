# src/evaluators/dryrun_analyzer.py
"""
Robot Framework Dryrun Analysis for generated test cases
"""

import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from ..utils.file_handler import FileHandler


class DryrunAnalyzer:
    """Analyze Robot Framework test cases using dryrun"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = FileHandler.load_yaml(config_path)
        self.test_cases_dir = Path(self.config["paths"]["test_cases_dir"])
        self.results = defaultdict(dict)
        
        # Common error patterns
        self.error_patterns = {
            "syntax_error": [
                r"Invalid syntax",
                r"Parsing '.*' failed",
                r"Non-existing setting",
                r"Invalid setting"
            ],
            "missing_keyword": [
                r"No keyword with name '.*' found",
                r"Keyword '.*' not found"
            ],
            "invalid_argument": [
                r"Invalid argument syntax",
                r"Got \\d+ arguments, expected \\d+",
                r"Keyword '.*' expected \\d+ arguments"
            ],
            "missing_library": [
                r"Importing library '.*' failed",
                r"No library '.*' found"
            ],
            "invalid_variable": [
                r"Variable '.*' not found",
                r"Invalid variable name"
            ],
            "structural_error": [
                r"Test case contains no keywords",
                r"Setup failed",
                r"Teardown failed"
            ]
        }
    
    async def analyze_all_models(self) -> Dict[str, Any]:
        """Run dryrun analysis for all models"""
        logger.info("Starting dryrun analysis for all models")
        
        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "by_model": {},
            "error_analysis": {},
            "comparative_analysis": {}
        }
        
        # Get all model directories
        model_dirs = [d for d in self.test_cases_dir.iterdir() if d.is_dir()]
        
        for model_dir in model_dirs:
            model_name = model_dir.name
            logger.info(f"Analyzing model: {model_name}")
            
            model_results = await self.analyze_model(model_dir)
            analysis_results["by_model"][model_name] = model_results
        
        # Generate summary and comparisons
        analysis_results["summary"] = self._generate_summary(analysis_results["by_model"])
        analysis_results["error_analysis"] = self._analyze_errors(analysis_results["by_model"])
        analysis_results["comparative_analysis"] = self._compare_models(analysis_results["by_model"])
        
        return analysis_results
    
    async def analyze_specific_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Run dryrun analysis for specific models only"""
        logger.info(f"Starting dryrun analysis for specific models: {model_names}")
        
        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "by_model": {},
            "error_analysis": {},
            "comparative_analysis": {}
        }
        
        # Analyze only specified models
        for model_name in model_names:
            model_dir = self.test_cases_dir / model_name
            if model_dir.exists() and model_dir.is_dir():
                logger.info(f"Analyzing model: {model_name}")
                model_results = await self.analyze_model(model_dir)
                analysis_results["by_model"][model_name] = model_results
            else:
                logger.warning(f"No test cases found for model: {model_name} at {model_dir}")
                analysis_results["by_model"][model_name] = {
                    "total_files": 0,
                    "successful": 0,
                    "failed": 0,
                    "file_results": {},
                    "errors": ["No test cases directory found"],
                    "warnings": [],
                    "execution_time": 0,
                    "success_rate": 0
                }
        
        # Generate summary and comparisons only for analyzed models
        analysis_results["summary"] = self._generate_summary(analysis_results["by_model"])
        analysis_results["error_analysis"] = self._analyze_errors(analysis_results["by_model"])
        analysis_results["comparative_analysis"] = self._compare_models(analysis_results["by_model"])
        
        return analysis_results
        
    async def analyze_model(self, model_dir: Path) -> Dict[str, Any]:
        """Analyze all test cases for a specific model"""
        robot_files = list(model_dir.glob("*.robot"))
        
        model_results = {
            "total_files": len(robot_files),
            "successful": 0,
            "failed": 0,
            "file_results": {},
            "errors": [],
            "warnings": [],
            "execution_time": 0
        }
        
        for robot_file in robot_files:
            logger.debug(f"Running dryrun for: {robot_file.name}")
            
            start_time = datetime.now()
            result = self._run_dryrun(robot_file)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            model_results["file_results"][robot_file.name] = result
            model_results["execution_time"] += execution_time
            
            if result["success"]:
                model_results["successful"] += 1
            else:
                model_results["failed"] += 1
                model_results["errors"].extend(result.get("errors", []))
            
            model_results["warnings"].extend(result.get("warnings", []))
        
        # Calculate success rate
        model_results["success_rate"] = (
            model_results["successful"] / model_results["total_files"] 
            if model_results["total_files"] > 0 else 0
        )
        
        return model_results
    
    def _run_dryrun(self, robot_file: Path) -> Dict[str, Any]:
        """Run dryrun for a single robot file"""
        result = {
            "file": robot_file.name,
            "success": False,
            "errors": [],
            "warnings": [],
            "output": "",
            "error_types": []
        }
        
        try:
            # Run robot dryrun
            cmd = [
                "robot",
                "--dryrun",
                "--output", "NONE",
                "--report", "NONE",
                "--log", "NONE",
                str(robot_file)
            ]
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            result["output"] = process.stdout + process.stderr
            result["return_code"] = process.returncode
            
            # Success if return code is 0
            result["success"] = process.returncode == 0
            
            # Parse errors and warnings
            if not result["success"]:
                result["errors"] = self._parse_errors(result["output"])
                result["error_types"] = self._categorize_errors(result["errors"])
            
            result["warnings"] = self._parse_warnings(result["output"])
            
        except subprocess.TimeoutExpired:
            result["errors"] = ["Dryrun timeout (30s)"]
            result["error_types"] = ["timeout"]
        except Exception as e:
            result["errors"] = [f"Dryrun failed: {str(e)}"]
            result["error_types"] = ["execution_error"]
        
        return result
    
    def _parse_errors(self, output: str) -> List[str]:
        """Parse error messages from robot output"""
        errors = []
        
        # Look for ERROR lines
        for line in output.split('\n'):
            if '[ ERROR ]' in line or 'Error:' in line:
                error_msg = line.replace('[ ERROR ]', '').strip()
                if error_msg:
                    errors.append(error_msg)
        
        # Also look for specific error patterns
        error_indicators = [
            r"Test case '.*' contains no keywords",
            r"No keyword with name '.*' found",
            r"Parsing '.*' failed",
            r"Invalid syntax in file"
        ]
        
        for pattern in error_indicators:
            matches = re.findall(pattern, output)
            errors.extend(matches)
        
        return list(set(errors))  # Remove duplicates
    
    def _parse_warnings(self, output: str) -> List[str]:
        """Parse warning messages from robot output"""
        warnings = []
        
        for line in output.split('\n'):
            if '[ WARN ]' in line or 'Warning:' in line:
                warning_msg = line.replace('[ WARN ]', '').strip()
                if warning_msg:
                    warnings.append(warning_msg)
        
        return list(set(warnings))
    
    def _categorize_errors(self, errors: List[str]) -> List[str]:
        """Categorize errors by type"""
        categories = []
        
        for error in errors:
            categorized = False
            for category, patterns in self.error_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, error, re.IGNORECASE):
                        categories.append(category)
                        categorized = True
                        break
                if categorized:
                    break
            
            if not categorized:
                categories.append("other")
        
        return categories
    
    def _generate_summary(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary"""
        summary = {
            "total_models": len(model_results),
            "total_test_files": sum(r["total_files"] for r in model_results.values()),
            "total_successful": sum(r["successful"] for r in model_results.values()),
            "total_failed": sum(r["failed"] for r in model_results.values()),
            "overall_success_rate": 0,
            "best_model": None,
            "worst_model": None
        }
        
        # Calculate overall success rate
        if summary["total_test_files"] > 0:
            summary["overall_success_rate"] = (
                summary["total_successful"] / summary["total_test_files"]
            )
        
        # Find best and worst models
        model_scores = {
            model: results["success_rate"] 
            for model, results in model_results.items()
        }
        
        if model_scores:
            summary["best_model"] = max(model_scores, key=model_scores.get)
            summary["worst_model"] = min(model_scores, key=model_scores.get)
        
        return summary
    
    def _analyze_errors(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error patterns across models"""
        error_analysis = {
            "error_type_distribution": defaultdict(int),
            "common_errors": Counter(),
            "model_error_patterns": {}
        }
        
        for model, results in model_results.items():
            model_error_types = Counter()
            
            for file_name, file_result in results["file_results"].items():
                if not file_result["success"]:
                    # Count error types
                    for error_type in file_result.get("error_types", []):
                        error_analysis["error_type_distribution"][error_type] += 1
                        model_error_types[error_type] += 1
                    
                    # Count specific errors
                    for error in file_result.get("errors", []):
                        error_analysis["common_errors"][error] += 1
            
            error_analysis["model_error_patterns"][model] = dict(model_error_types)
        
        # Get top 10 most common errors
        error_analysis["top_errors"] = error_analysis["common_errors"].most_common(10)
        
        return error_analysis
    
    def _compare_models(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative analysis between models"""
        comparison = {
            "success_rates": {},
            "error_prone_models": [],
            "most_stable_models": [],
            "execution_efficiency": {}
        }
        
        # Collect metrics for each model
        for model, results in model_results.items():
            comparison["success_rates"][model] = results["success_rate"]
            comparison["execution_efficiency"][model] = {
                "avg_time_per_file": (
                    results["execution_time"] / results["total_files"]
                    if results["total_files"] > 0 else 0
                ),
                "total_time": results["execution_time"]
            }
        
        # Identify error-prone and stable models
        sorted_models = sorted(
            comparison["success_rates"].items(), 
            key=lambda x: x[1]
        )
        
        # Bottom 40% are error-prone
        error_threshold = len(sorted_models) * 0.4
        comparison["error_prone_models"] = [
            m[0] for m in sorted_models[:int(error_threshold)]
        ]
        
        # Top 40% are most stable
        stable_threshold = len(sorted_models) * 0.6
        comparison["most_stable_models"] = [
            m[0] for m in sorted_models[int(stable_threshold):]
        ]
        
        return comparison
    
    def generate_report(self, analysis_results: Dict[str, Any], output_dir: Path) -> str:
        """Generate comprehensive dryrun analysis report"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        self._create_visualizations(analysis_results, output_dir)
        
        # Generate markdown report
        report_lines = [
            "# Robot Framework Dryrun Analysis Report",
            f"**Generated:** {analysis_results['timestamp']}",
            "",
            "## Executive Summary",
            "",
            f"- **Total Models Analyzed:** {analysis_results['summary']['total_models']}",
            f"- **Total Test Files:** {analysis_results['summary']['total_test_files']}",
            f"- **Successful Tests:** {analysis_results['summary']['total_successful']} "
            f"({analysis_results['summary']['overall_success_rate']:.1%})",
            f"- **Failed Tests:** {analysis_results['summary']['total_failed']}",
            f"- **Best Performing Model:** {analysis_results['summary']['best_model']}",
            f"- **Worst Performing Model:** {analysis_results['summary']['worst_model']}",
            "",
            "## Model Performance Comparison",
            "",
            "| Model | Total Tests | Successful | Failed | Success Rate | Avg Execution Time |",
            "|-------|-------------|------------|--------|--------------|-------------------|"
        ]
        
        # Add model results
        for model, results in analysis_results["by_model"].items():
            avg_time = results["execution_time"] / results["total_files"] if results["total_files"] > 0 else 0
            report_lines.append(
                f"| {model} | {results['total_files']} | "
                f"{results['successful']} | {results['failed']} | "
                f"{results['success_rate']:.1%} | {avg_time:.3f}s |"
            )
        
        # Error analysis section
        report_lines.extend([
            "",
            "## Error Analysis",
            "",
            "### Error Type Distribution",
            "",
            "| Error Type | Occurrences | Percentage |",
            "|------------|-------------|------------|"
        ])
        
        total_errors = sum(analysis_results["error_analysis"]["error_type_distribution"].values())
        for error_type, count in sorted(
            analysis_results["error_analysis"]["error_type_distribution"].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            percentage = (count / total_errors * 100) if total_errors > 0 else 0
            report_lines.append(f"| {error_type} | {count} | {percentage:.1f}% |")
        
        # Most common errors
        report_lines.extend([
            "",
            "### Top 10 Most Common Errors",
            ""
        ])
        
        for i, (error, count) in enumerate(analysis_results["error_analysis"]["top_errors"], 1):
            report_lines.append(f"{i}. **{error}** (occurred {count} times)")
        
        # Model-specific insights
        report_lines.extend([
            "",
            "## Model-Specific Insights",
            ""
        ])
        
        for model, results in analysis_results["by_model"].items():
            if results["failed"] > 0:
                report_lines.extend([
                    f"### {model}",
                    f"- Success Rate: {results['success_rate']:.1%}",
                    f"- Most Common Error Types: {self._get_top_errors_for_model(model, analysis_results)}",
                    ""
                ])
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            "",
            "Based on the dryrun analysis:",
            ""
        ])
        
        if analysis_results["comparative_analysis"]["most_stable_models"]:
            report_lines.append(
                f"1. **Most Stable Models:** {', '.join(analysis_results['comparative_analysis']['most_stable_models'])} "
                "- These models consistently generate syntactically correct Robot Framework tests."
            )
        
        if analysis_results["comparative_analysis"]["error_prone_models"]:
            report_lines.append(
                f"2. **Models Requiring Improvement:** {', '.join(analysis_results['comparative_analysis']['error_prone_models'])} "
                "- These models frequently generate tests with syntax or structural errors."
            )
        
        # Common issues and solutions
        error_dist = analysis_results["error_analysis"]["error_type_distribution"]
        if "syntax_error" in error_dist and error_dist["syntax_error"] > 0:
            report_lines.append(
                "3. **Syntax Errors:** Common issue across models. Consider improving prompt engineering "
                "to emphasize Robot Framework syntax rules."
            )
        
        if "missing_keyword" in error_dist and error_dist["missing_keyword"] > 0:
            report_lines.append(
                "4. **Missing Keywords:** Models are using non-existent keywords. Ensure training/prompts "
                "include valid Browser library keywords."
            )
        
        # Technical details
        report_lines.extend([
            "",
            "## Technical Details",
            "",
            "### Dryrun Command Used",
            "```bash",
            "robot --dryrun --output NONE --report NONE --log NONE <test_file>",
            "```",
            "",
            "### Error Categories",
            "- **syntax_error**: Invalid Robot Framework syntax",
            "- **missing_keyword**: Referenced keyword not found",
            "- **invalid_argument**: Wrong number or format of arguments",
            "- **missing_library**: Required library not imported",
            "- **invalid_variable**: Variable syntax or reference errors",
            "- **structural_error**: Test structure issues (empty tests, setup/teardown problems)",
            ""
        ])
        
        # Save report
        report_content = "\n".join(report_lines)
        report_path = output_dir / "dryrun_analysis_report.md"
        FileHandler.save_text_file(report_content, str(report_path))
        
        # Save raw results as JSON
        json_path = output_dir / "dryrun_results.json"
        FileHandler.save_json(analysis_results, str(json_path))
        
        logger.success(f"Dryrun analysis report saved to: {report_path}")
        
        return report_content
    
    def _create_visualizations(self, analysis_results: Dict[str, Any], output_dir: Path):
        """Create visualization charts"""
        sns.set_style("whitegrid")
        
        # 1. Success rate comparison
        plt.figure(figsize=(10, 6))
        models = list(analysis_results["by_model"].keys())
        success_rates = [
            analysis_results["by_model"][m]["success_rate"] * 100 
            for m in models
        ]
        
        bars = plt.bar(models, success_rates)
        
        # Color bars based on performance
        for bar, rate in zip(bars, success_rates):
            if rate >= 80:
                bar.set_color('green')
            elif rate >= 60:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        
        plt.title('Robot Framework Test Success Rate by Model')
        plt.xlabel('Model')
        plt.ylabel('Success Rate (%)')
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dryrun_success_rates.png', dpi=300)
        plt.close()
        
        # 2. Error type distribution
        plt.figure(figsize=(10, 6))
        error_types = list(analysis_results["error_analysis"]["error_type_distribution"].keys())
        error_counts = list(analysis_results["error_analysis"]["error_type_distribution"].values())
        
        if error_types:
            plt.pie(error_counts, labels=error_types, autopct='%1.1f%%')
            plt.title('Distribution of Error Types')
            plt.tight_layout()
            plt.savefig(output_dir / 'error_type_distribution.png', dpi=300)
        plt.close()
        
        # 3. Model comparison heatmap
        self._create_comparison_heatmap(analysis_results, output_dir)
    
    def _create_comparison_heatmap(self, analysis_results: Dict[str, Any], output_dir: Path):
        """Create a heatmap comparing models across metrics"""
        models = list(analysis_results["by_model"].keys())
        
        # Prepare data matrix
        metrics = ['Success Rate', 'Avg Execution Time', 'Total Errors']
        data = []
        
        for model in models:
            results = analysis_results["by_model"][model]
            row = [
                results["success_rate"],
                results["execution_time"] / results["total_files"] if results["total_files"] > 0 else 0,
                len(results["errors"])
            ]
            data.append(row)
        
        # Normalize data for heatmap
        df = pd.DataFrame(data, index=models, columns=metrics)
        
        # Normalize success rate (already 0-1)
        # Normalize execution time (inverse - lower is better)
        if df['Avg Execution Time'].max() > 0:
            df['Avg Execution Time'] = 1 - (df['Avg Execution Time'] / df['Avg Execution Time'].max())
        
        # Normalize errors (inverse - fewer is better)
        if df['Total Errors'].max() > 0:
            df['Total Errors'] = 1 - (df['Total Errors'] / df['Total Errors'].max())
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5)
        plt.title('Model Performance Comparison (Normalized)')
        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison_heatmap.png', dpi=300)
        plt.close()
    
    def _get_top_errors_for_model(self, model: str, analysis_results: Dict[str, Any]) -> str:
        """Get top error types for a specific model"""
        model_errors = analysis_results["error_analysis"]["model_error_patterns"].get(model, {})
        if model_errors:
            sorted_errors = sorted(model_errors.items(), key=lambda x: x[1], reverse=True)
            return ", ".join([f"{err} ({count})" for err, count in sorted_errors[:3]])
        return "None"