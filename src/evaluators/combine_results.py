#!/usr/bin/env python3
# src/evaluators/combine_results.py
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import argparse
from loguru import logger

class ResultsCombiner:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def combine_evaluation_results(self):
        """Combine results from multiple evaluation runs"""
        all_models_data = {}
        
        # Collect all results
        for run_dir in self.results_dir.glob("run_*"):
            logger.info(f"Processing {run_dir}")
            
            for model_file in run_dir.glob("*_results.json"):
                if "comparison" in model_file.name:
                    continue
                    
                model_name = model_file.stem.replace("_results", "")
                
                try:
                    with open(model_file) as f:
                        data = json.load(f)
                        
                    if model_name not in all_models_data:
                        all_models_data[model_name] = []
                    
                    all_models_data[model_name].append(data)
                except Exception as e:
                    logger.warning(f"Failed to load {model_file}: {e}")
                    continue
        
        if not all_models_data:
            logger.error("No model data found in results directory")
            return None
            
        # Calculate averages
        model_summaries = self._calculate_summaries(all_models_data)
        
        # Create visualizations
        output_dir = self.results_dir / f"combined_analysis_{self.timestamp}"
        output_dir.mkdir(exist_ok=True)
        
        df_raw, df_norm = self._create_dataframes(model_summaries)
        
        # Save data
        df_raw.to_csv(output_dir / "raw_metrics.csv")
        df_norm.to_csv(output_dir / "normalized_metrics.csv")
        
        # Create all visualizations
        self.create_radar_chart(df_norm, output_dir)
        self.create_scatter_plot(df_raw, output_dir)
        self.create_comparison_heatmap(df_norm, output_dir)
        self.create_final_report(df_raw, df_norm, output_dir)
        
        logger.success(f"Combined analysis saved to: {output_dir}")
        return output_dir
    
    def _calculate_summaries(self, all_models_data):
        """Calculate average metrics per model"""
        model_summaries = {}
        
        for model, runs in all_models_data.items():
            summary = {
                "completeness": [],
                "test_validity": [],
                "generation_time": [],
                "memory_usage": [],
                "llm_uc_score": [],
                "llm_tc_score": [],
                "perplexity": [],
                "coherence": [],
                "readability": []
            }
            
            for run in runs:
                # Skip failed runs
                if "error" in run:
                    logger.warning(f"Skipping failed run for {model}: {run['error']}")
                    continue
                    
                # Standard metrics
                if "metrics" in run and "standard" in run["metrics"]:
                    std = run["metrics"]["standard"]
                    
                    # Completeness
                    completeness = std.get("use_case_metrics", {}).get("custom", {}).get("completeness", 0)
                    if completeness:
                        summary["completeness"].append(completeness)
                    
                    # Test validity
                    validity = std.get("test_case_metrics", {}).get("syntax_validity", {}).get("validity_rate", 0)
                    if validity:
                        summary["test_validity"].append(validity)
                
                # Extended metrics
                if "metrics" in run and "extended" in run["metrics"]:
                    ext = run["metrics"]["extended"]
                    
                    # Quality metrics
                    if "quality" in ext:
                        perp = ext["quality"].get("perplexity", {}).get("mean_perplexity")
                        if perp:
                            summary["perplexity"].append(perp)
                        
                        coh = ext["quality"].get("coherence", {}).get("mean_coherence")
                        if coh:
                            summary["coherence"].append(coh)
                    
                    # UX metrics
                    if "user_experience" in ext:
                        read = ext["user_experience"].get("readability", {}).get("flesch_reading_ease", {}).get("score")
                        if read:
                            summary["readability"].append(read)
                
                # Performance
                if "performance" in run:
                    time = run["performance"].get("total_time", 0)
                    memory = run["performance"].get("total_memory", 0)
                    if time:
                        summary["generation_time"].append(time)
                    if memory:
                        summary["memory_usage"].append(memory)
                
                # LLM scores
                if "llm_evaluation" in run:
                    llm = run["llm_evaluation"].get("summary", {})
                    uc_score = llm.get("avg_use_case_score")
                    tc_score = llm.get("avg_test_case_score")
                    if uc_score:
                        summary["llm_uc_score"].append(uc_score)
                    if tc_score:
                        summary["llm_tc_score"].append(tc_score)
            
            # Calculate averages
            model_summaries[model] = {
                metric: np.mean(values) if values else 0
                for metric, values in summary.items()
            }
            
            # Add count of successful runs
            model_summaries[model]['num_runs'] = len([
                r for r in runs if "error" not in r
            ])
        
        return model_summaries
    
    def _create_dataframes(self, model_summaries):
        """Create raw and normalized dataframes"""
        df_raw = pd.DataFrame.from_dict(model_summaries, orient='index')
        
        # Normalize metrics
        df_norm = df_raw.copy()
        
        # Remove non-metric columns before normalization
        non_metric_cols = ['num_runs']
        metric_cols = [col for col in df_norm.columns if col not in non_metric_cols]
        
        for col in metric_cols:
            if col in ['generation_time', 'memory_usage', 'perplexity']:  # Lower is better
                if df_norm[col].max() > df_norm[col].min():
                    df_norm[col] = 1 - (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
                else:
                    df_norm[col] = 1 if df_norm[col].sum() == 0 else 0.5
            else:  # Higher is better
                if df_norm[col].max() > df_norm[col].min():
                    df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
                else:
                    df_norm[col] = 1 if df_norm[col].sum() > 0 else 0
                    
        # Calculate composite score (only from metric columns)
        df_norm['composite_score'] = df_norm[metric_cols].mean(axis=1)
        df_raw['composite_score'] = df_norm['composite_score']
        
        # Add rankings
        df_norm['rank'] = df_norm['composite_score'].rank(ascending=False)
        df_raw['rank'] = df_norm['rank']
        
        return df_raw, df_norm
    
    def create_radar_chart(self, df_norm, output_dir):
        """Create radar chart for all models"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("Plotly not installed, skipping radar chart")
            return
            
        # Select key metrics
        metrics = ['completeness', 'test_validity', 'llm_uc_score', 'coherence', 'readability']
        categories = ['Completeness', 'Test Validity', 'LLM Score', 'Coherence', 'Readability']
        
        # Filter to available metrics
        available_metrics = []
        available_categories = []
        for metric, category in zip(metrics, categories):
            if metric in df_norm.columns:
                available_metrics.append(metric)
                available_categories.append(category)
        
        if not available_metrics:
            logger.warning("No metrics available for radar chart")
            return
            
        fig = go.Figure()
        
        for model in df_norm.index:
            values = []
            for metric in available_metrics:
                if metric in df_norm.columns:
                    # Normalize LLM scores (0-10) to 0-1
                    if 'llm' in metric:
                        val = df_norm.loc[model, metric]
                        # Check if already normalized (0-1) or needs normalization (0-10)
                        if val > 1:
                            values.append(val / 10)
                        else:
                            values.append(val)
                    else:
                        values.append(df_norm.loc[model, metric])
                else:
                    values.append(0)
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=available_categories,
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Model Performance Comparison - Key Metrics",
            width=800,
            height=600
        )
        
        # Save HTML version
        fig.write_html(output_dir / "radar_chart.html")
        
        # Try to save PNG version if kaleido is installed
        try:
            fig.write_image(output_dir / "radar_chart.png")
            logger.info("Radar chart saved (HTML and PNG)")
        except Exception as e:
            logger.warning(f"Could not save PNG version of radar chart: {e}")
            logger.info("Radar chart saved (HTML only)")
    
    def create_scatter_plot(self, df_raw, output_dir):
        """Create quality vs performance scatter plot"""
        plt.figure(figsize=(12, 8))
        
        # Use a colormap for different models
        colors = plt.cm.tab10(np.linspace(0, 1, len(df_raw.index)))
        
        for idx, model in enumerate(df_raw.index):
            # Quality: average of completeness, validity, and LLM scores
            quality_metrics = []
            if df_raw.loc[model, 'completeness'] > 0:
                quality_metrics.append(df_raw.loc[model, 'completeness'])
            if df_raw.loc[model, 'test_validity'] > 0:
                quality_metrics.append(df_raw.loc[model, 'test_validity'])
            if 'llm_uc_score' in df_raw.columns and df_raw.loc[model, 'llm_uc_score'] > 0:
                # Normalize LLM score to 0-1 range
                llm_score = df_raw.loc[model, 'llm_uc_score']
                if llm_score > 1:  # Assume it's on 0-10 scale
                    llm_score = llm_score / 10
                quality_metrics.append(llm_score)
            
            quality = np.mean(quality_metrics) if quality_metrics else 0
            
            # Speed: inverse of generation time
            speed = 3600 / df_raw.loc[model, 'generation_time'] if df_raw.loc[model, 'generation_time'] > 0 else 0
            
            plt.scatter(speed, quality, s=300, alpha=0.6, color=colors[idx])
            plt.annotate(model, (speed, quality), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10)
        
        plt.xlabel('Generation Speed (files/hour)', fontsize=12)
        plt.ylabel('Quality Score (0-1)', fontsize=12)
        plt.title('Quality vs Performance Trade-off', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "scatter_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Scatter plot saved")
    
    def create_comparison_heatmap(self, df_norm, output_dir):
        """Create pairwise comparison heatmap"""
        models = list(df_norm.index)
        n = len(models)
        
        if n < 2:
            logger.warning("Not enough models for comparison heatmap")
            return
            
        # Create win matrix
        wins = np.zeros((n, n))
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i != j:
                    score1 = df_norm.loc[model1, 'composite_score']
                    score2 = df_norm.loc[model2, 'composite_score']
                    wins[i, j] = 1 if score1 > score2 else 0
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(wins, xticklabels=models, yticklabels=models,
                    annot=True, fmt='.0f', cmap='RdYlGn', 
                    cbar_kws={'label': 'Win (1) / Loss (0)'},
                    square=True)
        plt.title('Model Pairwise Comparison Matrix', fontsize=14)
        plt.xlabel('Compared Model', fontsize=12)
        plt.ylabel('Reference Model', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / "comparison_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Heatmap saved")
    
    def create_final_report(self, df_raw, df_norm, output_dir):
        """Create markdown report"""
        report = f"""# Combined LLM Evaluation Results
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

### Model Rankings

| Rank | Model | Composite Score | Runs | Best Features |
|------|-------|-----------------|------|---------------|
"""
        
        # Sort by rank
        df_sorted = df_norm.sort_values('rank')
        
        for model in df_sorted.index:
            rank = int(df_sorted.loc[model, 'rank'])
            score = df_sorted.loc[model, 'composite_score']
            num_runs = int(df_raw.loc[model, 'num_runs']) if 'num_runs' in df_raw.columns else 'N/A'
            
            # Find best features
            best_features = []
            metric_cols = [col for col in df_sorted.columns 
                          if col not in ['composite_score', 'rank', 'num_runs']]
            
            for col in metric_cols:
                if df_sorted.loc[model, col] > 0.8:
                    feature_name = col.replace('_', ' ').title()
                    if 'Llm' in feature_name:
                        feature_name = feature_name.replace('Llm', 'LLM')
                    best_features.append(feature_name)
            
            features_str = ', '.join(best_features[:2]) if best_features else 'None > 0.8'
            report += f"| {rank} | {model} | {score:.3f} | {num_runs} | {features_str} |\n"
        
        report += f"""
## Detailed Metrics

### Raw Metrics Table

| Model | Completeness | Test Validity | Gen Time (s) | Memory (MB) | LLM Score |
|-------|--------------|---------------|--------------|-------------|-----------|
"""
        
        for model in df_sorted.index:
            report += f"| {model} | "
            report += f"{df_raw.loc[model, 'completeness']:.2f} | " if 'completeness' in df_raw.columns else "N/A | "
            report += f"{df_raw.loc[model, 'test_validity']:.2f} | " if 'test_validity' in df_raw.columns else "N/A | "
            report += f"{df_raw.loc[model, 'generation_time']:.1f} | " if 'generation_time' in df_raw.columns else "N/A | "
            report += f"{df_raw.loc[model, 'memory_usage']:.1f} | " if 'memory_usage' in df_raw.columns else "N/A | "
            report += f"{df_raw.loc[model, 'llm_uc_score']:.1f} |\n" if 'llm_uc_score' in df_raw.columns else "N/A |\n"
        
        report += """
## Visualizations

- `radar_chart.html` - Multi-dimensional comparison (interactive)
- `scatter_plot.png` - Quality vs Performance trade-off
- `comparison_heatmap.png` - Pairwise win/loss matrix
- `raw_metrics.csv` - Raw data for further analysis
- `normalized_metrics.csv` - Normalized data (0-1 scale)

## Recommendations

"""
        
        # Best overall
        best_overall = df_sorted.index[0]
        report += f"1. **Best Overall**: {best_overall} - Highest composite score ({df_sorted.loc[best_overall, 'composite_score']:.3f})\n"
        
        # Best for quality (if completeness data exists)
        if 'completeness' in df_raw.columns and df_raw['completeness'].sum() > 0:
            best_quality = df_raw['completeness'].idxmax()
            report += f"2. **Best Quality**: {best_quality} - Highest completeness score ({df_raw.loc[best_quality, 'completeness']:.2f})\n"
        
        # Best for speed (if generation time data exists)
        if 'generation_time' in df_raw.columns and df_raw['generation_time'].sum() > 0:
            best_speed = df_raw['generation_time'].idxmin()
            report += f"3. **Fastest**: {best_speed} - Lowest generation time ({df_raw.loc[best_speed, 'generation_time']:.1f}s)\n"
        
        # Most efficient (good balance)
        if len(df_sorted) > 1:
            # Find models with both good quality and speed
            if 'completeness' in df_norm.columns and 'generation_time' in df_norm.columns:
                efficiency_score = df_norm['completeness'] + df_norm['generation_time']
                most_efficient = efficiency_score.idxmax()
                report += f"4. **Most Efficient**: {most_efficient} - Best quality/speed balance\n"
        
        report += """
## Notes

- Composite score is the average of all normalized metrics
- LLM scores are normalized to 0-1 range (original scale: 0-10)
- Generation time and memory usage are inverted for normalization (lower is better)
- Rankings are based on the composite score
"""
        
        # Save report
        with open(output_dir / "combined_report.md", 'w') as f:
            f.write(report)
        
        logger.info("Report saved")

def main():
    parser = argparse.ArgumentParser(description="Combine LLM evaluation results")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    args = parser.parse_args()
    
    combiner = ResultsCombiner(args.results_dir)
    output_dir = combiner.combine_evaluation_results()
    
    if output_dir:
        print(f"\nCombined analysis complete! Results saved to: {output_dir}")
    else:
        print("\nNo results found to combine.")

if __name__ == "__main__":
    main()