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
                
                with open(model_file) as f:
                    data = json.load(f)
                    
                if model_name not in all_models_data:
                    all_models_data[model_name] = []
                
                all_models_data[model_name].append(data)
        
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
        
        return model_summaries
    
    def _create_dataframes(self, model_summaries):
        """Create raw and normalized dataframes"""
        df_raw = pd.DataFrame.from_dict(model_summaries, orient='index')
        
        # Normalize metrics
        df_norm = df_raw.copy()
        
        for col in df_norm.columns:
            if col in ['generation_time', 'memory_usage', 'perplexity']:  # Lower is better
                if df_norm[col].max() > df_norm[col].min():
                    df_norm[col] = 1 - (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
            else:  # Higher is better
                if df_norm[col].max() > df_norm[col].min():
                    df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
                    
        # Calculate composite score
        df_norm['composite_score'] = df_norm.mean(axis=1)
        df_raw['composite_score'] = df_norm['composite_score']
        
        # Add rankings
        df_norm['rank'] = df_norm['composite_score'].rank(ascending=False)
        df_raw['rank'] = df_norm['rank']
        
        return df_raw, df_norm
    
    def create_radar_chart(self, df_norm, output_dir):
        """Create radar chart for all models"""
        import plotly.graph_objects as go
        
        # Select key metrics
        metrics = ['completeness', 'test_validity', 'llm_uc_score', 'coherence', 'readability']
        categories = ['Completeness', 'Test Validity', 'LLM Score', 'Coherence', 'Readability']
        
        fig = go.Figure()
        
        for model in df_norm.index:
            values = []
            for metric in metrics:
                if metric in df_norm.columns:
                    # Normalize LLM scores (0-10) to 0-1
                    if 'llm' in metric:
                        values.append(df_norm.loc[model, metric] / 10)
                    else:
                        values.append(df_norm.loc[model, metric])
                else:
                    values.append(0)
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
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
        
        fig.write_html(output_dir / "radar_chart.html")
        fig.write_image(output_dir / "radar_chart.png")
        logger.info("Radar chart saved")
    
    def create_scatter_plot(self, df_raw, output_dir):
        """Create quality vs performance scatter plot"""
        plt.figure(figsize=(12, 8))
        
        # Calculate quality and speed scores
        for model in df_raw.index:
            # Quality: average of completeness, validity, and LLM scores
            quality_metrics = []
            if df_raw.loc[model, 'completeness'] > 0:
                quality_metrics.append(df_raw.loc[model, 'completeness'])
            if df_raw.loc[model, 'test_validity'] > 0:
                quality_metrics.append(df_raw.loc[model, 'test_validity'])
            if df_raw.loc[model, 'llm_uc_score'] > 0:
                quality_metrics.append(df_raw.loc[model, 'llm_uc_score'] / 10)
            
            quality = np.mean(quality_metrics) if quality_metrics else 0
            
            # Speed: inverse of generation time
            speed = 3600 / df_raw.loc[model, 'generation_time'] if df_raw.loc[model, 'generation_time'] > 0 else 0
            
            plt.scatter(speed, quality, s=300, alpha=0.6)
            plt.annotate(model, (speed, quality), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10)
        
        plt.xlabel('Generation Speed (files/hour)', fontsize=12)
        plt.ylabel('Quality Score (0-1)', fontsize=12)
        plt.title('Quality vs Performance Trade-off', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "scatter_plot.png", dpi=300)
        plt.close()
        logger.info("Scatter plot saved")
    
    def create_comparison_heatmap(self, df_norm, output_dir):
        """Create pairwise comparison heatmap"""
        models = list(df_norm.index)
        n = len(models)
        
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
                    annot=True, fmt='.0f', cmap='RdYlGn', cbar_kws={'label': 'Win (1) / Loss (0)'})
        plt.title('Model Pairwise Comparison Matrix', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "comparison_heatmap.png", dpi=300)
        plt.close()
        logger.info("Heatmap saved")
    
    def create_final_report(self, df_raw, df_norm, output_dir):
        """Create markdown report"""
        report = f"""# Combined LLM Evaluation Results
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

### Model Rankings

| Rank | Model | Composite Score | Best Features |
|------|-------|-----------------|---------------|
"""
        
        # Sort by rank
        df_sorted = df_norm.sort_values('rank')
        
        for model in df_sorted.index:
            rank = int(df_sorted.loc[model, 'rank'])
            score = df_sorted.loc[model, 'composite_score']
            
            # Find best features
            best_features = []
            for col in ['completeness', 'test_validity', 'coherence', 'readability']:
                if col in df_sorted.columns and df_sorted.loc[model, col] > 0.8:
                    best_features.append(col.replace('_', ' ').title())
            
            report += f"| {rank} | {model} | {score:.3f} | {', '.join(best_features[:2])} |\n"
        
        report += f"""
## Detailed Metrics

### Raw Metrics Table

| Model | Completeness | Test Validity | Gen Time (s) | Memory (MB) | LLM Score |
|-------|--------------|---------------|--------------|-------------|-----------|
"""
        
        for model in df_sorted.index:
            report += f"| {model} | {df_raw.loc[model, 'completeness']:.2f} | "
            report += f"{df_raw.loc[model, 'test_validity']:.2f} | "
            report += f"{df_raw.loc[model, 'generation_time']:.1f} | "
            report += f"{df_raw.loc[model, 'memory_usage']:.1f} | "
            report += f"{df_raw.loc[model, 'llm_uc_score']:.1f} |\n"
        
        report += """
## Visualizations

- `radar_chart.html/png` - Multi-dimensional comparison
- `scatter_plot.png` - Quality vs Performance trade-off
- `comparison_heatmap.png` - Pairwise win/loss matrix

## Recommendations

"""
        
        # Best overall
        best_overall = df_sorted.index[0]
        report += f"1. **Best Overall**: {best_overall} - Highest composite score\n"
        
        # Best for quality
        best_quality = df_raw['completeness'].idxmax()
        report += f"2. **Best Quality**: {best_quality} - Highest completeness score\n"
        
        # Best for speed
        best_speed = df_raw['generation_time'].idxmin()
        report += f"3. **Fastest**: {best_speed} - Lowest generation time\n"
        
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
    print(f"\nCombined analysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()