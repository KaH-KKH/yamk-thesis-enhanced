# src/evaluators/results_visualizer.py
"""
Visualize evaluation results for thesis presentation
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import yaml

class ResultsVisualizer:
    """Visualize historical evaluation results"""
    
    def __init__(self, results_dir: str = "results", config_path: str = "configs/config.yaml"):
        self.results_dir = Path(results_dir)
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        """Load configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            st.error(f"Error loading config: {e}")
            return {}
    
    def load_all_results(self) -> Dict[str, List[Dict]]:
        """Load all results from results directory"""
        all_results = {}
        
        # Find all run directories
        run_dirs = sorted([d for d in self.results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
        
        st.sidebar.info(f"Found {len(run_dirs)} evaluation runs")
        
        # Debug: Show first few result file paths
        debug_files = []
        
        for run_dir in run_dirs:
            # Find all result files in this run
            result_files = list(run_dir.glob("*_results.json"))
            
            for result_file in result_files:
                # Add to debug list
                if len(debug_files) < 3:
                    debug_files.append(str(result_file))
                
                # Extract model name from filename
                model_name = result_file.stem.replace("_results", "")
                
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        
                    # Add metadata
                    data['run_id'] = run_dir.name
                    data['timestamp'] = datetime.fromtimestamp(result_file.stat().st_mtime)
                    data['model'] = model_name
                    
                    # Initialize model list if needed
                    if model_name not in all_results:
                        all_results[model_name] = []
                    
                    all_results[model_name].append(data)
                    
                except Exception as e:
                    st.sidebar.warning(f"Error reading {result_file}: {e}")
        
        # Show debug info
        if debug_files:
            with st.sidebar.expander("Debug: Sample file paths"):
                for path in debug_files:
                    st.write(f"- {path}")
        
        return all_results
    
    def extract_metrics(self, results: Dict[str, List[Dict]]) -> pd.DataFrame:
        """Extract metrics into a DataFrame using correct paths from analyze_metrics.py"""
        rows = []
        
        for model, model_results in results.items():
            for result in model_results:
                row = {
                    'model': model,
                    'run_id': result.get('run_id'),
                    'timestamp': result.get('timestamp'),
                }
                
                # Performance metrics (direct path)
                if 'performance' in result:
                    perf = result['performance']
                    row['generation_time'] = perf.get('total_time', 0)
                    # Handle negative memory values
                    memory = perf.get('total_memory', 0)
                    row['memory_usage'] = abs(memory) if memory else 0
                    row['files_per_second'] = perf.get('files_per_second', 0)
                
                # Metrics under 'metrics' key
                if 'metrics' in result:
                    metrics = result['metrics']
                    
                    # Standard metrics path
                    if 'standard' in metrics:
                        standard = metrics['standard']
                        
                        # Use case metrics: metrics.standard.use_case_metrics.custom
                        if 'use_case_metrics' in standard and 'custom' in standard['use_case_metrics']:
                            custom = standard['use_case_metrics']['custom']
                            row['completeness'] = custom.get('completeness', 0)
                            row['avg_length'] = custom.get('avg_length', 0)
                            row['avg_steps'] = custom.get('avg_steps', 0)
                        
                        # Test case metrics: metrics.standard.test_case_metrics
                        if 'test_case_metrics' in standard:
                            tc_std = standard['test_case_metrics']
                            if 'syntax_validity' in tc_std:
                                row['test_validity'] = tc_std['syntax_validity'].get('validity_rate', 0)
                            if 'keyword_coverage' in tc_std:
                                row['keyword_coverage'] = tc_std['keyword_coverage'].get('coverage_rate', 0)
                    
                    # Extended metrics path
                    if 'extended' in metrics:
                        extended = metrics['extended']
                        
                        # Quality metrics: metrics.extended.quality
                        if 'quality' in extended:
                            quality = extended['quality']
                            if 'perplexity' in quality:
                                row['perplexity'] = quality['perplexity'].get('mean_perplexity', 0)
                            if 'diversity' in quality:
                                row['diversity'] = quality['diversity'].get('distinct_2', 0)
                                row['self_bleu'] = quality['diversity'].get('self_bleu', 0)
                            if 'coherence' in quality:
                                row['coherence'] = quality['coherence'].get('mean_coherence', 0)
                        
                        # User experience: metrics.extended.user_experience
                        if 'user_experience' in extended:
                            ux = extended['user_experience']
                            if 'readability' in ux and 'flesch_reading_ease' in ux['readability']:
                                row['readability'] = ux['readability']['flesch_reading_ease'].get('score', 0)
                            if 'clarity' in ux:
                                row['clarity'] = ux['clarity'].get('overall_clarity', 0)
                            if 'actionability' in ux:
                                row['actionability'] = ux['actionability'].get('overall_actionability', 0)
                        
                        # Robot Framework: metrics.extended.robot_framework
                        if 'robot_framework' in extended:
                            rf = extended['robot_framework']
                            if 'overall_quality' in rf:
                                row['rf_total_tests'] = rf['overall_quality'].get('total_tests', 0)
                                row['rf_doc_coverage'] = rf['overall_quality'].get('documentation_coverage', 0)
                            if 'best_practices' in rf:
                                row['rf_wait_strategy'] = rf['best_practices'].get('wait_strategy_score', 0)
                                row['rf_best_selector_ratio'] = rf['best_practices'].get('best_selector_ratio', 0)
                
                # LLM evaluation (direct path)
                if 'llm_evaluation' in result:
                    llm_eval = result['llm_evaluation']
                    if 'summary' in llm_eval:
                        row['llm_use_case_score'] = llm_eval['summary'].get('avg_use_case_score', 0)
                        row['llm_test_case_score'] = llm_eval['summary'].get('avg_test_case_score', 0)
                
                # Generation report data
                if 'use_case_generation' in result:
                    uc_gen = result['use_case_generation']
                    if 'generation_report' in uc_gen:
                        row['uc_total_files'] = uc_gen['generation_report'].get('total_files', 0)
                        row['uc_successful'] = uc_gen['generation_report'].get('successful', 0)
                    if 'performance' in uc_gen:
                        row['uc_generation_time'] = uc_gen['performance'].get('total_time', 0)
                
                if 'test_case_generation' in result:
                    tc_gen = result['test_case_generation']
                    if 'test_validation' in tc_gen:
                        row['tc_executability_rate'] = tc_gen['test_validation'].get('executability_rate', 0)
                    if 'performance' in tc_gen:
                        row['tc_generation_time'] = tc_gen['performance'].get('total_time', 0)
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Clean up any remaining negative values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].apply(lambda x: abs(x) if x < 0 else x)
        
        # Show what metrics were found
        if not df.empty and st.sidebar:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            st.sidebar.success(f"Found {len(numeric_cols)} numeric metrics")
            with st.sidebar.expander("Metrics found"):
                for col in numeric_cols:
                    non_zero = (df[col] != 0).sum()
                    st.write(f"- {col}: {non_zero}/{len(df)} non-zero")
        
        return df
    
    def create_dashboard(self):
        """Create the main dashboard"""
        st.set_page_config(
            page_title="LLM Evaluation Results",
            page_icon="📊",
            layout="wide"
        )
        
        st.title("📊 LLM Evaluation Results Analysis")
        st.markdown("---")
        
        # Load all results
        with st.spinner("Loading results..."):
            all_results = self.load_all_results()
            
            if not all_results:
                st.error("No results found in the results directory!")
                st.info("Please make sure you have run evaluations and results are stored in the 'results' directory.")
                return
            
            df = self.extract_metrics(all_results)
        
        if df.empty:
            st.error("Could not extract any metrics from the results!")
            st.info("Check the sidebar for debug information about the result structure.")
            return
        
        # Show basic info
        st.sidebar.success(f"Loaded {len(df)} result entries from {len(all_results)} models")
        
        # Sidebar filters
        with st.sidebar:
            st.header("Filters")
            
            # Debug option
            show_json_structure = st.checkbox("Show JSON Structure (Debug)")
            
            if show_json_structure and all_results:
                st.subheader("Example JSON Structure")
                # Get first result
                for model, results_list in all_results.items():
                    if results_list:
                        st.json(results_list[0])
                        break
            
            # Model selection
            selected_models = st.multiselect(
                "Select Models",
                options=sorted(df['model'].unique()) if 'model' in df else [],
                default=sorted(df['model'].unique()) if 'model' in df else []
            )
            
            # Run selection
            selected_runs = st.multiselect(
                "Select Runs",
                options=sorted(df['run_id'].unique()) if 'run_id' in df else [],
                default=sorted(df['run_id'].unique()) if 'run_id' in df else []
            )
            
            # Metric category
            metric_category = st.selectbox(
                "Metric Category",
                ["Overview", "Quality Metrics", "Performance", "LLM Evaluation", "Comparison"]
            )
        
        # Filter data
        if selected_models and selected_runs:
            filtered_df = df[
                (df['model'].isin(selected_models)) & 
                (df['run_id'].isin(selected_runs))
            ]
        else:
            filtered_df = df
        
        # Display based on category
        if metric_category == "Overview":
            self._show_overview(filtered_df)
        elif metric_category == "Quality Metrics":
            self._show_quality_metrics(filtered_df)
        elif metric_category == "Performance":
            self._show_performance_metrics(filtered_df)
        elif metric_category == "LLM Evaluation":
            self._show_llm_evaluation(filtered_df)
        elif metric_category == "Comparison":
            self._show_comparison(filtered_df)
        
        # Raw data view
        with st.expander("View Raw Data"):
            st.dataframe(filtered_df)
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    def _show_overview(self, df: pd.DataFrame):
        """Show overview metrics"""
        st.header("📈 Overview")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Models Evaluated",
                len(df['model'].unique()) if 'model' in df else 0,
                delta=None
            )
        
        with col2:
            avg_completeness = df['completeness'].mean() if 'completeness' in df and not df['completeness'].isna().all() else 0
            st.metric(
                "Avg Completeness",
                f"{avg_completeness:.1%}",
                delta=None
            )
        
        with col3:
            avg_validity = df['test_validity'].mean() if 'test_validity' in df and not df['test_validity'].isna().all() else 0
            st.metric(
                "Avg Test Validity",
                f"{avg_validity:.1%}",
                delta=None
            )
        
        with col4:
            avg_time = df['generation_time'].mean() if 'generation_time' in df and not df['generation_time'].isna().all() else 0
            st.metric(
                "Avg Gen Time",
                f"{avg_time:.1f}s",
                delta=None
            )
        
        # Model performance summary
        st.subheader("Model Performance Summary")
        
        # Check which metrics are available
        available_metrics = []
        metric_mapping = {
            'completeness': 'mean',
            'test_validity': 'mean',
            'generation_time': 'mean',
            'memory_usage': 'mean'
        }
        
        agg_dict = {}
        for metric, func in metric_mapping.items():
            if metric in df.columns and not df[metric].isna().all():
                agg_dict[metric] = func
                available_metrics.append(metric)
        
        if not agg_dict:
            st.warning("No metrics available for aggregation")
            return
        
        # Aggregate by model
        model_summary = df.groupby('model').agg(agg_dict).round(3)
        
        # Create bar chart
        fig = go.Figure()
        
        metrics_to_plot = ['completeness', 'test_validity']
        for metric in metrics_to_plot:
            if metric in model_summary.columns:
                fig.add_trace(go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=model_summary.index,
                    y=model_summary[metric],
                    text=[f"{v:.1%}" for v in model_summary[metric]],
                    textposition='auto',
                ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance vs Quality scatter
        if 'generation_time' in df.columns and 'completeness' in df.columns:
            if not df['generation_time'].isna().all() and not df['completeness'].isna().all():
                # Check if memory_usage is available and has valid values
                size_col = None
                if 'memory_usage' in df.columns:
                    # Filter out zero or negative values
                    valid_memory = df['memory_usage'] > 0
                    if valid_memory.any():
                        size_col = 'memory_usage'
                
                fig_scatter = px.scatter(
                    df,
                    x='generation_time',
                    y='completeness',
                    color='model',
                    size=size_col,
                    hover_data=['run_id'],
                    title="Performance vs Quality Trade-off"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
    
    def _show_quality_metrics(self, df: pd.DataFrame):
        """Show quality metrics"""
        st.header("📊 Quality Metrics")
        
        # Available quality metrics - use the actual column names from our extraction
        quality_metrics = ['completeness', 'perplexity', 'coherence', 'diversity', 
                          'self_bleu', 'readability', 'clarity', 'actionability']
        available_metrics = [m for m in quality_metrics if m in df.columns and not df[m].isna().all()]
        
        if not available_metrics:
            st.warning("No quality metrics found in the data")
            st.info("Available columns: " + ", ".join(df.columns))
            return
        
        # Tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Box Plots", "Time Series", "Radar Chart"])
        
        with tab1:
            # Box plots for each metric
            for metric in available_metrics:
                fig = px.box(
                    df,
                    x='model',
                    y=metric,
                    title=f"{metric.replace('_', ' ').title()} Distribution",
                    points="all"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Time series if multiple runs
            if len(df['timestamp'].unique()) > 1:
                for metric in available_metrics:
                    fig = px.line(
                        df.sort_values('timestamp'),
                        x='timestamp',
                        y=metric,
                        color='model',
                        title=f"{metric.replace('_', ' ').title()} Over Time",
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need multiple runs to show time series")
        
        with tab3:
            # Radar chart for latest results
            latest_df = df.sort_values('timestamp').groupby('model').last()
            
            fig = go.Figure()
            
            for model in latest_df.index:
                values = []
                for metric in available_metrics:
                    val = latest_df.loc[model, metric]
                    # Normalize to 0-1 scale for radar chart
                    if metric == 'perplexity':
                        # Lower is better for perplexity
                        val = 1 / (1 + val) if val > 0 else 0
                    elif metric == 'self_bleu':
                        # Lower is better for self-BLEU
                        val = 1 - (val / 100) if val > 0 else 1
                    values.append(val)
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=available_metrics,
                    fill='toself',
                    name=model
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Quality Metrics Comparison (Latest Run)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_performance_metrics(self, df: pd.DataFrame):
        """Show performance metrics"""
        st.header("⚡ Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Generation time comparison
            if 'generation_time' in df.columns:
                fig = px.bar(
                    df.groupby('model')['generation_time'].mean().reset_index(),
                    x='model',
                    y='generation_time',
                    title="Average Generation Time",
                    text='generation_time'
                )
                fig.update_traces(texttemplate='%{text:.1f}s', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Memory usage comparison
            if 'memory_usage' in df.columns:
                avg_memory = df.groupby('model')['memory_usage'].mean().reset_index()
                fig = px.bar(
                    avg_memory,
                    x='model',
                    y='memory_usage',
                    title="Average Memory Usage",
                    text='memory_usage'
                )
                fig.update_traces(texttemplate='%{text:.0f}MB', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
        
        # Files per second
        if 'files_per_second' in df.columns and not df['files_per_second'].isna().all():
            st.subheader("Processing Speed")
            fig = px.box(
                df,
                x='model',
                y='files_per_second',
                title="Files Processed per Second"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Efficiency matrix
        if 'generation_time' in df.columns and 'completeness' in df.columns:
            st.subheader("Efficiency Matrix")
            
            # Calculate efficiency score (quality per time)
            df_efficiency = df.copy()
            df_efficiency['efficiency'] = df_efficiency['completeness'] / (df_efficiency['generation_time'] + 1)
            
            efficiency_matrix = df_efficiency.groupby('model')['efficiency'].mean().reset_index()
            
            fig = px.bar(
                efficiency_matrix,
                x='model',
                y='efficiency',
                title="Efficiency Score (Quality per Time)",
                color='efficiency',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_llm_evaluation(self, df: pd.DataFrame):
        """Show LLM evaluation scores"""
        st.header("🤖 LLM Evaluation Scores")
        
        # Find LLM evaluation columns
        llm_columns = ['llm_use_case_score', 'llm_test_case_score']
        available_llm = [col for col in llm_columns if col in df.columns and not df[col].isna().all()]
        
        if not available_llm:
            st.warning("No LLM evaluation scores found")
            return
        
        # Overall scores comparison
        st.subheader("Overall LLM Evaluation Scores")
        
        # Prepare data for comparison
        comparison_data = []
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            
            for col in available_llm:
                score_type = col.replace('llm_', '').replace('_score', '').replace('_', ' ').title()
                avg_score = model_df[col].mean()
                if avg_score > 0:  # Only include non-zero scores
                    comparison_data.append({
                        'Model': model,
                        'Score Type': score_type,
                        'Average Score': avg_score
                    })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            
            # Bar chart comparison
            fig = px.bar(
                comp_df,
                x='Model',
                y='Average Score',
                color='Score Type',
                title="LLM Evaluation Scores by Model",
                barmode='group'
            )
            fig.update_yaxes(range=[0, 10])
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed scores table
            st.subheader("Detailed Scores")
            pivot_df = comp_df.pivot(index='Model', columns='Score Type', values='Average Score')
            st.dataframe(pivot_df.style.format("{:.2f}").background_gradient(cmap='RdYlGn', vmin=0, vmax=10))
        
        # If we have both use case and test case scores, show correlation
        if len(available_llm) == 2:
            st.subheader("Use Case vs Test Case Score Correlation")
            
            # Create scatter plot
            fig = px.scatter(
                df[df[available_llm[0]] > 0],  # Filter out zero scores
                x=available_llm[0],
                y=available_llm[1],
                color='model',
                title="Use Case Score vs Test Case Score"
            )
            fig.update_xaxes(range=[0, 10], title="Use Case Score")
            fig.update_yaxes(range=[0, 10], title="Test Case Score")
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_comparison(self, df: pd.DataFrame):
        """Show model comparison"""
        st.header("🔍 Model Comparison")
        
        # Select metrics to compare
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['timestamp']
        metric_options = [col for col in numeric_columns if col not in exclude_cols]
        
        if not metric_options:
            st.warning("No numeric metrics available for comparison")
            return
        
        # Default selection - pick the first 5 or all if less than 5
        default_metrics = []
        preferred_metrics = ['completeness', 'test_validity', 'generation_time', 'memory_usage', 'perplexity']
        for metric in preferred_metrics:
            if metric in metric_options:
                default_metrics.append(metric)
        
        # If no preferred metrics found, just take first few
        if not default_metrics:
            default_metrics = metric_options[:min(5, len(metric_options))]
        
        selected_metrics = st.multiselect(
            "Select Metrics to Compare",
            metric_options,
            default=default_metrics
        )
        
        if not selected_metrics:
            st.warning("Please select metrics to compare")
            return
        
        # Check if we have any models to compare
        if 'model' not in df or df['model'].nunique() == 0:
            st.error("No models found in the data")
            return
        
        # Aggregate data
        comparison_df = df.groupby('model')[selected_metrics].mean()
        
        # Normalize data for fair comparison
        normalized_df = comparison_df.copy()
        for col in normalized_df.columns:
            max_val = normalized_df[col].max()
            if max_val > 0:
                # For metrics where lower is better
                if any(x in col.lower() for x in ['perplexity', 'time', 'bleu']):
                    normalized_df[col] = 1 - (normalized_df[col] / max_val)
                else:
                    normalized_df[col] = normalized_df[col] / max_val
        
        # Create parallel coordinates plot
        fig = go.Figure()
        
        for idx, model in enumerate(normalized_df.index):
            fig.add_trace(go.Scatter(
                x=list(range(len(selected_metrics))),
                y=normalized_df.loc[model].values,
                mode='lines+markers',
                name=model,
                line=dict(width=3),
                marker=dict(size=10)
            ))
        
        fig.update_layout(
            title="Normalized Model Comparison",
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(len(selected_metrics))),
                ticktext=[m.replace('_', ' ').title() for m in selected_metrics]
            ),
            yaxis=dict(title="Normalized Score (0-1)"),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Ranking table
        st.subheader("Model Rankings")
        
        # Calculate overall score
        comparison_df['Overall Score'] = normalized_df.mean(axis=1)
        ranking_df = comparison_df.sort_values('Overall Score', ascending=False)
        
        # Style the dataframe
        styled_df = ranking_df.style.background_gradient(cmap='RdYlGn', subset=selected_metrics)
        st.dataframe(styled_df)
        
        # Best model for each metric
        st.subheader("Best Model per Metric")
        best_models = []
        for metric in selected_metrics:
            if any(x in metric.lower() for x in ['perplexity', 'time', 'memory']):
                # Lower is better
                best_model = comparison_df[metric].idxmin()
            else:
                best_model = comparison_df[metric].idxmax()
            best_value = comparison_df.loc[best_model, metric]
            best_models.append({
                'Metric': metric.replace('_', ' ').title(),
                'Best Model': best_model,
                'Value': f"{best_value:.3f}"
            })
        
        st.dataframe(pd.DataFrame(best_models))


def main():
    """Main entry point"""
    visualizer = ResultsVisualizer()
    visualizer.create_dashboard()


if __name__ == "__main__":
    main()