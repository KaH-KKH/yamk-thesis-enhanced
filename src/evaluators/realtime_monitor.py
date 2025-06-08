# src/evaluators/realtime_monitor.py
"""
Realtime monitoring dashboard for LLM evaluation
"""

import asyncio
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
import psutil
import GPUtil
from loguru import logger
import time
from collections import deque

# Optional: Weights & Biases integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. W&B integration disabled.")


class RealtimeMonitor:
    """Realtime monitoring for LLM evaluation"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.metrics_buffer = deque(maxlen=1000)  # Keep last 1000 data points
        self.start_time = datetime.now()
        
        # Initialize W&B if available
        if WANDB_AVAILABLE:
            self.wandb_run = None
    
    def create_streamlit_dashboard(self):
        """Create Streamlit dashboard for realtime monitoring"""
        st.set_page_config(
            page_title="LLM Evaluation Monitor",
            page_icon="📊",
            layout="wide"
        )
        
        # Header
        st.title("🚀 LLM Evaluation Realtime Monitor")
        st.markdown("---")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("Configuration")
            
            # Model selection
            model_options = ["mistral", "gemma_7b_it_4bit", "llama2", "falcon"]
            selected_models = st.multiselect(
                "Select Models to Monitor",
                model_options,
                default=["mistral"]
            )
            
            # Metric selection
            metric_categories = {
                "Quality": ["completeness", "perplexity", "coherence", "diversity"],
                "Performance": ["generation_time", "memory_usage", "gpu_utilization", "tokens_per_second"],
                "User Experience": ["readability", "clarity", "actionability"]
            }
            
            selected_metrics = []
            for category, metrics in metric_categories.items():
                selected = st.multiselect(f"{category} Metrics", metrics)
                selected_metrics.extend(selected)
            
            # Refresh rate
            refresh_rate = st.slider("Refresh Rate (seconds)", 1, 10, 2)
            
            # Control buttons
            col1, col2 = st.columns(2)
            with col1:
                start_monitoring = st.button("▶️ Start", type="primary")
            with col2:
                stop_monitoring = st.button("⏹️ Stop")
        
        # Main dashboard area
        if start_monitoring or st.session_state.get('monitoring_active', False):
            st.session_state['monitoring_active'] = True
            self._run_monitoring_dashboard(selected_models, selected_metrics, refresh_rate)
        
        if stop_monitoring:
            st.session_state['monitoring_active'] = False
            st.info("Monitoring stopped")
    
    def _run_monitoring_dashboard(self, models: List[str], metrics: List[str], refresh_rate: int):
        """Run the main monitoring dashboard"""
        # Create placeholder for dynamic content
        placeholder = st.empty()
        
        # Metrics summary row
        col1, col2, col3, col4 = st.columns(4)
        metric_placeholders = {
            "total_processed": col1.empty(),
            "success_rate": col2.empty(),
            "avg_time": col3.empty(),
            "system_health": col4.empty()
        }
        
        # Charts area
        charts_container = st.container()
        
        # Data table area
        data_container = st.container()
        
        # Run monitoring loop
        while st.session_state.get('monitoring_active', False):
            # Get current metrics
            current_metrics = self._collect_current_metrics(models)
            
            # Update metric cards
            self._update_metric_cards(metric_placeholders, current_metrics)
            
            # Update charts
            with charts_container:
                self._create_monitoring_charts(current_metrics, metrics)
            
            # Update data table
            with data_container:
                self._create_data_table(current_metrics)
            
            # Sleep before next update
            time.sleep(refresh_rate)
            
            # Force streamlit to rerun
            st.experimental_rerun()
    
    def _collect_current_metrics(self, models: List[str]) -> Dict[str, Any]:
        """Collect current metrics from running evaluations"""
        metrics = {
            "timestamp": datetime.now(),
            "models": {},
            "system": self._get_system_metrics()
        }
        
        # Read latest results for each model
        results_dir = Path("results")
        for model in models:
            model_metrics = self._read_model_metrics(results_dir, model)
            metrics["models"][model] = model_metrics
        
        # Add to buffer
        self.metrics_buffer.append(metrics)
        
        return metrics
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_gb": psutil.virtual_memory().used / (1024**3)
        }
        
        # GPU metrics
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            metrics.update({
                "gpu_percent": gpu.load * 100,
                "gpu_memory_percent": gpu.memoryUtil * 100,
                "gpu_temp": gpu.temperature
            })
        
        return metrics
    
    def _read_model_metrics(self, results_dir: Path, model: str) -> Dict[str, Any]:
        """Read latest metrics for a model"""
        # Find most recent run
        model_runs = list(results_dir.glob(f"run_*/{model}_results.json"))
        
        if not model_runs:
            return {"status": "no_data"}
        
        # Get most recent
        latest_run = max(model_runs, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_run) as f:
                data = json.load(f)
            
            # Extract key metrics
            metrics = {
                "status": "active" if data else "idle",
                "last_updated": datetime.fromtimestamp(latest_run.stat().st_mtime)
            }
            
            # Add evaluation metrics
            if "metrics" in data:
                eval_metrics = data["metrics"]
                
                # Quality metrics
                if "use_case_metrics" in eval_metrics:
                    uc_metrics = eval_metrics["use_case_metrics"]
                    metrics["completeness"] = uc_metrics.get("custom", {}).get("completeness", 0)
                    
                    # Add quality metrics if available
                    if "quality" in uc_metrics:
                        metrics.update({
                            "perplexity": uc_metrics["quality"].get("perplexity", {}).get("mean_perplexity", 0),
                            "coherence": uc_metrics["quality"].get("coherence", {}).get("mean_coherence", 0),
                            "diversity": uc_metrics["quality"].get("diversity", {}).get("distinct_2", 0)
                        })
                
                # Test case metrics
                if "test_case_metrics" in eval_metrics:
                    tc_metrics = eval_metrics["test_case_metrics"]
                    metrics["test_validity"] = tc_metrics.get("syntax_validity", {}).get("validity_rate", 0)
            
            # Performance metrics
            if "performance" in data:
                perf = data["performance"]
                metrics["generation_time"] = perf.get("total_time", 0)
                metrics["memory_usage"] = perf.get("total_memory", 0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error reading metrics for {model}: {e}")
            return {"status": "error", "error": str(e)}
    
    def _update_metric_cards(self, placeholders: Dict[str, any], metrics: Dict[str, Any]):
        """Update metric cards with current values"""
        # Calculate aggregate metrics
        total_processed = sum(
            1 for m in metrics.get("models", {}).values() 
            if m.get("status") == "active"
        )
        
        success_rates = []
        avg_times = []
        
        for model_metrics in metrics.get("models", {}).values():
            if "test_validity" in model_metrics:
                success_rates.append(model_metrics["test_validity"])
            if "generation_time" in model_metrics:
                avg_times.append(model_metrics["generation_time"])
        
        # Update cards
        placeholders["total_processed"].metric(
            "Models Active",
            total_processed,
            delta=None
        )
        
        placeholders["success_rate"].metric(
            "Avg Success Rate",
            f"{np.mean(success_rates) * 100:.1f}%" if success_rates else "N/A",
            delta=None
        )
        
        placeholders["avg_time"].metric(
            "Avg Generation Time",
            f"{np.mean(avg_times):.1f}s" if avg_times else "N/A",
            delta=None
        )
        
        # System health
        system = metrics.get("system", {})
        health_score = 100 - (system.get("cpu_percent", 0) + system.get("memory_percent", 0)) / 2
        placeholders["system_health"].metric(
            "System Health",
            f"{health_score:.0f}%",
            delta=f"{system.get('cpu_percent', 0):.0f}% CPU"
        )
    
    def _create_monitoring_charts(self, metrics: Dict[str, Any], selected_metrics: List[str]):
        """Create monitoring charts"""
        # Prepare data from buffer
        df_data = []
        for m in self.metrics_buffer:
            for model, model_metrics in m.get("models", {}).items():
                row = {
                    "timestamp": m["timestamp"],
                    "model": model,
                    **model_metrics
                }
                df_data.append(row)
        
        if not df_data:
            st.info("No data available yet")
            return
        
        df = pd.DataFrame(df_data)
        
        # Create subplots
        col1, col2 = st.columns(2)
        
        # Performance over time
        with col1:
            st.subheader("Performance Metrics")
            
            if "generation_time" in selected_metrics and "generation_time" in df.columns:
                fig = px.line(
                    df,
                    x="timestamp",
                    y="generation_time",
                    color="model",
                    title="Generation Time Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Quality metrics
        with col2:
            st.subheader("Quality Metrics")
            
            quality_metrics = ["completeness", "coherence", "diversity"]
            available_quality = [m for m in quality_metrics if m in df.columns and m in selected_metrics]
            
            if available_quality:
                # Get latest values for radar chart
                latest_df = df.groupby('model').last().reset_index()
                
                fig = go.Figure()
                
                for model in latest_df['model'].unique():
                    model_data = latest_df[latest_df['model'] == model]
                    values = [model_data[m].iloc[0] for m in available_quality]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=available_quality,
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
                    title="Quality Metrics Comparison"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # System resources
        st.subheader("System Resources")
        
        # Prepare system data
        system_data = []
        for m in self.metrics_buffer:
            system_data.append({
                "timestamp": m["timestamp"],
                **m.get("system", {})
            })
        
        system_df = pd.DataFrame(system_data)
        
        if not system_df.empty:
            # Create subplot for CPU and Memory
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("CPU & Memory Usage", "GPU Usage")
            )
            
            # CPU and Memory
            fig.add_trace(
                go.Scatter(
                    x=system_df["timestamp"],
                    y=system_df["cpu_percent"],
                    name="CPU %",
                    line=dict(color="blue")
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=system_df["timestamp"],
                    y=system_df["memory_percent"],
                    name="Memory %",
                    line=dict(color="red")
                ),
                row=1, col=1
            )
            
            # GPU if available
            if "gpu_percent" in system_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=system_df["timestamp"],
                        y=system_df["gpu_percent"],
                        name="GPU %",
                        line=dict(color="green")
                    ),
                    row=1, col=2
                )
                
                if "gpu_temp" in system_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=system_df["timestamp"],
                            y=system_df["gpu_temp"],
                            name="GPU Temp °C",
                            line=dict(color="orange"),
                            yaxis="y2"
                        ),
                        row=1, col=2
                    )
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    def _create_data_table(self, metrics: Dict[str, Any]):
        """Create data table with current metrics"""
        st.subheader("Current Model Status")
        
        # Prepare table data
        table_data = []
        for model, model_metrics in metrics.get("models", {}).items():
            row = {
                "Model": model,
                "Status": model_metrics.get("status", "unknown"),
                "Completeness": f"{model_metrics.get('completeness', 0):.2%}",
                "Test Validity": f"{model_metrics.get('test_validity', 0):.2%}",
                "Gen Time (s)": f"{model_metrics.get('generation_time', 0):.1f}",
                "Memory (MB)": f"{model_metrics.get('memory_usage', 0):.0f}",
                "Last Updated": model_metrics.get("last_updated", "N/A")
            }
            table_data.append(row)
        
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No model data available")
    
    def start_wandb_logging(self, project_name: str = "yamk-thesis-monitor"):
        """Start Weights & Biases logging"""
        if not WANDB_AVAILABLE:
            logger.warning("W&B not available")
            return
        
        self.wandb_run = wandb.init(
            project=project_name,
            config={
                "monitoring_start": self.start_time.isoformat(),
                "config_path": self.config_path
            }
        )
        
        logger.info(f"W&B logging started: {self.wandb_run.url}")
    
    def log_to_wandb(self, metrics: Dict[str, Any]):
        """Log metrics to W&B with correct data types"""
        if not WANDB_AVAILABLE or not self.wandb_run:
            return
        
        wandb_metrics = {}
        
        # System metrics - käsittele oikea rakenne
        system_data = metrics.get("system", {})
        
        # Logita system-metriikat suoraan numeroarvoina
        for key, value in system_data.items():
            if isinstance(value, (int, float)):
                wandb_metrics[f"system/{key}"] = value
            elif isinstance(value, str):
                # Logita stringit ilman _text suffiksia
                wandb_metrics[f"system/{key}"] = value
            elif isinstance(value, dict):
                # Käsittele nested dictionaries (cpu, memory, gpu)
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        wandb_metrics[f"system/{key}_{subkey}"] = subvalue
                    elif isinstance(subvalue, str):
                        wandb_metrics[f"system/{key}_{subkey}"] = subvalue
            elif isinstance(value, list) and len(value) > 0:
                # GPU data - ota ensimmäinen GPU
                if key == "gpu" and isinstance(value[0], dict):
                    gpu = value[0]
                    for gpu_key, gpu_value in gpu.items():
                        if isinstance(gpu_value, (int, float)):
                            wandb_metrics[f"system/gpu_{gpu_key}"] = gpu_value
        
        # Model metrics - poista _text suffiks
        for model, model_metrics in metrics.get("models", {}).items():
            for key, value in model_metrics.items():
                if isinstance(value, (int, float)):
                    wandb_metrics[f"{model}/{key}"] = value
                elif isinstance(value, str) and key != "error":
                    # Logita stringit suoraan ilman suffiksia
                    wandb_metrics[f"{model}/{key}"] = value
        
        # Logita vain jos on dataa
        if wandb_metrics:
            wandb.log(wandb_metrics)
    
    def create_cli_monitor(self):
        """Create CLI-based monitoring (alternative to Streamlit)"""
        from rich.console import Console
        from rich.table import Table
        from rich.live import Live
        from rich.layout import Layout
        from rich.panel import Panel
        
        console = Console()
        
        def generate_layout():
            """Generate the monitoring layout"""
            layout = Layout()
            
            # Get current metrics
            metrics = self._collect_current_metrics(["mistral", "gemma_7b_it_4bit"])
            
            # Create header
            header = Panel(
                f"[bold blue]LLM Evaluation Monitor[/bold blue]\n"
                f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
                style="blue"
            )
            
            # Create metrics table
            table = Table(title="Model Metrics")
            table.add_column("Model", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Completeness", style="yellow")
            table.add_column("Gen Time", style="magenta")
            table.add_column("Memory", style="red")
            
            for model, model_metrics in metrics.get("models", {}).items():
                table.add_row(
                    model,
                    model_metrics.get("status", "unknown"),
                    f"{model_metrics.get('completeness', 0):.2%}",
                    f"{model_metrics.get('generation_time', 0):.1f}s",
                    f"{model_metrics.get('memory_usage', 0):.0f}MB"
                )
            
            # Create system stats
            system = metrics.get("system", {})
            system_panel = Panel(
                f"CPU: {system.get('cpu_percent', 0):.1f}%\n"
                f"Memory: {system.get('memory_percent', 0):.1f}%\n"
                f"GPU: {system.get('gpu_percent', 0):.1f}%",
                title="System Resources",
                style="green"
            )
            
            # Arrange layout
            layout.split_column(
                header,
                Layout(table),
                system_panel
            )
            
            return layout
        
        # Run live monitoring
        with Live(generate_layout(), refresh_per_second=1) as live:
            while True:
                time.sleep(1)
                live.update(generate_layout())


# CLI entry point
def main():
    """Main entry point for monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Evaluation Monitor")
    parser.add_argument("--mode", choices=["streamlit", "cli"], default="streamlit",
                       help="Monitoring mode")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--project", default="yamk-thesis", help="W&B project name")
    
    args = parser.parse_args()
    
    monitor = RealtimeMonitor()
    
    if args.wandb:
        monitor.start_wandb_logging(args.project)
    
    if args.mode == "streamlit":
        # Note: This should be run with: streamlit run realtime_monitor.py
        monitor.create_streamlit_dashboard()
    else:
        monitor.create_cli_monitor()


if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        # This will only work in Streamlit context
        monitor = RealtimeMonitor()
        monitor.create_streamlit_dashboard()
    except:
        # Otherwise run CLI
        main()