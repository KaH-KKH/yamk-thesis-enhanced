# src/evaluators/metrics/performance_metrics.py
"""
Performance metrics for system monitoring during evaluation
"""

import time
import psutil
import GPUtil
import numpy as np
from typing import Dict, Any, List, Callable
from loguru import logger
import threading
from dataclasses import dataclass
from datetime import datetime

# Optional imports for advanced monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available. Some GPU metrics will be limited.")

try:
    import pyRAPL
    PYRAPL_AVAILABLE = True
except ImportError:
    PYRAPL_AVAILABLE = False
    logger.warning("pyRAPL not available. Energy monitoring will be disabled.")


@dataclass
class PerformanceSnapshot:
    """Single performance measurement"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_temperature: float = 0.0
    cpu_temperature: float = 0.0
    

class PerformanceMetrics:
    """Monitor and calculate performance metrics"""
    
    def __init__(self):
        self.monitoring = False
        self.snapshots: List[PerformanceSnapshot] = []
        self.monitor_thread = None
        self.start_time = None
        self.end_time = None
        
        # Initialize NVIDIA monitoring if available
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.nvml_available = True
            except:
                self.nvml_available = False
        else:
            self.nvml_available = False
            
        # Initialize energy monitoring if available
        self.energy_meter = None
        if PYRAPL_AVAILABLE:
            try:
                pyRAPL.setup()
                self.energy_available = True
            except:
                self.energy_available = False
                logger.warning("pyRAPL setup failed. Energy monitoring disabled.")
        else:
            self.energy_available = False
    
    def start_monitoring(self, interval: float = 0.5):
        """Start performance monitoring in background thread"""
        self.monitoring = True
        self.snapshots = []
        self.start_time = time.time()
        
        # Start energy monitoring
        if self.energy_available:
            self.energy_meter = pyRAPL.Measurement('generation')
            self.energy_meter.begin()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return metrics"""
        self.monitoring = False
        self.end_time = time.time()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        # Get energy consumption
        energy_data = None
        if self.energy_available and self.energy_meter:
            self.energy_meter.end()
            energy_data = self.energy_meter.result
        
        logger.info(f"Performance monitoring stopped. Collected {len(self.snapshots)} snapshots")
        
        # Calculate metrics
        return self._calculate_metrics(energy_data)
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop"""
        while self.monitoring:
            snapshot = self._take_snapshot()
            self.snapshots.append(snapshot)
            time.sleep(interval)
    
    def _take_snapshot(self) -> PerformanceSnapshot:
        """Take a single performance snapshot"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Initialize snapshot
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=memory.used / (1024 * 1024)
        )
        
        # GPU metrics
        if GPUtil.getGPUs():
            gpu = GPUtil.getGPUs()[0]
            snapshot.gpu_percent = gpu.load * 100
            snapshot.gpu_memory_percent = gpu.memoryUtil * 100
            snapshot.gpu_memory_mb = gpu.memoryUsed
            snapshot.gpu_temperature = gpu.temperature
        
        # Advanced GPU metrics with pynvml
        if self.nvml_available:
            try:
                # More detailed GPU metrics
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                snapshot.gpu_percent = util.gpu
                
                # Power draw
                power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Convert to Watts
                snapshot.gpu_power_draw = power
                
                # Clock speeds
                sm_clock = pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_SM)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_MEM)
                snapshot.gpu_sm_clock = sm_clock
                snapshot.gpu_mem_clock = mem_clock
            except Exception as e:
                logger.debug(f"Error getting advanced GPU metrics: {e}")
        
        # CPU temperature
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                snapshot.cpu_temperature = temps['coretemp'][0].current
        except:
            pass
        
        return snapshot
    
    def _calculate_metrics(self, energy_data: Any = None) -> Dict[str, Any]:
        """Calculate aggregate metrics from snapshots"""
        if not self.snapshots:
            return {}
        
        metrics = {}
        
        # Time metrics
        total_time = self.end_time - self.start_time
        metrics["total_time"] = total_time
        
        # CPU metrics
        cpu_values = [s.cpu_percent for s in self.snapshots]
        metrics["cpu"] = {
            "mean_percent": np.mean(cpu_values),
            "max_percent": np.max(cpu_values),
            "min_percent": np.min(cpu_values),
            "std_percent": np.std(cpu_values)
        }
        
        # Memory metrics
        memory_values = [s.memory_mb for s in self.snapshots]
        metrics["memory"] = {
            "mean_mb": np.mean(memory_values),
            "max_mb": np.max(memory_values),
            "min_mb": np.min(memory_values),
            "peak_mb": np.max(memory_values) - np.min(memory_values)
        }
        
        # GPU metrics
        if any(s.gpu_percent > 0 for s in self.snapshots):
            gpu_values = [s.gpu_percent for s in self.snapshots]
            gpu_memory_values = [s.gpu_memory_mb for s in self.snapshots]
            
            metrics["gpu"] = {
                "mean_percent": np.mean(gpu_values),
                "max_percent": np.max(gpu_values),
                "mean_memory_mb": np.mean(gpu_memory_values),
                "max_memory_mb": np.max(gpu_memory_values),
                "peak_memory_mb": np.max(gpu_memory_values) - np.min(gpu_memory_values)
            }
            
            # Temperature
            gpu_temps = [s.gpu_temperature for s in self.snapshots if s.gpu_temperature > 0]
            if gpu_temps:
                metrics["gpu"]["mean_temperature"] = np.mean(gpu_temps)
                metrics["gpu"]["max_temperature"] = np.max(gpu_temps)
            
            # Power metrics if available
            if hasattr(self.snapshots[0], 'gpu_power_draw'):
                power_values = [s.gpu_power_draw for s in self.snapshots if hasattr(s, 'gpu_power_draw')]
                if power_values:
                    metrics["gpu"]["mean_power_watts"] = np.mean(power_values)
                    metrics["gpu"]["max_power_watts"] = np.max(power_values)
                    metrics["gpu"]["total_energy_joules"] = np.sum(power_values) * (total_time / len(power_values))
        
        # Energy metrics
        if energy_data:
            metrics["energy"] = {
                "cpu_energy_joules": energy_data.pkg[0] if hasattr(energy_data, 'pkg') else None,
                "dram_energy_joules": energy_data.dram[0] if hasattr(energy_data, 'dram') else None,
                "total_energy_joules": sum(filter(None, [
                    energy_data.pkg[0] if hasattr(energy_data, 'pkg') else 0,
                    energy_data.dram[0] if hasattr(energy_data, 'dram') else 0
                ]))
            }
        
        # Efficiency metrics
        metrics["efficiency"] = self._calculate_efficiency_metrics(metrics, total_time)
        
        return metrics
    
    def _calculate_efficiency_metrics(self, metrics: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """Calculate efficiency metrics"""
        efficiency = {}
        
        # CPU efficiency
        if "cpu" in metrics:
            efficiency["cpu_efficiency"] = 100 - metrics["cpu"]["mean_percent"]
        
        # Memory efficiency
        if "memory" in metrics:
            total_memory = psutil.virtual_memory().total / (1024 * 1024)
            efficiency["memory_efficiency"] = 100 - (metrics["memory"]["mean_mb"] / total_memory * 100)
        
        # GPU efficiency
        if "gpu" in metrics:
            efficiency["gpu_efficiency"] = 100 - metrics["gpu"]["mean_percent"]
            
            # Performance per watt
            if "mean_power_watts" in metrics["gpu"]:
                # Simplified metric: inverse of power consumption
                efficiency["performance_per_watt"] = 1000 / metrics["gpu"]["mean_power_watts"]
        
        # Time efficiency (if we have reference time)
        # This would need to be set based on baseline measurements
        
        return efficiency
    
    def measure_function_performance(self, func: Callable, *args, **kwargs) -> tuple:
        """Measure performance of a specific function"""
        # Start monitoring
        self.start_monitoring(interval=0.1)
        
        # Measure execution time precisely
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Stop monitoring
        metrics = self.stop_monitoring()
        
        # Add precise timing
        metrics["execution_time"] = end_time - start_time
        
        return result, metrics
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get static system information"""
        info = {
            "cpu": {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None,
                "cpu_model": self._get_cpu_model()
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3)
            },
            "gpu": []
        }
        
        # GPU info
        for gpu in GPUtil.getGPUs():
            gpu_info = {
                "name": gpu.name,
                "memory_total_mb": gpu.memoryTotal,
                "driver_version": gpu.driver
            }
            
            # Additional info from pynvml
            if self.nvml_available:
                try:
                    gpu_info["compute_capability"] = pynvml.nvmlDeviceGetCudaComputeCapability(self.gpu_handle)
                    gpu_info["pcie_link_width"] = pynvml.nvmlDeviceGetCurrPcieLinkWidth(self.gpu_handle)
                    gpu_info["pcie_link_gen"] = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(self.gpu_handle)
                except:
                    pass
            
            info["gpu"].append(gpu_info)
        
        return info
    
    def _get_cpu_model(self) -> str:
        """Get CPU model name"""
        try:
            import platform
            return platform.processor()
        except:
            return "Unknown"