import time
import logging
from comfy.cli_args import args

_initialized = False

# Metric objects (initialized only if prometheus is enabled and imported successfully)
queue_length_gauge = None
queue_wait_histogram = None
job_duration_histogram = None
jobs_counter = None
vram_gauge = None
loaded_models_gauge = None
model_swaps_counter = None
node_execution_histogram = None
cache_requests_counter = None

def init_metrics():
    global _initialized
    global queue_length_gauge, queue_wait_histogram, job_duration_histogram, jobs_counter
    global vram_gauge, loaded_models_gauge, model_swaps_counter, node_execution_histogram, cache_requests_counter
    
    if _initialized:
        return True
    
    if not getattr(args, "enable_prometheus", False):
        return False
        
    try:
        import prometheus_client
    except ImportError:
        logging.error("Failed to import prometheus_client. Please install it using `pip install prometheus-client` to enable Prometheus metrics.")
        return False
        
    try:
        # Define and register metrics
        queue_length_gauge = prometheus_client.Gauge(
            "comfyui_queue_length",
            "Current number of pending workflows in the prompt queue"
        )
        queue_wait_histogram = prometheus_client.Histogram(
            "comfyui_queue_wait_seconds",
            "Latency between workflow submission and the start of processing"
        )
        job_duration_histogram = prometheus_client.Histogram(
            "comfyui_job_duration_seconds",
            "Execution time for completed workflows"
        )
        jobs_counter = prometheus_client.Counter(
            "comfyui_jobs_total",
            "Tracks completed, failed, and interrupted execution counts",
            ["status"]
        )
        vram_gauge = prometheus_client.Gauge(
            "comfyui_vram_bytes",
            "VRAM usage (allocated or reserved) by PyTorch allocator",
            ["device", "type"]
        )
        loaded_models_gauge = prometheus_client.Gauge(
            "comfyui_loaded_models_count",
            "Count of models currently held in memory"
        )
        model_swaps_counter = prometheus_client.Counter(
            "comfyui_model_swaps_total",
            "Number of model load/unload transfers between RAM and VRAM"
        )
        node_execution_histogram = prometheus_client.Histogram(
            "comfyui_node_execution_seconds",
            "Execution time for individual node executions",
            ["node_type"]
        )
        cache_requests_counter = prometheus_client.Counter(
            "comfyui_cache_requests_total",
            "Tracks hit or miss for node execution cache checks",
            ["result"]
        )
        
        # Start a separate HTTP server if prometheus_port is specified
        if getattr(args, "prometheus_port", None) is not None:
            prometheus_client.start_http_server(args.prometheus_port)
            logging.info(f"Prometheus metrics server started on port {args.prometheus_port}")
            
        _initialized = True
        return True
    except Exception as e:
        logging.error(f"Failed to initialize Prometheus metrics: {e}")
        return False

# Safe update functions
def update_queue_length(length):
    if queue_length_gauge is not None:
        queue_length_gauge.set(length)

def record_queue_wait(seconds):
    if queue_wait_histogram is not None:
        queue_wait_histogram.observe(seconds)

def record_job_duration(seconds):
    if job_duration_histogram is not None:
        job_duration_histogram.observe(seconds)

def increment_jobs_total(status):
    if jobs_counter is not None:
        jobs_counter.labels(status=status).inc()

def update_vram_metrics(device):
    if vram_gauge is None:
        return
    allocated, reserved = 0, 0
    try:
        import torch
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == 'cpu':
            return
        
        device_name = f"{device.type}:{device.index or 0}"
        
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
        elif device.type == 'mps' and hasattr(torch, 'mps'):
            allocated = torch.mps.current_allocated_memory()
            reserved = torch.mps.driver_allocated_memory() if hasattr(torch.mps, 'driver_allocated_memory') else allocated
        elif device.type == 'xpu' and hasattr(torch, 'xpu'):
            allocated = torch.xpu.memory_allocated(device)
            reserved = torch.xpu.memory_reserved(device)
        elif device.type == 'npu' and hasattr(torch, 'npu'):
            allocated = torch.npu.memory_allocated(device)
            reserved = torch.npu.memory_reserved(device)
        elif hasattr(torch, device.type) and hasattr(getattr(torch, device.type), 'memory_allocated'):
            dev_mod = getattr(torch, device.type)
            allocated = dev_mod.memory_allocated(device)
            reserved = dev_mod.memory_reserved(device)
            
        vram_gauge.labels(device=device_name, type="allocated").set(allocated)
        vram_gauge.labels(device=device_name, type="reserved").set(reserved)
    except Exception:
        pass

def update_loaded_models_count(count):
    if loaded_models_gauge is not None:
        loaded_models_gauge.set(count)

def increment_model_swaps():
    if model_swaps_counter is not None:
        model_swaps_counter.inc()

def record_node_execution(node_type, seconds):
    if node_execution_histogram is not None:
        node_execution_histogram.labels(node_type=node_type).observe(seconds)

def increment_cache_requests(result):
    if cache_requests_counter is not None:
        cache_requests_counter.labels(result=result).inc()
