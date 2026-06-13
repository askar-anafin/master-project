import os
import sys
import time
import torch
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.getcwd())

from src.train_unified import get_model

def get_model_file_size(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)  # Size in MB
    return 0.0

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_model(model, device, num_runs=100, warm_up=10):
    dummy_input = torch.randn(1, 5000, 12).to(device)
    model.eval()
    model.to(device)
    
    # Warm up runs
    with torch.no_grad():
        for _ in range(warm_up):
            _ = model(dummy_input)
            
    # Measure latency
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000.0)  # Convert to ms
            
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    throughput = 1000.0 / mean_lat  # samples per second
    
    return mean_lat, std_lat, throughput

def main():
    print("=" * 60)
    print("ECG MODEL INFERENCE LATENCY BENCHMARK")
    print("=" * 60)
    
    # Check device availability
    cpu_device = torch.device('cpu')
    gpu_available = torch.cuda.is_available()
    gpu_device = torch.device('cuda') if gpu_available else None
    
    print(f"CPU Device: {cpu_device}")
    print(f"GPU Available: {gpu_available}")
    if gpu_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print("-" * 60)
    
    models_config = {
        'cnn': ('cnn', 'experiments/cnn_filtered/best_model.pth'),
        'gnn': ('gnn', 'experiments/gnn_filtered/best_model.pth'),
        'vit': ('vit', 'experiments/vit_filtered/best_model.pth'),
        'hybrid': ('hybrid', 'experiments/hybrid_filtered/best_model.pth')
    }
    
    results = []
    
    for name, (type_name, path) in models_config.items():
        print(f"Benchmarking model: {name} (Type: {type_name})")
        if not os.path.exists(path):
            print(f"Checkpoint not found at {path}. Initializing weights randomly for benchmark.")
            model = get_model(type_name)
            file_size = 0.0
        else:
            try:
                model = get_model(type_name)
                model.load_state_dict(torch.load(path, map_location='cpu'))
                file_size = get_model_file_size(path)
                print(f"Loaded checkpoint from {path}")
            except Exception as e:
                print(f"Error loading checkpoint, using random init: {e}")
                model = get_model(type_name)
                file_size = get_model_file_size(path)
                
        params = count_parameters(model)
        
        # Benchmark CPU
        print("  Running CPU benchmark...")
        cpu_mean, cpu_std, cpu_throughput = benchmark_model(model, cpu_device)
        print(f"  CPU Latency: {cpu_mean:.2f} ms ± {cpu_std:.2f} ms | Throughput: {cpu_throughput:.1f} RPS")
        
        # Benchmark GPU (if available)
        gpu_mean, gpu_std, gpu_throughput = 0.0, 0.0, 0.0
        if gpu_available:
            print("  Running GPU benchmark...")
            gpu_mean, gpu_std, gpu_throughput = benchmark_model(model, gpu_device)
            print(f"  GPU Latency: {gpu_mean:.2f} ms ± {gpu_std:.2f} ms | Throughput: {gpu_throughput:.1f} RPS")
        else:
            print("  GPU benchmark skipped (not available).")
            
        results.append({
            'Model': name.upper(),
            'Parameters': params,
            'FileSize_MB': file_size,
            'CPU_Latency_ms': cpu_mean,
            'CPU_Std_ms': cpu_std,
            'CPU_Throughput_RPS': cpu_throughput,
            'GPU_Latency_ms': gpu_mean,
            'GPU_Std_ms': gpu_std,
            'GPU_Throughput_RPS': gpu_throughput
        })
        print("-" * 60)
        
    df = pd.DataFrame(results)
    
    # Save results
    output_dir = 'experiments'
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'latency_benchmark.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Display results summary
    print("\nBenchmark Summary Table:")
    print(df.to_string(index=False))
    print("=" * 60)

if __name__ == "__main__":
    main()
