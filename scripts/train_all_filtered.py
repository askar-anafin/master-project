"""
Run all 4 models sequentially on the filtered dataset.
Saves results to experiments/{model}_filtered/
"""
import subprocess
import sys
import os
import time

models = ['gnn', 'vit', 'hybrid']

for model in models:
    name = f"{model}_filtered"
    print(f"\n{'='*60}", flush=True)
    print(f"  Training: {model.upper()} -> experiments/{name}", flush=True)
    print(f"{'='*60}", flush=True)
    start = time.time()
    
    result = subprocess.run(
        [
            sys.executable, '-m', 'src.train_unified',
            '--model', model,
            '--name', name,
            '--epochs', '50',
            '--batch_size', '32',
            '--patience', '10',
            '--num_workers', '0',   # 0 workers avoids multiprocessing CUDA issues
        ],
        cwd=os.getcwd()
    )
    
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n[OK] {model.upper()} done in {elapsed/60:.1f} min", flush=True)
    else:
        print(f"\n[FAIL] {model.upper()} failed (exit code {result.returncode})", flush=True)
        sys.exit(result.returncode)

print("\n\nAll models trained successfully!", flush=True)
