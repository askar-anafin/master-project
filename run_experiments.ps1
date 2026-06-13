Write-Host "Starting Full Training Sequence for PTB-XL Models..."

# 1. CNN (ResNet-18)
# Write-Host "-------------------------------------------"
# Write-Host "Training CNN (ResNet-18)..."
# .\.venv\Scripts\python.exe -u src/train_unified.py --model cnn --name cnn_full --epochs 100 --patience 10 --batch_size 64 --num_workers 0

# 2. Mamba (S2M2ECG/S4)
# Write-Host "-------------------------------------------"
# Write-Host "Training Mamba (S2M2ECG)..."
# .\.venv\Scripts\python.exe -u src/train_unified.py --model mamba --name mamba_full --epochs 100 --patience 10 --batch_size 64 --num_workers 0 --limit 5000

# 3. GNN (ST-ReGE)
# Write-Host "-------------------------------------------"
# Write-Host "Training GNN (ST-ReGE)..."
# .\.venv\Scripts\python.exe -u src/train_unified.py --model gnn --name gnn_full --epochs 100 --patience 10 --batch_size 64 --num_workers 0

# 4. ViT (Transformer)
Write-Host "-------------------------------------------"
Write-Host "Training ViT (Transformer)..."
.\.venv\Scripts\python.exe -u src/train_unified.py --model vit --name vit_full --epochs 100 --patience 10 --batch_size 64 --num_workers 0

# 5. Hybrid (ResNet-Transformer)
Write-Host "-------------------------------------------"
Write-Host "Training Hybrid (ResNet-Transformer)..."
.\.venv\Scripts\python.exe -u src/train_unified.py --model hybrid --name hybrid_full --epochs 100 --patience 10 --batch_size 64 --num_workers 0

Write-Host "-------------------------------------------"
Write-Host "All Experiments Completed."
