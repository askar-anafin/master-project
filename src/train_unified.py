
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root
sys.path.append(os.getcwd())

from src.ptbxl.loader import PTBXLDataset
from src.datasets.base import ECGDataset
from src.augmentations import Augmentation
from src.metrics import calculate_metrics, compute_physionet_score, save_metrics

# Architectures
from src.models.cnn import ResNet18
from src.models.mamba import MambaECG
from src.models.gnn import STReGE
from src.models.vit import ViT1D

def get_model(model_name, num_classes=5, input_channels=12):
    if model_name == 'cnn':
        return ResNet18(num_classes=num_classes, input_channels=input_channels)
    elif model_name == 'mamba':
        return MambaECG(num_classes=num_classes, input_channels=input_channels, d_model=128, num_layers=4)
    elif model_name == 'gnn':
        return STReGE(num_classes=num_classes, num_nodes=input_channels, feature_dim=256)
    elif model_name == 'vit':
        return ViT1D(num_classes=num_classes, input_channels=input_channels, seq_len=5000, patch_size=50, d_model=128, nhead=4, num_layers=4)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_experiment(model_name, experiment_name=None, limit=None, epochs=20, batch_size=32, lr=1e-3, num_workers=4):
    
    # 1. Setup Directories
    if experiment_name is None:
        experiment_name = model_name
        
    output_dir = os.path.join('experiments', experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Starting Experiment: {experiment_name} ---")
    print(f"Output Directory: {output_dir}")
    print(f"Batch Size: {batch_size}, Workers: {num_workers}")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")
    
    # Enable Benchmark for speed
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # 2. Load Data
    print("Loading PTB-XL Data...")
    dataset = PTBXLDataset()
    try:
        splits = dataset.get_train_val_test_split(limit=limit)
    except FileNotFoundError:
        print("Data not processed. Please run preprocessing first.")
        return

    augmenter = Augmentation()
    train_data = ECGDataset(splits['X'], splits['y'], indices=splits['train_idx'], transform=augmenter)
    val_data = ECGDataset(splits['X'], splits['y'], indices=splits['val_idx'], transform=None)
    test_data = ECGDataset(splits['X'], splits['y'], indices=splits['test_idx'], transform=None)
    
    # Optimization: num_workers and pin_memory
    dataloader_kwargs = {'num_workers': num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}
    
    # Drop last to avoid batch norm issues with size 1
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, **dataloader_kwargs)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, **dataloader_kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, **dataloader_kwargs)
    
    # 3. Model & Optimization
    model = get_model(model_name).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_score = 0.0
    
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE, non_blocking=True), y_batch.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # AMP Context
            with torch.cuda.amp.autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            
            # Scaled Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        val_f1 = val_metrics['Macro_F1']
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_metrics['Loss']:.4f} - Val F1: {val_f1:.4f}")
        
        scheduler.step(val_f1)
        
        if val_f1 > best_val_score:
            best_val_score = val_f1
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"  [+] New Best Model Saved! ({val_f1:.4f})")
            
    # 5. Final Evaluation on Test Set
    print("\n--- Final Evaluation (Test Set) ---")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=DEVICE))
    test_metrics = evaluate(model, test_loader, criterion, DEVICE)
    
    # Calculate PhysioNet Score (requires class names if available, using indices for now)
    # Re-running inference to get full arrays for PhysioNet Score
    y_true_all, y_prob_all = predict_all(model, test_loader, DEVICE)
    physio_score = compute_physionet_score(y_true_all, y_prob_all)
    test_metrics['PhysioNet_Score'] = physio_score
    
    print(f"Test F1: {test_metrics['Macro_F1']:.4f}")
    print(f"Test AUPRC: {test_metrics['Macro_AUPRC']:.4f}")
    print(f"Test AUROC: {test_metrics['Macro_AUROC']:.4f}")
    print(f"PhysioNet Score: {physio_score:.4f}")
    
    # 6. Save Metrics
    save_path = os.path.join(output_dir, 'results.csv')
    save_metrics(test_metrics, classes=None, model_name=model_name, output_path=save_path)
    print(f"Results saved to {save_path}")

def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    y_true_list = []
    y_prob_list = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            losses.append(loss.item())
            
            probs = torch.sigmoid(outputs)
            
            y_true_list.append(y_batch.float().cpu().numpy())
            y_prob_list.append(probs.float().cpu().numpy())
            
    y_true = np.vstack(y_true_list)
    y_prob = np.vstack(y_prob_list)
    
    metrics = calculate_metrics(y_true, y_prob)
    metrics['Loss'] = np.mean(losses)
    
    return metrics

def predict_all(model, loader, device):
    model.eval()
    y_true_list = []
    y_prob_list = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(X_batch)
                probs = torch.sigmoid(outputs)
            y_true_list.append(y_batch.float().cpu().numpy())
            y_prob_list.append(probs.float().cpu().numpy())
    return np.vstack(y_true_list), np.vstack(y_prob_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['cnn', 'mamba', 'gnn', 'vit'])
    parser.add_argument('--name', type=str, help='Custom experiment name (folder name)')
    parser.add_argument('--limit', type=int, help='Limit dataset size for debugging')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    
    train_experiment(args.model, experiment_name=args.name, limit=args.limit, epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers)
