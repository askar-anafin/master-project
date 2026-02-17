import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import argparse
from sklearn.metrics import f1_score

# Add project root to path
sys.path.append(os.getcwd())

from src.models import BaselineCNN1D, ResNet1D, ECGTransformer, InceptionTime
from src.models_gnn import STGNN
from src.ptbxl.loader import PTBXLDataset
from src.mitbih.loader import MITBIHDataset
from src.datasets.base import ECGDataset
from src.augmentations import Augmentation

def train_model(limit=None, model_type='baseline', dataset_name='ptbxl', split_strategy='inter_patient', use_class_weights=False):
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.0005 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset Config
    if dataset_name == 'ptbxl':
        print("Using PTB-XL Dataset")
        DatasetClass = PTBXLDataset
        input_channels = 12
        num_classes = 5
    elif dataset_name == 'mitbih':
        print(f"Using MIT-BIH Dataset ({split_strategy} split)")
        DatasetClass = MITBIHDataset
        input_channels = 2
        num_classes = 5 # N, S, V, F, Q
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if limit:
        print(f"Training with limit: {limit}")
        EPOCHS = 10 
    else:
        print("Training on FULL dataset")
        EPOCHS = 15 
        
    print(f"Using device: {DEVICE}", flush=True)
    print(f"Training Model: {model_type}", flush=True)

    # Load Data
    print("Loading data...", flush=True)
    dataset_loader = DatasetClass()
    try:
        splits = dataset_loader.get_train_val_test_split(limit=limit, split_strategy=split_strategy)
    except FileNotFoundError:
        print(f"Data not processed yet for {dataset_name}.", flush=True)
        return

    # Class Weights Calculation
    pos_weight = None
    if use_class_weights:
        print("Calculating class weights...", flush=True)
        try:
             # Get training labels
            if 'train_idx' in splits:
                 y_train = splits['y'][splits['train_idx']]
            else:
                 # Fallback if splits structure is different (unlikely)
                 y_train = splits['y']
            
            # Calculate counts
            num_pos = np.sum(y_train, axis=0)
            num_neg = len(y_train) - num_pos
            
            # Avoid division by zero
            num_pos = np.maximum(num_pos, 1)
            
            # Compute weights: neg / pos
            weights = num_neg / num_pos
            
            # Limit extreme weights (optional, but good for stability)
            weights = np.clip(weights, 1.0, 20.0) 
            
            pos_weight = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
            print(f"Class Weights applied: {weights}", flush=True)
            
        except Exception as e:
            print(f"Error calculating class weights: {e}. Proceeding without.", flush=True)

    # Augmentation
    augmenter = Augmentation()
    print("Data Augmentation enabled for training set.", flush=True)

    # Create Datasets
    train_data = ECGDataset(splits['X'], splits['y'], indices=splits['train_idx'], transform=augmenter)
    val_data = ECGDataset(splits['X'], splits['y'], indices=splits['val_idx'], transform=None)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    suffix = '_weighted' if use_class_weights else ''
    
    if model_type == 'resnet':
        model = ResNet1D(num_classes=num_classes, input_channels=input_channels).to(DEVICE)
        save_path = f'models/{dataset_name}_{split_strategy}_resnet1d{suffix}.pth' if dataset_name == 'mitbih' else f'models/{dataset_name}_resnet1d{suffix}.pth'
    elif model_type == 'transformer':
        LEARNING_RATE = 0.0001
        model = ECGTransformer(num_classes=num_classes, input_channels=input_channels).to(DEVICE)
        save_path = f'models/{dataset_name}_{split_strategy}_transformer{suffix}.pth' if dataset_name == 'mitbih' else f'models/{dataset_name}_transformer{suffix}.pth'
    elif model_type == 'inception':
        LEARNING_RATE = 0.0001
        model = InceptionTime(num_classes=num_classes, input_channels=input_channels).to(DEVICE)
        # Unique save path for intra-patient experiment
        if dataset_name == 'mitbih' and split_strategy == 'intra_patient':
             save_path = f'models/{dataset_name}_intra_inception{suffix}.pth'
        else:
             save_path = f'models/{dataset_name}_inception{suffix}.pth'
    elif model_type == 'stgnn':
        LEARNING_RATE = 0.0001
        # input_channels here refers to number of leads, which STGNN expects as num_nodes
        # Feature dim logic: STGNN uses 256 internal dim
        model = STGNN(num_classes=num_classes, num_nodes=input_channels).to(DEVICE)
        save_path = f'models/{dataset_name}_{split_strategy}_stgnn{suffix}.pth' if dataset_name == 'mitbih' else f'models/{dataset_name}_stgnn{suffix}.pth'
    else:
        model = BaselineCNN1D(num_classes=num_classes, input_channels=input_channels).to(DEVICE)
        save_path = f'models/{dataset_name}_{split_strategy}_baseline_cnn{suffix}.pth' if dataset_name == 'mitbih' else f'models/{dataset_name}_baseline_cnn{suffix}.pth'

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    best_val_f1 = 0.0
    
    print("Starting training...", flush=True)
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1} Batch {i}/{len(train_loader)} Loss: {loss.item():.4f}", flush=True)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
                preds = torch.sigmoid(outputs)
                all_val_preds.append(preds.cpu().numpy())
                all_val_targets.append(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        
        # Calculate Metrics
        val_preds = np.vstack(all_val_preds)
        val_targets = np.vstack(all_val_targets)
        val_binary_preds = (val_preds > 0.5).astype(int)
        
        val_f1_macro = f1_score(val_targets, val_binary_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val F1 (Macro): {val_f1_macro:.4f}", flush=True)
        
        scheduler.step(val_f1_macro)
        
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} (F1: {best_val_f1:.4f})", flush=True)
        
    print("Training complete.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Limit number of records to use for training')
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'resnet', 'transformer', 'inception', 'stgnn'], help='Model architecture to train')
    parser.add_argument('--dataset', type=str, default='ptbxl', choices=['ptbxl', 'mitbih'], help='Dataset to use')
    parser.add_argument('--split_strategy', type=str, default='inter_patient', choices=['inter_patient', 'intra_patient'], help='Splitting strategy for MIT-BIH')
    parser.add_argument('--use_class_weights', action='store_true', help='Apply class weights to loss function')
    args = parser.parse_args()
    
    train_model(limit=args.limit, model_type=args.model, dataset_name=args.dataset, split_strategy=args.split_strategy, use_class_weights=args.use_class_weights)
