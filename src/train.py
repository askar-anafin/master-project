import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.models import BaselineCNN1D
from src.data_loader import PTBXLDataset

def train_model():
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Load Data
    print("Loading data...")
    dataset = PTBXLDataset()
    try:
        splits = dataset.get_train_val_test_split()
    except FileNotFoundError:
        print("Data not processed yet. Run scripts/process_ptbxl_data.py first.")
        return

    # Create DataLoaders
    # Note: X is (N, 5000, 12), y is (N, 5)
    train_data = TensorDataset(torch.from_numpy(splits['X_train']), torch.from_numpy(splits['y_train']))
    val_data = TensorDataset(torch.from_numpy(splits['X_val']), torch.from_numpy(splits['y_val']))
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    model = BaselineCNN1D(num_classes=5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
                # Simple accuracy for multi-label (exact match is hard, let's use threshold 0.5)
                preds = torch.sigmoid(outputs) > 0.5
                # This is strict exact match accuracy
                # correct_preds += (preds == y_batch.bool()).all(dim=1).sum().item()
                
                # Let's use average accuracy (hamming score substitute)
                # correct_preds += (preds == y_batch.bool()).float().mean().item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
    # Save Model
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 'models/baseline_cnn.pth')
    print("Model saved to models/baseline_cnn.pth")

if __name__ == "__main__":
    train_model()
