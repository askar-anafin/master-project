import torch
import numpy as np
import pandas as pd
import sys
import os

# Add project root
sys.path.append(os.getcwd())

from src.ptbxl.loader import PTBXLDataset
from src.train_unified import get_model

def load_checkpoint(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def run_demo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Demo on {device}")
    
    # Load Data (Test Set)
    dataset = PTBXLDataset()
    data = dataset.get_train_val_test_split()
    X_test = data['X'][data['test_idx']]
    y_test = data['y'][data['test_idx']]
    
    # Pick a random sample with at least one positive label
    index = np.random.randint(0, len(X_test))
    # Ensure it has a label
    while y_test[index].sum() == 0:
        index = np.random.randint(0, len(X_test))
        
    sample_x = torch.tensor(X_test[index]).unsqueeze(0).float().to(device) # (1, 12, 5000)
    sample_y = y_test[index]
    
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    true_labels = [classes[i] for i, val in enumerate(sample_y) if val == 1]
    
    print(f"\n--- Sample ID: {index} ---")
    print(f"True Labels: {', '.join(true_labels)}")
    
    # Load Models
    models = {
        'CNN (ResNet)': ('cnn', 'experiments/cnn_full/best_model.pth'),
        'GNN (ST-ReGE)': ('gnn', 'experiments/gnn_full/best_model.pth'),
        'ViT (Transformer)': ('vit', 'experiments/vit_full/best_model.pth')
    }
    
    results = {}
    
    for name, (type_name, path) in models.items():
        if not os.path.exists(path):
            print(f"Skipping {name} (Checkpoint not found)")
            continue
            
        try:
            model = get_model(type_name) # Removed device arg
            model = load_checkpoint(model, path)
            model.to(device)
            
            with torch.no_grad():
                logits = model(sample_x)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                
            preds = [classes[i] for i, p in enumerate(probs) if p > 0.5]
            results[name] = (preds, probs)
            
        except Exception as e:
            print(f"Error running {name}: {e}")
            
    # Print Results
    print("\n--- Model Predictions ---")
    for name, (preds, probs) in results.items():
        pred_str = ", ".join(preds) if preds else "None"
        # Format probs
        prob_str = " | ".join([f"{c}: {p:.2f}" for c, p in zip(classes, probs)])
        
        print(f"\n{name}:")
        print(f"  Prediction: {pred_str}")
        print(f"  Confidence: {prob_str}")
        
        # Check correctness (Exact Match)
        is_correct = set(preds) == set(true_labels)
        print(f"  Status: {'[CORRECT]' if is_correct else '[INCORRECT]'}")

if __name__ == "__main__":
    run_demo()
