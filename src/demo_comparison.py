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
    
    # Define classes
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    # Load Models
    models = {}
    model_configs = {
        'CNN (ResNet)': ('cnn', 'experiments/cnn_full/best_model.pth'),
        'GNN (ST-ReGE)': ('gnn', 'experiments/gnn_full/best_model.pth'),
        'ViT (Transformer)': ('vit', 'experiments/vit_full/best_model.pth')
    }
    
    for name, (type_name, path) in model_configs.items():
        if os.path.exists(path):
            try:
                model = get_model(type_name)
                model = load_checkpoint(model, path)
                model.to(device)
                models[name] = model
            except Exception as e:
                print(f"Error loading {name}: {e}")
    
    print("\n--- Multi-Class Diagnosis Demo ---")
    
    # Find one example for each class
    samples_found = {}
    
    # Shuffle indices to get random examples each time
    indices = np.random.permutation(len(X_test))
    
    for idx in indices:
        y = y_test[idx]
        if y.sum() == 0: continue # Skip unlabeled
        
        # Get labels for this sample
        sample_labels = [classes[i] for i, val in enumerate(y) if val == 1]
        
        # We try to find a sample that has *only* this specific label if possible, or at least includes it
        for label in sample_labels:
            if label not in samples_found:
                samples_found[label] = idx
                
        if len(samples_found) == 5:
            break
            
    print("Starting Multi-Class Demo...", flush=True)
    
    # Run Inference
    for label in classes:
        if label not in samples_found:
            print(f"\nCould not find test example for {label}", flush=True)
            continue
            
        idx = samples_found[label]
        sample_x = torch.tensor(X_test[idx]).unsqueeze(0).float().to(device)
        sample_y = y_test[idx]
        true_labels = [classes[i] for i, val in enumerate(sample_y) if val == 1]
        
        print(f"\n=== Case: {label} (Sample ID: {idx}) ===", flush=True)
        print(f"True Diagnosis: {', '.join(true_labels)}", flush=True)
        
        for name, model in models.items():
            try:
                with torch.no_grad():
                    logits = model(sample_x)
                    probs = torch.sigmoid(logits).cpu().numpy()[0]
                
                preds = [classes[i] for i, p in enumerate(probs) if p > 0.5]
                pred_str = ", ".join(preds) if preds else "None"
                
                # Get confidence specifically for the target label we are looking at
                target_idx = classes.index(label)
                conf = probs[target_idx]
                
                is_correct = label in preds
                status = "[CORRECT]" if is_correct else "[MISSED]"
                if not is_correct and len(preds) > 0: status = "[WRONG]" # Predicted something else
                
                print(f"  {name}: {pred_str} | Conf({label}): {conf:.2f} {status}", flush=True)
                
            except Exception as e:
                print(f"  {name}: Error {e}", flush=True)

if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
