
import os
import sys
import torch
import numpy as np
import random
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.getcwd())

from src.models import ResNet1D, ECGTransformer, InceptionTime
from src.mitbih.loader import MITBIHDataset
from src.ptbxl.loader import PTBXLDataset
from src.datasets.base import ECGDataset

def demo_predictions(num_samples=10, specific_indices=None, mode='arrhythmia'):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    print(f"Running Diagnostic Mode: {mode.upper()}")
    
    if mode == 'arrhythmia':
        # Dataset Config
        dataset_name = 'mitbih'
        split_strategy = 'intra_patient'
        input_channels = 2
        num_classes = 5
        classes = ['N', 'S', 'V', 'F', 'Q']
        class_map = {
            'N': 'Normal / Bundle Branch Block',
            'S': 'Supraventricular (incl. A-Flutter)',
            'V': 'Ventricular (incl. V-Tach)',
            'F': 'Fusion Beat',
            'Q': 'Unknown / Paced'
        }
        DatasetLoader = MITBIHDataset
    elif mode == 'disease':
        # Dataset Config for PTB-XL (MI, Ischemia, etc.)
        dataset_name = 'ptbxl'
        split_strategy = 'inter_patient' # PTB-XL default
        input_channels = 12
        num_classes = 5
        classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        class_map = {
            'NORM': 'Normal ECG',
            'MI': 'Myocardial Infarction',
            'STTC': 'ST/T Change (Ischemia)',
            'CD': 'Conduction Disturbance',
            'HYP': 'Hypertrophy'
        }
        DatasetLoader = PTBXLDataset
    else:
        print("Invalid mode. Use 'arrhythmia' or 'disease'.")
        return

    # Load Data
    print(f"Loading {dataset_name} test data...")
    dataset_loader = DatasetLoader()
    try:
        splits = dataset_loader.get_train_val_test_split(split_strategy=split_strategy)
    except FileNotFoundError:
        print(f"Data not found for {dataset_name}.")
        return
    
    # Use ECGDataset wrapper
    test_data = ECGDataset(splits['X'], splits['y'], indices=splits['test_idx'], transform=None)
    
    # Select Samples
    selected_indices = []
    
    # Try to pick diverse samples if none provided
    if specific_indices is None:
        y_test = splits['y'][splits['test_idx']]
        y_labels = np.argmax(y_test, axis=1) # Convert one-hot to index
        
        samples_per_class = num_samples // num_classes
        
        for class_idx in range(num_classes):
            # Find indices for this class
            class_indices = np.where(y_labels == class_idx)[0]
            if len(class_indices) > 0:
                picked = np.random.choice(class_indices, min(len(class_indices), 2), replace=False)
                selected_indices.extend(picked)
        
        if len(selected_indices) < num_samples:
            remaining = num_samples - len(selected_indices)
            all_indices = np.arange(len(test_data))
            extras = np.random.choice(all_indices, remaining, replace=False)
            selected_indices.extend(extras)
    else:
        selected_indices = specific_indices

    print(f"Selected {len(selected_indices)} samples for demonstration.")

    # Load Models
    models = []
    
    if mode == 'arrhythmia':
        # Load Ensemble Models (Previous Logic)
        p1 = f'models/{dataset_name}_{split_strategy}_resnet1d.pth'
        if os.path.exists(p1):
            m1 = ResNet1D(num_classes=num_classes, input_channels=input_channels).to(DEVICE)
            m1.load_state_dict(torch.load(p1, map_location=DEVICE))
            m1.eval()
            models.append(m1)
        
        p2 = f'models/{dataset_name}_{split_strategy}_transformer.pth'
        if os.path.exists(p2):
            m2 = ECGTransformer(num_classes=num_classes, input_channels=input_channels).to(DEVICE)
            m2.load_state_dict(torch.load(p2, map_location=DEVICE))
            m2.eval()
            models.append(m2)
            
        p3 = f'models/{dataset_name}_intra_inception_weighted.pth'
        if os.path.exists(p3):
            m3 = InceptionTime(num_classes=num_classes, input_channels=input_channels).to(DEVICE)
            m3.load_state_dict(torch.load(p3, map_location=DEVICE))
            m3.eval()
            models.append(m3)
            
    elif mode == 'disease':
        # Load PTB-XL InceptionTime Model
        p1 = f'models/ptbxl_inception.pth'
        # Fallback names if original training setup differed
        if not os.path.exists(p1):
             p1 = 'models/inception.pth' 
             
        if os.path.exists(p1):
            m1 = InceptionTime(num_classes=num_classes, input_channels=input_channels).to(DEVICE)
            m1.load_state_dict(torch.load(p1, map_location=DEVICE))
            m1.eval()
            models.append(m1)
        else:
            print(f"PTB-XL model not found at {p1}")
            return

    if not models:
        print("No models found!")
        return

    print("Running inference...")
    print("-" * 140)
    # Adjust column headers based on classes
    header_classes = ", ".join(classes)
    print(f"{'True Label':<10} | {'Description':<40} | {'Pred':<5} | {'Conf':<6} | " + f"{'Likelihoods (' + header_classes + ')':<50}")
    print("-" * 140)

    with torch.no_grad():
        for idx in selected_indices:
            try:
                x, y = test_data[idx]
                x = x.unsqueeze(0).to(DEVICE) # Add batch dim
                y_true_idx = torch.argmax(y).item()
                y_true_label = classes[y_true_idx]
                
                # Prediction
                batch_probs = []
                for m in models:
                    output = m(x)
                    probs = torch.sigmoid(output)
                    batch_probs.append(probs)
                
                avg_probs = torch.stack(batch_probs).mean(dim=0).squeeze()
                
                # Top Prediction
                y_pred_idx = torch.argmax(avg_probs).item()
                y_pred_label = classes[y_pred_idx]
                confidence = avg_probs[y_pred_idx].item()
                
                # Format likelihoods
                likelihoods = [f"{p:.2f}" for p in avg_probs.tolist()]
                likelihood_str = ", ".join([f"{c}:{l}" for c, l in zip(classes, likelihoods)])
                
                # Status indicator
                status = "[OK]" if y_true_idx == y_pred_idx else "[XX]"
                
                # Description
                description = class_map.get(y_pred_label, "Unknown")
                
                print(f"{y_true_label:<10} | {description:<40} | {y_pred_label:<5} | {confidence:.0%} | {likelihood_str} {status}")
                
            except IndexError:
                print(f"Index error at {idx}")
                
    print("-" * 140)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='arrhythmia', choices=['arrhythmia', 'disease'], help='Diagnosis mode')
    args = parser.parse_args()
    
    demo_predictions(num_samples=10, mode=args.mode)
