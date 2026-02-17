import torch
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.getcwd())

from src.models import BaselineCNN1D, ResNet1D, ECGTransformer, InceptionTime
from src.ptbxl.loader import PTBXLDataset
from src.mitbih.loader import MITBIHDataset

def predict_single(model_type='inception', index=None, plot=True, dataset_name='ptbxl'):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    print(f"Dataset: {dataset_name}")

    if dataset_name == 'ptbxl':
        DatasetClass = PTBXLDataset
        CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        input_channels = 12
        num_classes = 5
    elif dataset_name == 'mitbih':
        DatasetClass = MITBIHDataset
        CLASSES = ['N', 'S', 'V', 'F', 'Q']
        input_channels = 2
        num_classes = 5
    else:
        print(f"Unknown dataset: {dataset_name}")
        return

    # Load Dataset
    print(f"Loading {dataset_name} dataset...")
    dataset = DatasetClass()
    try:
        # We need the full data to index into it, but we should predict on Test set samples ideally
        # Let's get the splits to identify test indices
        splits = dataset.get_train_val_test_split()
        test_indices = splits['test_idx']
        X_all = splits['X']
        y_all = splits['y']
    except FileNotFoundError:
        print("Data not found. Ensure you have processed data.")
        return

    # Select Index
    if index is None:
        # Pick a random index from the test set
        idx = np.random.choice(test_indices)
        print(f"Selected random test sample at index: {idx}")
    else:
        # User provided index. Check if it's valid.
        idx = int(index)
        if idx < 0 or idx >= len(X_all):
            print(f"Index {idx} out of bounds (0-{len(X_all)-1}).")
            return
        # check if it is in test set
        if idx in test_indices:
             print(f"Index {idx} is in the Test set.")
        else:
             print(f"Index {idx} is NOT in the Test set (Train or Val).")

    # Get sample
    # X_all is mmap, so slicing loads it into memory
    signal = X_all[idx].copy() # Shape (5000, 12) or (5000, 2)
    label = y_all[idx]        # Shape (5,)

    input_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(DEVICE) # (1, 5000, Channels)

    # Load Model
    print(f"Loading {model_type} model...")
    if model_type == 'resnet':
        model = ResNet1D(num_classes=num_classes, input_channels=input_channels).to(DEVICE)
        model_path = f'models/{dataset_name}_resnet1d.pth'
    elif model_type == 'transformer':
        model = ECGTransformer(num_classes=num_classes, input_channels=input_channels).to(DEVICE)
        model_path = f'models/{dataset_name}_transformer.pth'
    elif model_type == 'inception':
        model = InceptionTime(num_classes=num_classes, input_channels=input_channels).to(DEVICE)
        model_path = f'models/{dataset_name}_inception.pth'
    else:
        model = BaselineCNN1D(num_classes=num_classes, input_channels=input_channels).to(DEVICE)
        model_path = f'models/{dataset_name}_baseline_cnn.pth'
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0] # (5,)

    # Display Results
    print("\n--- Prediction Results ---")
    print(f"{'Class':<10} {'Probability':<15} {'Ground Truth':<15}")
    print("-" * 40)
    
    for i, class_name in enumerate(CLASSES):
        prob = probs[i]
        truth = int(label[i])
        marker = "<-- PREDICTED" if prob > 0.5 else ""
        truth_marker = "(Actual)" if truth == 1 else ""
        
        # Color code likely predictions
        start_bold = "\033[1m" if prob > 0.5 or truth == 1 else ""
        end_bold = "\033[0m" if prob > 0.5 or truth == 1 else ""
        
        print(f"{start_bold}{class_name:<10} {prob:.4f}          {truth:<15} {marker} {truth_marker}{end_bold}")

    print("-" * 40)

    # Visualization
    if plot:
        print("Plotting ECG signal...")
        plt.figure(figsize=(15, 6))
        
        # Plot up to 4 leads
        num_leads = signal.shape[1]
        leads_to_plot = range(min(4, num_leads))
        
        for i, lead_idx in enumerate(leads_to_plot):
            plt.subplot(len(leads_to_plot), 1, i+1)
            plt.plot(signal[:, lead_idx])
            plt.title(f'Lead {lead_idx+1}')
            plt.grid(True)
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='inception', choices=['baseline', 'resnet', 'transformer', 'inception'], help='Model to use')
    parser.add_argument('--dataset', type=str, default='ptbxl', choices=['ptbxl', 'mitbih'], help='Dataset to use')
    parser.add_argument('--index', type=int, default=None, help='Index of valid recording to predict (default: random test sample)')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    
    args = parser.parse_args()
    
    predict_single(model_type=args.model, index=args.index, plot=not args.no_plot, dataset_name=args.dataset)
