import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
from sklearn.metrics import average_precision_score, f1_score, accuracy_score, classification_report

# Add project root to path
sys.path.append(os.getcwd())

import argparse
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, average_precision_score

from src.models import BaselineCNN1D, ResNet1D, ECGTransformer, InceptionTime
from src.models_gnn import STGNN
from src.ptbxl.loader import PTBXLDataset
from src.mitbih.loader import MITBIHDataset
from src.datasets.base import ECGDataset

def evaluate_model(model_type='baseline', dataset_name='ptbxl', split_strategy='inter_patient', use_class_weights=False):
    BATCH_SIZE = 32
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    print(f"Evaluating Model: {model_type} on {dataset_name} ({split_strategy}) [Weighted: {use_class_weights}]")

    # Dataset Config
    if dataset_name == 'ptbxl':
        DatasetClass = PTBXLDataset
        input_channels = 12
        num_classes = 5
    elif dataset_name == 'mitbih':
        DatasetClass = MITBIHDataset
        input_channels = 2
        num_classes = 5 
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load Data
    print("Loading test data...")
    dataset_loader = DatasetClass()
    try:
        splits = dataset_loader.get_train_val_test_split(split_strategy=split_strategy)
    except FileNotFoundError:
        print(f"Data not found for {dataset_name}.")
        return
    
    # Use ECGDataset wrapper (handles mmap and indices)
    test_data = ECGDataset(splits['X'], splits['y'], indices=splits['test_idx'], transform=None)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    models = []
    suffix = '_weighted' if use_class_weights else ''
    
    if model_type == 'ensemble':
        # Hybrid Ensemble Logic: If weighted, only weight Inception, keep others unweighted
        resnet_suffix = '' if (use_class_weights and dataset_name == 'mitbih') else suffix
        transformer_suffix = '' if (use_class_weights and dataset_name == 'mitbih') else suffix
        inception_suffix = suffix
        
        # Load ResNet
        if dataset_name == 'mitbih':
             p1 = f'models/{dataset_name}_{split_strategy}_resnet1d{resnet_suffix}.pth'
        else:
             p1 = f'models/{dataset_name}_resnet1d{resnet_suffix}.pth'
             
        if os.path.exists(p1):
            m1 = ResNet1D(num_classes=num_classes, input_channels=input_channels).to(DEVICE)
            m1.load_state_dict(torch.load(p1, map_location=DEVICE))
            m1.eval()
            models.append(m1)
            print(f"Loaded ResNet1D from {p1}")
        else:
            print(f"Warning: ResNet1D not found at {p1}")
        
        # Load Transformer
        if dataset_name == 'mitbih':
             p2 = f'models/{dataset_name}_{split_strategy}_transformer{transformer_suffix}.pth'
        else:
             p2 = f'models/{dataset_name}_transformer{transformer_suffix}.pth'
             
        if os.path.exists(p2):
            m2 = ECGTransformer(num_classes=num_classes, input_channels=input_channels).to(DEVICE)
            m2.load_state_dict(torch.load(p2, map_location=DEVICE))
            m2.eval()
            models.append(m2)
            print(f"Loaded ECGTransformer from {p2}")
        else:
            print(f"Warning: Transformer not found at {p2}")

        # Load InceptionTime
        if dataset_name == 'mitbih' and split_strategy == 'intra_patient':
             p3 = f'models/{dataset_name}_intra_inception{inception_suffix}.pth'
        else:
             p3 = f'models/{dataset_name}_inception{inception_suffix}.pth'
             
        if os.path.exists(p3):
            m3 = InceptionTime(num_classes=num_classes, input_channels=input_channels).to(DEVICE)
            m3.load_state_dict(torch.load(p3, map_location=DEVICE))
            m3.eval()
            models.append(m3)
            print(f"Loaded InceptionTime from {p3}")
        else:
            print(f"Warning: InceptionTime not found at {p3}")
        
        if not models:
            print("No models found for ensemble!")
            return
            
        print(f"Ensemble composed of {len(models)} models.")
    else:
        # Load Single Model
        if model_type == 'resnet':
            model = ResNet1D(num_classes=num_classes, input_channels=input_channels).to(DEVICE)
            model_path = f'models/{dataset_name}_resnet1d{suffix}.pth'
        elif model_type == 'transformer':
            model = ECGTransformer(num_classes=num_classes, input_channels=input_channels).to(DEVICE)
            model_path = f'models/{dataset_name}_transformer{suffix}.pth'
        elif model_type == 'inception':
            model = InceptionTime(num_classes=num_classes, input_channels=input_channels).to(DEVICE)
            if dataset_name == 'mitbih' and split_strategy == 'intra_patient':
                model_path = f'models/{dataset_name}_intra_inception{suffix}.pth'
            else:
                model_path = f'models/{dataset_name}_inception{suffix}.pth'
        elif model_type == 'stgnn':
            model = STGNN(num_classes=num_classes, num_nodes=input_channels).to(DEVICE)
            if dataset_name == 'mitbih':
                 model_path = f'models/{dataset_name}_{split_strategy}_stgnn{suffix}.pth'
            else:
                 model_path = f'models/{dataset_name}_stgnn{suffix}.pth'
        else:
            model = BaselineCNN1D(num_classes=num_classes, input_channels=input_channels).to(DEVICE)
            model_path = f'models/{dataset_name}_baseline_cnn{suffix}.pth'
            
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return
            
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        models.append(model)
        
    print("Running evaluation...")
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            batch_probs = []
            for m in models:
                outputs = m(X_batch)
                probs = torch.sigmoid(outputs)
                batch_probs.append(probs)
            
            # Average probabilities for ensemble
            avg_probs = torch.stack(batch_probs).mean(dim=0)
            
            all_preds.append(avg_probs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            
    # Calculate Metrics
    y_pred_probs = np.vstack(all_preds)
    y_true = np.vstack(all_targets)
    y_pred_binary = (y_pred_probs > 0.5).astype(int)
    
    # mAP
    map_macro = average_precision_score(y_true, y_pred_probs, average='macro')
    map_micro = average_precision_score(y_true, y_pred_probs, average='micro')
    
    # F1 Score
    # sklearn f1_score
    from sklearn.metrics import f1_score
    f1_macro_val = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
    f1_micro_val = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
    
    exact_match_acc = accuracy_score(y_true, y_pred_binary)

    print("\n--- Evaluation Results ---")
    print(f"mAP (Macro): {map_macro:.4f}")
    print(f"mAP (Micro): {map_micro:.4f}")
    print(f"F1 Score (Macro): {f1_macro_val:.4f}")
    print(f"F1 Score (Micro): {f1_micro_val:.4f}")
    print(f"Exact Match Accuracy: {exact_match_acc:.4f}")
    
    # Detailed Report
    from sklearn.metrics import classification_report
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP'] if dataset_name == 'ptbxl' else ['N', 'S', 'V', 'F', 'Q']
    
    print("\n--- Per Class Report ---")
    print(classification_report(y_true, y_pred_binary, target_names=classes, zero_division=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'resnet', 'transformer', 'ensemble', 'inception', 'stgnn'], help='Model architecture to evaluate')
    parser.add_argument('--dataset', type=str, default='ptbxl', choices=['ptbxl', 'mitbih'], help='Dataset to evaluate on')
    parser.add_argument('--split_strategy', type=str, default='inter_patient', choices=['inter_patient', 'intra_patient'], help='Splitting strategy for MIT-BIH')
    parser.add_argument('--use_class_weights', action='store_true', help='Use weighted model')
    args = parser.parse_args()
    
    evaluate_model(model_type=args.model, dataset_name=args.dataset, split_strategy=args.split_strategy, use_class_weights=args.use_class_weights)
