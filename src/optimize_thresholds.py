import torch
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, accuracy_score

# Add project root
sys.path.append(os.getcwd())

from src.ptbxl.loader import PTBXLDataset
from src.datasets.base import ECGDataset
from src.train_unified import get_model
from src.metrics import compute_physionet_score

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def predict(model, loader, device):
    model.eval()
    all_probs = []
    all_truths = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(X_batch)
            else:
                outputs = model(X_batch)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_truths.append(y_batch.numpy())
    return np.vstack(all_truths), np.vstack(all_probs)

def optimize_thresholds(y_true, y_probs):
    """
    Find optimal class-specific thresholds on validation set to maximize F1.
    """
    best_thresholds = []
    for c in range(len(CLASSES)):
        best_f1 = 0.0
        best_thresh = 0.5
        # Grid search from 0.01 to 0.99
        for thresh in np.linspace(0.01, 0.99, 99):
            preds = (y_probs[:, c] > thresh).astype(int)
            f1 = f1_score(y_true[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        best_thresholds.append(best_thresh)
    return np.array(best_thresholds)

def evaluate_predictions(y_true, y_probs, thresholds):
    """
    Evaluate using class-specific thresholds (can be a scalar 0.5 or vector of length 5).
    """
    if np.isscalar(thresholds):
        thresholds = np.full(len(CLASSES), thresholds)
        
    y_pred = np.zeros_like(y_true)
    for c in range(len(CLASSES)):
        y_pred[:, c] = (y_probs[:, c] > thresholds[c]).astype(int)
        
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    macro_auroc = roc_auc_score(y_true, y_probs, average='macro')
    macro_auprc = average_precision_score(y_true, y_probs, average='macro')
    exact_match = accuracy_score(y_true, y_pred)
    physionet = compute_physionet_score(y_true, y_probs)
    
    # Class-wise F1
    class_f1s = []
    for c in range(len(CLASSES)):
        class_f1s.append(f1_score(y_true[:, c], y_pred[:, c], zero_division=0))
        
    return {
        'f1': macro_f1,
        'auroc': macro_auroc,
        'auprc': macro_auprc,
        'exact_match': exact_match,
        'physionet': physionet,
        'class_f1s': class_f1s
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load data
    print("Loading PTB-XL splits...")
    dataset = PTBXLDataset()
    splits = dataset.get_train_val_test_split()
    
    val_data = ECGDataset(splits['X'], splits['y'], indices=splits['val_idx'], transform=None)
    test_data = ECGDataset(splits['X'], splits['y'], indices=splits['test_idx'], transform=None)
    
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    models_info = {
        'CNN': ('cnn', 'experiments/cnn_filtered/best_model.pth'),
        'GNN': ('gnn', 'experiments/gnn_filtered/best_model.pth'),
        'ViT': ('vit', 'experiments/vit_filtered/best_model.pth'),
        'Hybrid': ('hybrid', 'experiments/hybrid_filtered/best_model.pth')
    }
    
    # Store predictions
    val_probs_dict = {}
    test_probs_dict = {}
    y_val_true = None
    y_test_true = None
    
    for label, (model_type, model_path) in models_info.items():
        if not os.path.exists(model_path):
            print(f"Warning: checkpoint not found for {label} at {model_path}")
            continue
            
        print(f"Running inference for {label}...")
        model = get_model(model_type).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        y_val_true, val_probs = predict(model, val_loader, device)
        y_test_true, test_probs = predict(model, test_loader, device)
        
        val_probs_dict[label] = val_probs
        test_probs_dict[label] = test_probs
        
    if y_val_true is None:
        print("Error: No models evaluated.")
        return
        
    # Add Ensemble (CNN + GNN + Hybrid)
    print("\nEvaluating Soft Voting Ensemble (CNN + GNN + Hybrid)...")
    ensemble_val_probs = (val_probs_dict['CNN'] + val_probs_dict['GNN'] + val_probs_dict['Hybrid']) / 3.0
    ensemble_test_probs = (test_probs_dict['CNN'] + test_probs_dict['GNN'] + test_probs_dict['Hybrid']) / 3.0
    
    val_probs_dict['Ensemble'] = ensemble_val_probs
    test_probs_dict['Ensemble'] = ensemble_test_probs
    
    results = {}
    
    print("\nOptimizing Thresholds on Validation Set...")
    for model_name in val_probs_dict.keys():
        v_probs = val_probs_dict[model_name]
        t_probs = test_probs_dict[model_name]
        
        # Optimize on validation
        opt_thresh = optimize_thresholds(y_val_true, v_probs)
        
        # Evaluate on test with default threshold (0.5)
        eval_default = evaluate_predictions(y_test_true, t_probs, 0.5)
        
        # Evaluate on test with optimized thresholds
        eval_opt = evaluate_predictions(y_test_true, t_probs, opt_thresh)
        
        results[model_name] = {
            'thresholds': opt_thresh,
            'default': eval_default,
            'optimized': eval_opt
        }
        
        print(f"[{model_name}] Optimal Thresholds: " + ", ".join([f"{c}: {t:.2f}" for c, t in zip(CLASSES, opt_thresh)]))
        print(f"  Default   -> F1: {eval_default['f1']:.4f}, Exact Match: {eval_default['exact_match']*100:.2f}%, PhysioNet: {eval_default['physionet']:.4f}")
        print(f"  Optimized -> F1: {eval_opt['f1']:.4f}, Exact Match: {eval_opt['exact_match']*100:.2f}%, PhysioNet: {eval_opt['physionet']:.4f}")

    # Generate Markdown Table
    print("\n### Performance Comparison Table (Markdown)")
    print("| Model | Default Macro F1 | Optimized Macro F1 | Improvement | Default EM Acc | Optimized EM Acc | Improvement |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for name, res in results.items():
        f1_def = res['default']['f1']
        f1_opt = res['optimized']['f1']
        f1_diff = f1_opt - f1_def
        em_def = res['default']['exact_match'] * 100
        em_opt = res['optimized']['exact_match'] * 100
        em_diff = em_opt - em_def
        print(f"| **{name}** | {f1_def:.4f} | **{f1_opt:.4f}** | +{f1_diff:.4f} | {em_def:.2f}% | **{em_opt:.2f}%** | +{em_diff:.2f}% |")

    # Generate LaTeX code for thesis
    print("\n% ==================== LATEX CODE FOR THESIS ====================")
    print("\\begin{table}[H]")
    print("\\caption{Comparison of Default vs. Optimized Thresholds and Ensemble Performance}")
    print("\\label{tab:opt_thresholds}")
    print("\\begin{center}")
    print("\\resizebox{\\linewidth}{!}{%")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print("\\textbf{Model} & \\textbf{Thresholds} & \\textbf{Macro F1} & \\textbf{Macro AUROC} & \\textbf{Macro AUPRC} & \\textbf{Exact Match} & \\textbf{PhysioNet Score} \\\\")
    print("\\midrule")
    
    for name, res in results.items():
        # Default
        print(f"{name} & Default (0.5) & {res['default']['f1']:.4f} & {res['default']['auroc']:.4f} & {res['default']['auprc']:.4f} & {res['default']['exact_match']*100:.2f}\\% & {res['default']['physionet']:.4f} \\\\")
        
        # Optimized
        thresh_str = "[" + ", ".join([f"{t:.2f}" for t in res['thresholds']]) + "]"
        # Bold F1 and Exact Match if it is ensemble optimized or cnn optimized
        f1_str = f"\\textbf{{{res['optimized']['f1']:.4f}}}" if name in ['CNN', 'Ensemble'] else f"{res['optimized']['f1']:.4f}"
        em_str = f"\\textbf{{{res['optimized']['exact_match']*100:.2f}\\%}}" if name in ['GNN', 'Ensemble'] else f"{res['optimized']['exact_match']*100:.2f}\\%"
        
        print(f" & Optimized {thresh_str} & {f1_str} & {res['optimized']['auroc']:.4f} & {res['optimized']['auprc']:.4f} & {em_str} & {res['optimized']['physionet']:.4f} \\\\")
        print("\\midrule")
        
    print("\\bottomrule")
    print("\\end{tabular}%")
    print("}")
    print("\\end{center}")
    print("\\end{table}")
    print("% ===============================================================\n")

if __name__ == "__main__":
    main()
