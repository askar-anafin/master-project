import torch
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, accuracy_score

# Add project root to path
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
            # Use CPU or CUDA autocast based on device
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(X_batch)
            else:
                outputs = model(X_batch)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_truths.append(y_batch.numpy())
    return np.vstack(all_truths), np.vstack(all_probs)

def evaluate_model(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs > threshold).astype(int)
    
    # Macro metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    macro_auroc = roc_auc_score(y_true, y_probs, average='macro')
    macro_auprc = average_precision_score(y_true, y_probs, average='macro')
    exact_match = accuracy_score(y_true, y_pred)
    physionet = compute_physionet_score(y_true, y_probs)
    
    # Class-wise metrics
    class_metrics = {}
    for i, class_name in enumerate(CLASSES):
        class_f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        class_prec = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        class_rec = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        try:
            class_auroc = roc_auc_score(y_true[:, i], y_probs[:, i])
        except ValueError:
            class_auroc = 0.0
        class_auprc = average_precision_score(y_true[:, i], y_probs[:, i])
        
        class_metrics[class_name] = {
            'f1': class_f1,
            'precision': class_prec,
            'recall': class_rec,
            'auroc': class_auroc,
            'auprc': class_auprc
        }
        
    return {
        'macro': {
            'f1': macro_f1,
            'auroc': macro_auroc,
            'auprc': macro_auprc,
            'exact_match': exact_match,
            'physionet': physionet
        },
        'class_wise': class_metrics
    }

def generate_table_3_2(results):
    metrics = [
        ('Macro F1-Score', 'f1', False),
        ('Macro AUROC', 'auroc', False),
        ('Macro AUPRC', 'auprc', False),
        ('Exact Match Acc', 'exact_match', True),
        ('PhysioNet Score', 'physionet', False)
    ]
    
    print("\n% ==================== TABLE 3.2 LATEX CODE ====================")
    print("\\begin{table}[H]")
    print("\\caption{Quantitative Benchmark Results on PTB-XL Test Set}")
    print("\\begin{center}")
    print("\\resizebox{\\linewidth}{!}{%")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\textbf{Metric} & \\textbf{CNN (1D ResNet-18)} & \\textbf{GNN (ST-ReGE)} & \\textbf{ViT (1D Transformer)} & \\textbf{Hybrid (ResNet-Transformer)} \\\\")
    print("\\midrule")
    
    model_keys = ['CNN', 'GNN', 'ViT', 'Hybrid']
    
    for metric_label, key, is_percentage in metrics:
        vals = []
        for m in model_keys:
            val = results[m]['macro'][key]
            vals.append(val)
        
        max_idx = np.argmax(vals)
        
        formatted_vals = []
        for idx, val in enumerate(vals):
            if is_percentage:
                s = f"{val*100:.2f}\\%"
            else:
                s = f"{val:.4f}"
            if idx == max_idx:
                s = f"\\textbf{{{s}}}"
            formatted_vals.append(s)
            
        print(f"{metric_label} & {formatted_vals[0]} & {formatted_vals[1]} & {formatted_vals[2]} & {formatted_vals[3]} \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}%")
    print("}")
    print("\\label{tab:bench_results}")
    print("\\end{center}")
    print("\\end{table}")
    print("% ===============================================================\n")

def generate_table_3_3(results):
    metrics_keys = ['f1', 'precision', 'recall', 'auroc', 'auprc']
    model_keys = ['CNN', 'GNN', 'ViT', 'Hybrid']
    
    print("\n% ==================== TABLE 3.3 LATEX CODE ====================")
    print("\\begin{table}[H]")
    print("\\caption{Class-wise Performance Metrics on PTB-XL Test Partition}")
    print("\\label{tab:classwise_results}")
    print("\\begin{center}")
    print("\\resizebox{\\linewidth}{!}{%")
    print("\\begin{tabular}{llccccc}")
    print("\\toprule")
    print("\\textbf{Class} & \\textbf{Model} & \\textbf{F1-Score} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{AUROC} & \\textbf{AUPRC} \\\\")
    print("\\midrule")
    
    for c_idx, class_name in enumerate(CLASSES):
        max_vals = {}
        for m_key in metrics_keys:
            vals = [results[model]['class_wise'][class_name][m_key] for model in model_keys]
            max_vals[m_key] = max(vals)
            
        for idx, model in enumerate(model_keys):
            model_class_metrics = results[model]['class_wise'][class_name]
            
            formatted_metrics = []
            for m_key in metrics_keys:
                val = model_class_metrics[m_key]
                s = f"{val:.3f}"
                if abs(val - max_vals[m_key]) < 1e-9:
                    s = f"\\textbf{{{s}}}"
                formatted_metrics.append(s)
                
            class_str = f"\\textbf{{{class_name}}}" if idx == 0 else "              "
            model_display_name = model
            print(f"{class_str} & {model_display_name} & {formatted_metrics[0]} & {formatted_metrics[1]} & {formatted_metrics[2]} & {formatted_metrics[3]} & {formatted_metrics[4]} \\\\")
            
        if c_idx < len(CLASSES) - 1:
            print("\\midrule")
            
    print("\\bottomrule")
    print("\\end{tabular}%")
    print("}")
    print("\\end{center}")
    print("\\end{table}")
    print("% ===============================================================\n")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load test data
    print("Loading test split...")
    dataset = PTBXLDataset()
    splits = dataset.get_train_val_test_split()
    test_data = ECGDataset(splits['X'], splits['y'], indices=splits['test_idx'], transform=None)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    models_info = {
        'CNN': ('cnn', 'experiments/cnn_filtered/best_model.pth'),
        'GNN': ('gnn', 'experiments/gnn_filtered/best_model.pth'),
        'ViT': ('vit', 'experiments/vit_filtered/best_model.pth'),
        'Hybrid': ('hybrid', 'experiments/hybrid_filtered/best_model.pth')
    }
    
    results = {}
    
    for model_label, (model_type, model_path) in models_info.items():
        if not os.path.exists(model_path):
            print(f"Error: checkpoint not found for {model_label} at {model_path}")
            continue
            
        print(f"Loading {model_label} model...")
        model = get_model(model_type).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        print(f"Evaluating {model_label}...")
        y_true, y_probs = predict(model, test_loader, device)
        
        results[model_label] = evaluate_model(y_true, y_probs)
        
    if len(results) < len(models_info):
        print("Warning: not all models were successfully evaluated.")
        
    generate_table_3_2(results)
    generate_table_3_3(results)

if __name__ == "__main__":
    main()
