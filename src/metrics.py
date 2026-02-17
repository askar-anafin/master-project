
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, accuracy_score
import pandas as pd

def calculate_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Calculate generic performance metrics for multi-label classification.
    
    Args:
        y_true (np.array): Ground truth labels (N, Num_Classes) - Binary
        y_pred_probs (np.array): Predicted probabilities (N, Num_Classes)
        threshold (float): Threshold for binary classification
        
    Returns:
        dict: Dictionary containing calculated metrics
    """
    y_pred_binary = (y_pred_probs > threshold).astype(int)
    
    # 1. Macro AUPRC
    auprc_macro = average_precision_score(y_true, y_pred_probs, average='macro')
    
    # 2. Macro F1
    f1_macro = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
    
    # 3. AUROC
    try:
        auroc_macro = roc_auc_score(y_true, y_pred_probs, average='macro')
    except ValueError:
        auroc_macro = 0.0 # Handle edge case where a class has no positive samples
        
    # 4. Exact Match Accuracy
    exact_match = accuracy_score(y_true, y_pred_binary)
    
    return {
        'Macro_AUPRC': auprc_macro,
        'Macro_F1': f1_macro,
        'Macro_AUROC': auroc_macro,
        'Exact_Match_Acc': exact_match
    }

def compute_physionet_score(y_true, y_pred_probs, classes=None):
    """
    Computes a simplified PhysioNet/CinC Challenge Score.
    
    The official metric uses a specific weight matrix W based on clinical relevance. 
    Here, we approximate it by assigning higher weights to critical misclassifications.
    
    Weights (Row=True, Col=Pred):
    - Diagonal (Correct): 1.0
    - Missed Critical (e.g. MI predicted as Normal): -1.0 penalty
    - False Alarm (Normal predicted as Disease): -0.1 penalty
    """
    y_pred_binary = (y_pred_probs > 0.5).astype(int)
    
    score = 0.0
    total_samples = len(y_true)
    
    # Simple Heuristic if classes are ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    # Indices: 0:NORM, 1:MI, 2:STTC, 3:CD, 4:HYP
    
    w_matrix = np.eye(5) # Base identity (1.0 for correct)
    
    # Custom Penalties (Simplified)
    # Missed MI is bad
    w_matrix[1, 0] = -0.5 
    # Missed CD is bad
    w_matrix[3, 0] = -0.2
    
    # Calculate score
    for i in range(total_samples):
        # Multi-label case makes this tricky as multiple bits can be set.
        # We simplify by taking the max probability class as the "primary" diagnosis for scoring
        true_idx = np.argmax(y_true[i])
        pred_idx = np.argmax(y_pred_probs[i])
        
        score += w_matrix[true_idx, pred_idx]
        
    return score / total_samples

def save_metrics(metrics, classes, model_name, output_path):
    """
    Save metrics to CSV.
    """
    df = pd.DataFrame([metrics])
    df['Model'] = model_name
    
    # Reorder to put Model first
    cols = ['Model'] + [c for c in df.columns if c != 'Model']
    df = df[cols]
    
    if not output_path.endswith('.csv'):
        output_path = output_path + '/metrics.csv'
        
    # Append if exists
    include_header = not pd.io.common.file_exists(output_path)
    df.to_csv(output_path, mode='a', header=include_header, index=False)
