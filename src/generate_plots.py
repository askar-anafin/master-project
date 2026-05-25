import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.ptbxl.loader import PTBXLDataset
from src.train_unified import get_model

def generate_confusion_matrix():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data via Class to ensure correct split
    print("Loading Test Data...")
    from src.ptbxl.loader import PTBXLDataset
    dataset = PTBXLDataset()
    data = dataset.get_train_val_test_split(test_size=0.1, random_state=42)
    X_test = data['X'][data['test_idx']]
    y_test = data['y'][data['test_idx']]
    
    # Load Model (CNN)
    print("Loading CNN Model...")
    model = get_model('cnn')
    model.to(device)
    model.load_state_dict(torch.load('experiments/cnn_full/best_model.pth', map_location=device))
    model.eval()
    
    # Inference
    print("Running Inference...")
    batch_size = 32
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_x = torch.tensor(X_test[i:i+batch_size]).float().to(device)
            outputs = model(batch_x)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            
    y_pred_probs = np.concatenate(all_preds)
    
    # Convert Multi-label to Single-label (Dominant Class) for Visualization
    # Classes: ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    # Strategy: Take the class with Max Probability as the "Primary Prediction"
    # And the class with Max Truth as "Primary Label" (handling multi-label by priority or max)
    # For simplicity in this visual, we use argmax
    
    y_true_indices = np.argmax(y_test, axis=1)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    
    # Compute Matrix
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Normalized Confusion Matrix (CNN - ResNet18)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/confusion_matrix.png', dpi=300)
    plt.savefig('figures/confusion_matrix.eps', format='eps')
    plt.savefig('confusion_matrix.png', dpi=300)
    print("confusion_matrix.png and figures/confusion_matrix.eps saved!")

if __name__ == "__main__":
    generate_confusion_matrix()
