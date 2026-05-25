import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.getcwd())

from src.models.cnn import ResNet18
from src.ptbxl.loader import PTBXLDataset
from src.interpretability import GradCAM

def visualize_sample():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # Load Data
    print("Loading PTB-XL Dataset...")
    dataset = PTBXLDataset()
    splits = dataset.get_train_val_test_split()
    X_test = splits['X'][splits['test_idx']]
    y_test = splits['y'][splits['test_idx']]
    
    # Select a sample (e.g., a True Positive MI case)
    # MI is index 1
    mi_indices = np.where(y_test[:, 1] == 1)[0]
    if len(mi_indices) == 0:
        print("No MI cases found in test set.")
        return
        
    idx = mi_indices[0]
    signal = X_test[idx] # (5000, 12)
    label = y_test[idx]
    
    # Prepare input: shape (1, 5000, 12) to match (batch, seq_len, channels)
    input_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # Load Model
    print("Loading CNN Model...")
    model = ResNet18(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load('experiments/cnn_full/best_model.pth', map_location=DEVICE))
    model.eval()
    
    # Initialize GradCAM
    # Target the last convolutional layer of the last ResBlock
    # ResNet18 structure: layer4 -> [1] (2nd block) -> conv2
    target_layer = model.layer4[1].conv2
    
    print("Generating Grad-CAM Heatmap...")
    gradcam = GradCAM(model, target_layer)
    
    # Generate Heatmap for MI class (index 1)
    heatmap = gradcam(input_tensor, class_idx=1)
    
    # Plot
    plt.figure(figsize=(15, 6))
    
    # Plot Lead I (Index 0) and Lead II (Index 1)
    lead_idx = 0
    clean_signal = signal[:, lead_idx]
    
    # Resize heatmap to signal length (5000 points) using 1D linear interpolation
    x_original = np.linspace(0, 1, len(heatmap))
    x_new = np.linspace(0, 1, signal.shape[0])
    heatmap_resized = np.interp(x_new, x_original, heatmap)
    
    # Normalize signal for plotting
    norm_signal = (clean_signal - np.min(clean_signal)) / (np.max(clean_signal) - np.min(clean_signal))
    
    plt.plot(norm_signal, label=f'Lead I Signal', color='black', linewidth=0.8)
    plt.imshow(heatmap_resized.reshape(1, -1), cmap='jet', aspect='auto', alpha=0.5, 
               extent=[0, len(norm_signal), 0, 1])
    
    plt.title(f"Grad-CAM Heatmap for Myocardial Infarction Prediction (Sample {idx})")
    plt.xlabel("Time (samples)")
    plt.ylabel("Normalized Amplitude")
    plt.colorbar(label="Importance")
    plt.legend()
    
    output_path = 'gradcam_mi.png'
    plt.savefig(output_path)
    
    # Create figures directory and save vector EPS / PNG copy
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/gradcam_mi.png', dpi=300)
    plt.savefig('figures/gradcam_mi.eps', format='eps')
    print(f"Saved Grad-CAM visualization to {output_path} and figures/gradcam_mi.eps")

if __name__ == "__main__":
    visualize_sample()
