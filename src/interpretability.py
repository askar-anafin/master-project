import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook for gradients and activations
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple, take the first element
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
            
        # Zero grads
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Global Average Pooling of Gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2])
        
        # Weight activations by pooled gradients
        activations = self.activations.detach()
        for i in range(activations.shape[1]):
            activations[:, i, :] *= pooled_gradients[i]
            
        # Average the channels to get the heatmap
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # ReLU on heatmap
        heatmap = F.relu(heatmap)
        
        # Normalize
        heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().numpy()

def plot_gradcam(signal, heatmap, title="Grad-CAM"):
    """
    Plot the signal and the heatmap overlay.
    Signal shape: (Seq_Len, Channels) -> We'll plot average or a specific lead
    Heatmap shape: (Seq_Len_Reduced) -> Needs resizing
    """
    import cv2
    
    # Resize heatmap to match signal length
    heatmap = cv2.resize(heatmap.reshape(1, -1), (signal.shape[0], 1))
    heatmap = heatmap.reshape(-1)
    
    plt.figure(figsize=(12, 4))
    
    # Plot a representative lead (e.g., Lead I or II - index 0 or 1)
    # Or plot RMS of all leads
    rms_signal = np.sqrt(np.mean(signal**2, axis=1))
    
    # Normalize signal for visualization
    rms_signal = (rms_signal - np.min(rms_signal)) / (np.max(rms_signal) - np.min(rms_signal))
    
    plt.plot(rms_signal, label='ECG Signal (RMS)', alpha=0.6, color='black')
    plt.imshow(heatmap.reshape(1, -1), cmap='jet', aspect='auto', alpha=0.5, 
               extent=[0, len(rms_signal), 0, 1])
    
    plt.colorbar(label='Importance')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
