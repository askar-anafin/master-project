import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.models import BaselineCNN1D

def test_arch():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    model = BaselineCNN1D(num_classes=5).to(device)
    print("Model initialized.")
    
    # Input shape: (Batch, Seq, Channels) = (32, 5000, 12)
    dummy_input = torch.randn(32, 5000, 12).to(device)
    
    try:
        output = model(dummy_input)
        print(f"Forward pass successful. Output shape: {output.shape}")
        # Expected: (32, 5)
        
        if output.shape == (32, 5):
            print("Output shape correct.")
        else:
            print("Output shape mismatched.")
            
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    test_arch()
