import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.models.hybrid import ResNetTransformer

def test_hybrid():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing hybrid model on device: {device}")
    
    try:
        model = ResNetTransformer(num_classes=5).to(device)
        print("ResNetTransformer initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize ResNetTransformer: {e}")
        import traceback
        traceback.print_exc()
        return

    # Input shape: (Batch, Seq, Channels) = (8, 5000, 12)
    dummy_input = torch.randn(8, 5000, 12).to(device)
    
    try:
        output = model(dummy_input)
        print(f"Forward pass successful. Output shape: {output.shape}")
        
        if output.shape == (8, 5):
            print("Output shape correct.")
        else:
            print("Output shape mismatched.")
            
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hybrid()
