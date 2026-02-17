import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from src.data_loader import PTBXLDataset

def verify_data():
    print("Verifying processed data...")
    dataset = PTBXLDataset()
    
    try:
        X, y, ids = dataset.load_data()
        print(f"Loaded Data Shape: {X.shape}")
        print(f"Loaded Labels Shape: {y.shape}")
        print(f"Computed IDs Shape: {ids.shape}")
        
        # Check normalization (mean ~ 0, std ~ 1 per lead)
        print("\nChecking normalization stats (first 1000 records):")
        print(f"Mean: {np.mean(X[:1000]):.4f}")
        print(f"Std: {np.std(X[:1000]):.4f}")
        
        # Test Splitting
        print("\nTesting Train/Val/Test Split...")
        splits = dataset.get_train_val_test_split(test_size=0.1, val_size=0.1)
        
        for key, val in splits.items():
            if 'X' in key or 'y' in key:
                 print(f"{key}: {val.shape}")
                 
        print("\nVerification Successful.")
        
    except Exception as e:
        print(f"Verification Failed: {e}")

if __name__ == "__main__":
    verify_data()
