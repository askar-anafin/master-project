import os
import numpy as np
from sklearn.model_selection import train_test_split

class PTBXLDataset:
    def __init__(self, data_dir='data/processed/ptb-xl'):
        self.data_dir = data_dir
        self.X_path = os.path.join(data_dir, 'X_ptbxl.npy')
        self.y_path = os.path.join(data_dir, 'y_ptbxl.npy')
        self.ids_path = os.path.join(data_dir, 'ids_ptbxl.npy')
        
    def load_data(self):
        if not os.path.exists(self.X_path):
            raise FileNotFoundError(f"Processed data not found at {self.data_dir}. Run scripts/process_ptbxl_data.py first.")
            
        # Use mmap_mode='r' to avoid loading everything into RAM
        X = np.load(self.X_path, mmap_mode='r')
        y = np.load(self.y_path)
        ids = np.load(self.ids_path)
        
        return X, y, ids
    
    def get_train_val_test_split(self, test_size=0.1, val_size=0.1, random_state=42, limit=None, **kwargs):
        X, y, ids = self.load_data()
        
        all_indices = np.arange(len(y))
        
        if limit:
             all_indices = all_indices[:limit]
             y = y[:limit]
        
        try:
            # First split off Test set using INDICES
            train_val_idx, test_idx = train_test_split(
                all_indices, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError as e:
            print(f"Stratified split failed: {e}. Falling back to random split.")
            train_val_idx, test_idx = train_test_split(
                all_indices, test_size=test_size, random_state=random_state, stratify=None
            )
        
        # Validation split
        val_prop = val_size / (1.0 - test_size)
        y_train_val = y[train_val_idx]
        
        try:
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=val_prop, random_state=random_state, stratify=y_train_val
            )
        except ValueError:
             train_idx, val_idx = train_test_split(
                train_val_idx, test_size=val_prop, random_state=random_state, stratify=None
            )
        
        return {
            'X': X, 'y': y,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }
