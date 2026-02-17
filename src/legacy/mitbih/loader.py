import os
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

class MITBIHDataset:
    def __init__(self, data_dir='data/processed/mit-bih'):
        self.data_dir = data_dir
        self.X_path = os.path.join(data_dir, 'X_mitbih.npy')
        self.y_path = os.path.join(data_dir, 'y_mitbih.npy')
        self.ids_path = os.path.join(data_dir, 'ids_mitbih.npy')
        
    def load_data(self):
        if not os.path.exists(self.X_path):
            raise FileNotFoundError(f"Processed data not found at {self.data_dir}. Run src/mitbih/process.py first.")
            
        # Use mmap_mode='r'
        X = np.load(self.X_path, mmap_mode='r')
        y = np.load(self.y_path)
        ids = np.load(self.ids_path)
        
        return X, y, ids
    
    def get_train_val_test_split(self, test_size=0.2, val_size=0.1, random_state=42, limit=None, split_strategy='inter_patient'):
        X, y, ids = self.load_data()
        
        # Extract Patient Groups (Record names)
        groups = np.array([i.split('_')[0] for i in ids])
        
        if limit:
             # Shuffle to get mixed groups if limiting
             indices = np.arange(len(X))
             np.random.shuffle(indices)
             indices = indices[:limit]
             
             X = X[indices]
             y = y[indices]
             ids = ids[indices]
             groups = groups[indices]
        
        if split_strategy == 'inter_patient':
            # Split by Groups (Strict patient separation)
            print("Using Inter-Patient Split (Strict)")
            gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_val_idx, test_idx = next(gss_test.split(X, y, groups))
            
            # Validation Split
            val_prop = val_size / (1.0 - test_size)
            y_train_val = y[train_val_idx]
            groups_train_val = groups[train_val_idx]
            
            gss_val = GroupShuffleSplit(n_splits=1, test_size=val_prop, random_state=random_state)
            # Pass y_train_val as X to ensure consistent lengths
            train_sub_idx, val_sub_idx = next(gss_val.split(y_train_val, y_train_val, groups_train_val))
            
            # Map sub-indices back to original indices
            train_idx = train_val_idx[train_sub_idx]
            val_idx = train_val_idx[val_sub_idx]
            
        elif split_strategy == 'intra_patient':
            # Split randomly (Mixing patients)
            print("Using Intra-Patient Split (Mixed)")
            from sklearn.model_selection import train_test_split
            all_indices = np.arange(len(X))
            
            # Stratify by label if possible, else random
            try:
                train_val_idx, test_idx = train_test_split(all_indices, test_size=test_size, random_state=random_state, stratify=y)
            except ValueError:
                train_val_idx, test_idx = train_test_split(all_indices, test_size=test_size, random_state=random_state, stratify=None)
                
            val_prop = val_size / (1.0 - test_size)
            y_train_val = y[train_val_idx]
            
            try:
                train_idx, val_idx = train_test_split(train_val_idx, test_size=val_prop, random_state=random_state, stratify=y_train_val)
            except ValueError:
                train_idx, val_idx = train_test_split(train_val_idx, test_size=val_prop, random_state=random_state, stratify=None)
                
        else:
            raise ValueError(f"Unknown split strategy: {split_strategy}")
        
        return {
            'X': X, 'y': y,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }
