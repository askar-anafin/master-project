import numpy as np
import os
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
            
        X = np.load(self.X_path)
        y = np.load(self.y_path)
        ids = np.load(self.ids_path)
        return X, y, ids
    
    def get_train_val_test_split(self, test_size=0.1, val_size=0.1, random_state=42):
        """
        Split data into train, val, test sets.
        Strategy: Split (Train+Val) and Test -> Then split Train and Val.
        
        Args:
            test_size (float): Proportion of dataset to include in test split.
            val_size (float): Proportion of dataset to include in validation split (relative to total).
            random_state (int): Random seed.
            
        Returns:
            dict: Dictionary with keys 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'
        """
        X, y, ids = self.load_data()
        
        try:
            # First split off Test set
            X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
                X, y, ids, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError as e:
            print(f"Stratified split failed (likely rare unique labels): {e}. Falling back to random split.")
            X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
                X, y, ids, test_size=test_size, random_state=random_state, stratify=None
            )
        
        # Calculate validation size relative to the remaining (Train+Val) set
        # We want val_size of total. So val_prop = val_size / (1 - test_size)
        val_prop = val_size / (1.0 - test_size)
        
        try:
            X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
                X_temp, y_temp, ids_temp, test_size=val_prop, random_state=random_state, stratify=y_temp
            )
        except ValueError:
             X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
                X_temp, y_temp, ids_temp, test_size=val_prop, random_state=random_state, stratify=None
            )
        
        return {
            'X_train': X_train, 'y_train': y_train, 'ids_train': ids_train,
            'X_val': X_val, 'y_val': y_val, 'ids_val': ids_val,
            'X_test': X_test, 'y_test': y_test, 'ids_test': ids_test
        }
