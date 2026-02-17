import torch
from torch.utils.data import Dataset
import numpy as np

class ECGDataset(Dataset):
    def __init__(self, X, y, indices=None, transform=None):
        """
        Generic ECG Dataset wrapper.
        Args:
            X (np.array or mmap): ECG signals (N, Seq, Channels)
            y (np.array): Labels (N, Classes)
            indices (np.array): Indices to use (for train/val/test splits)
            transform (callable): Augmentation function
        """
        self.X = X
        self.y = y
        self.indices = indices if indices is not None else np.arange(len(X))
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = self.X[real_idx].copy() # Copy to avoid read-only error with mmap
        y = self.y[real_idx]
        
        # Handle potential NaNs
        if np.isnan(x).any():
             x = np.nan_to_num(x, nan=0.0)
        
        if self.transform:
            x = self.transform(x)
            
        # Convert to tensor
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        return x, y
