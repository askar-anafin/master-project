import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.getcwd())

from src.utils import load_ptbxl_metadata, get_labels

def combine_batches():
    processed_dir = os.path.join('data', 'processed', 'ptb-xl')
    raw_data_dir = os.path.join('data', 'raw', 'ptb-xl')
    
    print("Combining available batches...")
    all_X = []
    all_ids = []
    
    # Find all X_part_*.npy files
    files = os.listdir(processed_dir)
    batch_files = [f for f in files if f.startswith('X_part_') and f.endswith('.npy')]
    
    # Sort by batch index
    batch_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    
    if not batch_files:
        print("No batch files found.")
        return

    print(f"Found {len(batch_files)} batches.")
    
    for f in batch_files:
        batch_idx = f.split('_')[2].split('.')[0]
        ids_file = f'ids_part_{batch_idx}.npy'
        
        if ids_file not in files:
            print(f"Skipping batch {batch_idx}: Missing ids file.")
            continue
            
        print(f"Loading batch {batch_idx}...", flush=True)
        dataset_X = np.load(os.path.join(processed_dir, f))
        dataset_ids = np.load(os.path.join(processed_dir, ids_file))
        
        all_X.append(dataset_X)
        all_ids.append(dataset_ids)

    if not all_X:
        print("No valid batches loaded.")
        return

    X_final = np.concatenate(all_X)
    ids_final = np.concatenate(all_ids)
    
    print(f"Combined Data Shape: {X_final.shape}")
    
    # Generate labels
    print("Loading metadata to generate labels...", flush=True)
    df = load_ptbxl_metadata(raw_data_dir)
    
    # Filter metadata to matched IDs
    # Note: ids_final contains the index (ecg_id)
    # Check if all ids are in df
    valid_ids = [i for i in ids_final if i in df.index]
    if len(valid_ids) != len(ids_final):
        print("Warning: Some processed IDs not found in metadata.")
    
    df_processed = df.loc[ids_final]
    y_final, classes = get_labels(df_processed)
    
    print(f"Labels Shape: {y_final.shape}")
    print(f"Classes: {classes}")
    
    print(f"Saving combined dataset to {processed_dir}...")
    np.save(os.path.join(processed_dir, 'X_ptbxl.npy'), X_final)
    np.save(os.path.join(processed_dir, 'y_ptbxl.npy'), y_final)
    np.save(os.path.join(processed_dir, 'ids_ptbxl.npy'), ids_final)
    df_processed.to_csv(os.path.join(processed_dir, 'metadata.csv'))
    
    print("Combination complete.")

if __name__ == "__main__":
    combine_batches()
