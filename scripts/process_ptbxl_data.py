import sys
import os
import numpy as np
import wfdb
import pandas as pd
import gc

# Add project root to path
sys.path.append(os.getcwd())

from src.preprocessing import preprocess_signal
from src.utils import load_ptbxl_metadata, get_labels

def process_ptbxl():
    print("Starting PTB-XL processing...", flush=True)
    raw_data_dir = os.path.join('data', 'raw', 'ptb-xl')
    processed_dir = os.path.join('data', 'processed', 'ptb-xl')
    
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    print("Loading metadata...", flush=True)
    df = load_ptbxl_metadata(raw_data_dir)
    
    # Target 500 Hz
    fs = 500
    
    BATCH_SIZE = 1000
    total_records = len(df)
    print(f"Total records to process: {total_records}", flush=True)
    
    # Processing loop
    current_batch_X = []
    current_batch_ids = []
    
    # Check if we can resume? No, complex. Overwrite.
    # We will save chunks: X_ptbxl_part_0.npy, etc.
    
    batch_idx = 0
    count = 0
    
    for ecg_id, row in df.iterrows():
        try:
            filename = row['filename_hr']
            record_path = os.path.join(raw_data_dir, filename)
            
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal
            
            processed_signal = preprocess_signal(signal, record.fs, target_fs=500)
            
            if processed_signal.shape != (5000, 12):
                if processed_signal.shape[0] > 5000:
                    processed_signal = processed_signal[:5000, :]
                else:
                    pad_len = 5000 - processed_signal.shape[0]
                    processed_signal = np.pad(processed_signal, ((0, pad_len), (0,0)))
            
            current_batch_X.append(processed_signal.astype(np.float32))
            current_batch_ids.append(ecg_id)
            count += 1
            
            if len(current_batch_X) >= BATCH_SIZE:
                print(f"Saving batch {batch_idx} ({count}/{total_records})...", flush=True)
                X_batch = np.stack(current_batch_X)
                ids_batch = np.array(current_batch_ids)
                
                np.save(os.path.join(processed_dir, f'X_part_{batch_idx}.npy'), X_batch)
                np.save(os.path.join(processed_dir, f'ids_part_{batch_idx}.npy'), ids_batch)
                
                current_batch_X = []
                current_batch_ids = []
                batch_idx += 1
                gc.collect() # Free memory
                
        except Exception as e:
            print(f"Error processing {ecg_id}: {e}", flush=True)

    # Save last batch
    if current_batch_X:
        print(f"Saving final batch {batch_idx}...", flush=True)
        X_batch = np.stack(current_batch_X)
        ids_batch = np.array(current_batch_ids)
        np.save(os.path.join(processed_dir, f'X_part_{batch_idx}.npy'), X_batch)
        np.save(os.path.join(processed_dir, f'ids_part_{batch_idx}.npy'), ids_batch)
        batch_idx += 1
    
    print("Combining batches...", flush=True)
    # Load all parts and concatenate
    all_X = []
    all_ids = []
    
    for i in range(batch_idx):
        dataset_X = np.load(os.path.join(processed_dir, f'X_part_{i}.npy'))
        dataset_ids = np.load(os.path.join(processed_dir, f'ids_part_{i}.npy'))
        all_X.append(dataset_X)
        all_ids.append(dataset_ids)
        
        # Cleanup parts
        os.remove(os.path.join(processed_dir, f'X_part_{i}.npy'))
        os.remove(os.path.join(processed_dir, f'ids_part_{i}.npy'))

    X_final = np.concatenate(all_X)
    ids_final = np.concatenate(all_ids)
    
    print(f"Final Data Shape: {X_final.shape}", flush=True)
    
    # Generate labels
    print("Generating labels...", flush=True)
    df_processed = df.loc[ids_final]
    y_final, classes = get_labels(df_processed)
    
    print(f"Labels Shape: {y_final.shape}", flush=True)
    
    np.save(os.path.join(processed_dir, 'X_ptbxl.npy'), X_final)
    np.save(os.path.join(processed_dir, 'y_ptbxl.npy'), y_final)
    np.save(os.path.join(processed_dir, 'ids_ptbxl.npy'), ids_final)
    df_processed.to_csv(os.path.join(processed_dir, 'metadata.csv'))
    
    print("Processing complete.", flush=True)

if __name__ == "__main__":
    process_ptbxl()
