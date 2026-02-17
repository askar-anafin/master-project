import sys
import os
import numpy as np
import wfdb
from scipy.signal import resample

# Add project root to path
sys.path.append(os.getcwd())

# AAMI Class Mapping
AAMI_CLASSES = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}
CLASS_MAP = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}

def process_mitbih(download=True):
    print("Starting MIT-BIH processing...", flush=True)
    raw_dir = os.path.join('data', 'raw', 'mit-bih')
    processed_dir = os.path.join('data', 'processed', 'mit-bih')
    
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
        
    if download:
        try:
            print("Checking/Downloading MIT-BIH data...", flush=True)
            wfdb.dl_database('mitdb', raw_dir)
        except Exception as e:
            print(f"Error downloading: {e}. Assuming data exists.", flush=True)

    # MIT-BIH records (excluding problematic ones if any, usually 102, 104, 107, 217 have paced beats)
    # Getting list of records
    records = [f.split('.')[0] for f in os.listdir(raw_dir) if f.endswith('.dat')]
    records = sorted(list(set(records)))
    print(f"Found {len(records)} records.", flush=True)
    
    X_list = []
    y_list = []
    ids_list = []
    
    # 10 seconds window at 360Hz = 3600 samples
    WINDOW_SIZE = 3600
    TARGET_FS = 500
    TARGET_SIZE = 5000
    STRIDE = 3600 # Non-overlapping
    
    for rec_name in records:
        try:
            path = os.path.join(raw_dir, rec_name)
            record = wfdb.rdrecord(path)
            annotation = wfdb.rdann(path, 'atr')
            
            signal = record.p_signal # (650000, 2)
            original_fs = record.fs
            
            # Iterate through signal with windows
            num_samples = signal.shape[0]
            
            for start in range(0, num_samples - WINDOW_SIZE, STRIDE):
                end = start + WINDOW_SIZE
                segment = signal[start:end, :] # (3600, 2)
                
                # Resample
                resampled_segment = resample(segment, TARGET_SIZE) # (5000, 2)
                
                # Get labels in this window
                # Annotation sample indices
                ann_indices = annotation.sample
                ann_symbols = annotation.symbol
                
                # Check which annotations fall in this window
                mask = (ann_indices >= start) & (ann_indices < end)
                window_symbols = np.array(ann_symbols)[mask]
                
                # Labeling
                # Start with all zeros
                label = np.zeros(5, dtype=np.float32)
                
                has_valid_beat = False
                for sym in window_symbols:
                    if sym in AAMI_CLASSES:
                        aami_class = AAMI_CLASSES[sym]
                        class_idx = CLASS_MAP[aami_class]
                        label[class_idx] = 1.0
                        has_valid_beat = True
                        
                # Only add if there's at least one annotated beat (some segments might be noise/unannotated)
                if has_valid_beat:
                    X_list.append(resampled_segment.astype(np.float32))
                    y_list.append(label)
                    ids_list.append(f"{rec_name}_{start}")
                    
        except Exception as e:
            print(f"Error processing {rec_name}: {e}", flush=True)
            
    # Convert to numpy
    X_final = np.stack(X_list) # (N, 5000, 2)
    y_final = np.stack(y_list) # (N, 5)
    ids_final = np.array(ids_list)
    
    print(f"Processed MIT-BIH Data Shape: {X_final.shape}", flush=True)
    print(f"Labels Shape: {y_final.shape}", flush=True)
    
    np.save(os.path.join(processed_dir, 'X_mitbih.npy'), X_final)
    np.save(os.path.join(processed_dir, 'y_mitbih.npy'), y_final)
    np.save(os.path.join(processed_dir, 'ids_mitbih.npy'), ids_final)
    
    print("MIT-BIH Processing complete.", flush=True)

if __name__ == "__main__":
    process_mitbih()
