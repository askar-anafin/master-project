"""
Filter unlabeled records from processed PTB-XL dataset.

Records with no diagnostic superclass label (all-zeros in y) are removed.
These are records whose SCP codes are exclusively rhythm/form annotations,
not mapping to any of: NORM, MI, STTC, CD, HYP.

This matches the methodology of the PTB-XL benchmark paper (Strodthoff et al., 2021).
"""
import numpy as np
import os
import sys

processed_dir = os.path.join('data', 'processed', 'ptb-xl')

X_path   = os.path.join(processed_dir, 'X_ptbxl.npy')
y_path   = os.path.join(processed_dir, 'y_ptbxl.npy')
ids_path = os.path.join(processed_dir, 'ids_ptbxl.npy')
meta_path = os.path.join(processed_dir, 'metadata.csv')

print('Loading y and ids...', flush=True)
y   = np.load(y_path)
ids = np.load(ids_path)

print(f'Total records before filtering: {len(y)}')

labeled_mask = y.sum(axis=1) > 0
n_removed = int((~labeled_mask).sum())
idx_list = np.where(labeled_mask)[0]

print(f'Unlabeled records to remove: {n_removed}')
print(f'Labeled records to keep:     {len(idx_list)}')

# Filter y and ids (small, in memory)
y_filtered   = y[labeled_mask]
ids_filtered = ids[labeled_mask]

# Filter X: load with mmap, copy only needed rows, then close before overwriting
print('Filtering X (loading mmap, copying rows)...', flush=True)
X_mmap = np.load(X_path, mmap_mode='r')
X_filtered = X_mmap[idx_list].copy()   # copy closes the mmap dependency on disk
del X_mmap  # release the mmap handle before writing

print(f'X_filtered shape: {X_filtered.shape}')

# Save temp files first, then rename to avoid partial-write corruption
# np.save always appends .npy, so use stems without .npy extension for temps
tmp_X_stem   = os.path.join(processed_dir, 'X_ptbxl_tmp')
tmp_y_stem   = os.path.join(processed_dir, 'y_ptbxl_tmp')
tmp_ids_stem = os.path.join(processed_dir, 'ids_ptbxl_tmp')

print('Saving filtered arrays...', flush=True)
np.save(tmp_X_stem,   X_filtered.astype(np.float32))  # writes X_ptbxl_tmp.npy
np.save(tmp_y_stem,   y_filtered)                     # writes y_ptbxl_tmp.npy
np.save(tmp_ids_stem, ids_filtered)                   # writes ids_ptbxl_tmp.npy

# Atomic rename
os.replace(tmp_X_stem   + '.npy', X_path)
os.replace(tmp_y_stem   + '.npy', y_path)
os.replace(tmp_ids_stem + '.npy', ids_path)

# Also filter metadata.csv if present
if os.path.exists(meta_path):
    import pandas as pd
    df = pd.read_csv(meta_path, index_col=0)
    df_filtered = df.loc[ids_filtered]
    df_filtered.to_csv(meta_path)
    print(f'metadata.csv updated: {len(df_filtered)} rows')

# Verification
print('\nVerifying saved files...')
y_check   = np.load(y_path)
X_check   = np.load(X_path, mmap_mode='r')
ids_check = np.load(ids_path)

unlabeled_remaining = int((y_check.sum(axis=1) == 0).sum())
assert unlabeled_remaining == 0, f"Still {unlabeled_remaining} unlabeled records!"
assert len(y_check) == len(X_check) == len(ids_check), "Shape mismatch!"

print(f'Final dataset size: {len(y_check)} records')
print(f'X shape:            {X_check.shape}')
print(f'y shape:            {y_check.shape}')
print(f'Unlabeled records:  {unlabeled_remaining}')
print('\nDone. All records now have at least one diagnostic superclass label.')
