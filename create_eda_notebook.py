import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """\
# Exploratory Data Analysis (EDA)
This notebook explores the MIT-BIH Arrhythmia Database.
We will:
1. Visualizing raw ECG signals.
2. Analyzing the distribution of heartbeat classes (annotations).
"""

code_imports = """\
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wfdb
from collections import Counter

# Add src to path
sys.path.append(os.path.abspath('../src'))
from data_loader import load_record

%matplotlib inline
"""

code_load_sample = """\
# Load a sample record (e.g., '100')
record_id = '100'
record, annotation = load_record(record_id, data_dir='../data/raw')

print(f"Record Name: {record.record_name}")
print(f"Sampling Frequency: {record.fs} Hz")
print(f"Signal Length: {record.sig_len} samples")
print(f"Number of Annotations: {len(annotation.sample)}")
"""

code_plot_signal = """\
# Plot a 10-second segment
fs = record.fs
start_sec = 0
end_sec = 10
start_sample = start_sec * fs
end_sample = end_sec * fs

signal = record.p_signal[start_sample:end_sample, 0] # Lead I
time = np.arange(start_sample, end_sample) / fs

# Filter annotations within this range
ann_indices = [i for i, x in enumerate(annotation.sample) if start_sample <= x < end_sample]
ann_samples = [annotation.sample[i] for i in ann_indices]
ann_symbols = [annotation.symbol[i] for i in ann_indices]

plt.figure(figsize=(15, 5))
plt.plot(time, signal, label='ECG Signal (Lead I)')
plt.scatter([s/fs for s in ann_samples], [signal[s-start_sample] for s in ann_samples], c='red', marker='x', label='Annotations')

for s, sym in zip(ann_samples, ann_symbols):
    plt.annotate(sym, (s/fs, signal[s-start_sample]), xytext=(0, 10), textcoords='offset points', ha='center', color='red')

plt.title(f"ECG Signal Segment - Record {record_id}")
plt.xlabel("Time (s)")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.grid(True)
plt.show()
"""

code_class_dist = """\
# Analyze Class Distribution
data_dir = '../data/raw'
# Get all record IDs (filenames ending in .dat, stripped of extension)
record_ids = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.dat')]
record_ids = sorted(list(set(record_ids)))

all_symbols = []

print(f"Analyzing {len(record_ids)} records...")

for rid in record_ids:
    try:
        ann = wfdb.rdann(os.path.join(data_dir, rid), 'atr')
        all_symbols.extend(ann.symbol)
    except Exception as e:
        print(f"Error reading {rid}: {e}")

counts = Counter(all_symbols)
print("Annotation Counts:")
for sym, count in counts.most_common():
    print(f"{sym}: {count}")
"""

code_plot_dist = """\
# Visualize Distribution
labels, values = zip(*counts.most_common())

plt.figure(figsize=(12, 6))
sns.barplot(x=list(labels), y=list(values), palette='viridis')
plt.title("Distribution of Heartbeat Types in MIT-BIH Database")
plt.xlabel("Annotation Symbol")
plt.ylabel("Count")
plt.yscale('log') # Use log scale because 'N' (Normal) dominates
plt.show()

print("Note: 'N' is Normal beat. Other symbols represent various arrhythmias and artifacts.")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_load_sample),
    nbf.v4.new_code_cell(code_plot_signal),
    nbf.v4.new_code_cell(code_class_dist),
    nbf.v4.new_code_cell(code_plot_dist)
]

with open('notebooks/01_eda.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully.")
