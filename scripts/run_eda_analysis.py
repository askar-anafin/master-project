import wfdb
import os
import matplotlib.pyplot as plt
import numpy as np

def analyze_mit_bih():
    print("\n--- Analysing MIT-BIH Arrhythmia Database ---")
    data_path = os.path.join('data', 'raw', 'mit-bih')
    record_name = '100'
    
    try:
        record_path = os.path.join(data_path, record_name)
        if not os.path.exists(record_path + '.hea'):
             print(f"Record {record_name} not found in {data_path}")
             return

        record = wfdb.rdrecord(record_path)
        print(f"Record: {record.record_name}")
        print(f"Sampling Rate: {record.fs} Hz")
        print(f"Signal Shape: {record.p_signal.shape}")
        print(f"Leads: {record.sig_name}")
        
        # Plot
        plt.figure(figsize=(15, 5))
        plt.plot(record.p_signal[:1000, 0])
        plt.title(f"MIT-BIH Record {record_name} - Lead {record.sig_name[0]}")
        plt.savefig('notebooks/figures/mit_bih_sample.png')
        print("Saved figure: notebooks/figures/mit_bih_sample.png")
        plt.close()
        
    except Exception as e:
        print(f"Error analyzing MIT-BIH: {e}")

def analyze_chapman():
    print("\n--- Analysing Chapman-Shaoxing Database ---")
    data_path = os.path.join('data', 'raw', 'chapman-shaoxing')
    
    # Find a record
    sample_path = None
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.hea'):
                 sample_path = os.path.join(root, os.path.splitext(file)[0])
                 break
        if sample_path: break
    
    if not sample_path:
        print("No .hea files found in Chapman directory")
        return

    try:
        record = wfdb.rdrecord(sample_path)
        print(f"Record: {record.record_name}")
        print(f"Sampling Rate: {record.fs} Hz")
        print(f"Signal Shape: {record.p_signal.shape}")
        print(f"Leads: {record.sig_name}")
        
        # Plot 12 leads (first 5000 samples)
        plt.figure(figsize=(15, 10))
        for i in range(min(12, record.p_signal.shape[1])):
            plt.subplot(6, 2, i+1)
            plt.plot(record.p_signal[:1000, i])
            plt.title(f"Lead {record.sig_name[i]}")
        plt.tight_layout()
        plt.savefig('notebooks/figures/chapman_sample.png')
        print("Saved figure: notebooks/figures/chapman_sample.png")
        plt.close()

    except Exception as e:
        print(f"Error analyzing Chapman: {e}")

if __name__ == "__main__":
    analyze_mit_bih()
    analyze_chapman()
