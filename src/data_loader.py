import wfdb
import os
import numpy as np

def download_mitdb(download_dir='data/raw'):
    """
    Downloads the MIT-BIH Arrhythmia Database from PhysioNet.
    
    Args:
        download_dir (str): Directory where the data will be saved.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        
    print(f"Downloading MIT-BIH Arrhythmia Database to {download_dir}...")
    
    # The MIT-BIH Arrhythmia Database name on PhysioNet is 'mitdb'
    wfdb.dl_database('mitdb', download_dir)
    
    print("Download complete.")

def load_record(record_id, data_dir='data/raw'):
    """
    Loads a specific record from the local database.
    
    Args:
        record_id (str): The ID of the record (e.g., '100').
        data_dir (str): Directory where the data is stored.
        
    Returns:
        record (wfdb.Record): The record object containing signal and metadata.
        annotation (wfdb.Annotation): The annotation object containing labels.
    """
    record_path = os.path.join(data_dir, str(record_id))
    
    try:
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        return record, annotation
    except Exception as e:
        print(f"Error loading record {record_id}: {e}")
        return None, None

if __name__ == "__main__":
    # Example usage: download data if run as a script
    download_mitdb()
