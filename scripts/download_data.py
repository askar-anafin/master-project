import wfdb
import os

def download_dataset(dataset_name, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        print(f"Created directory: {target_dir}")
    
    print(f"Starting download of {dataset_name} to {target_dir}...")
    try:
        wfdb.dl_database(dataset_name, target_dir)
        print(f"Download of {dataset_name} completed successfully.")
    except Exception as e:
        print(f"Error downloading {dataset_name}: {e}")

def main():
    datasets = {
        'ptb-xl': os.path.join('data', 'raw', 'ptb-xl'),
        'mitdb': os.path.join('data', 'raw', 'mit-bih'),
        # Chapman-Shaoxing is 'chapman-shaoxing' on PhysioNet
        'chapman-shaoxing': os.path.join('data', 'raw', 'chapman-shaoxing')
    }

    for db_name, path in datasets.items():
        download_dataset(db_name, path)

if __name__ == "__main__":
    main()
