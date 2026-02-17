import requests
import os
import zipfile
import tarfile

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
        return True
    else:
        print(f"Failed to download. Status code: {response.status_code}")
        return False

def download_ptb_xl():
    # PTB-XL v1.0.3 (static link)
    url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    target_dir = os.path.join('data', 'raw')
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, 'ptb-xl.zip')
    
    if download_file(url, target_path):
        print("Extracting PTB-XL...")
        with zipfile.ZipFile(target_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print("Extraction complete.")
        # Cleanup
        os.remove(target_path)

def download_mit_bih():
    # MIT-BIH v1.0.0
    url = "https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip"
    target_dir = os.path.join('data', 'raw')
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, 'mitdb.zip')
    
    if download_file(url, target_path):
        print("Extracting MIT-BIH...")
        with zipfile.ZipFile(target_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print("Extraction complete.")
        os.remove(target_path)

def download_chapman():
    # Chapman-Shaoxing v1.0.0
    url = "https://physionet.org/static/published-projects/chapman-shaoxing/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip"
    target_dir = os.path.join('data', 'raw')
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, 'chapman.zip')
    
    if download_file(url, target_path):
        print("Extracting Chapman-Shaoxing...")
        with zipfile.ZipFile(target_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print("Extraction complete.")
        os.remove(target_path)

if __name__ == "__main__":
    download_ptb_xl()
    download_mit_bih()
    download_chapman()
