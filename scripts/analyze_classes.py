import wfdb
import pandas as pd
import os
import sys

def analyze_mit_bih_classes():
    print("\n--- MIT-BIH Class Analysis ---", flush=True)
    data_path = os.path.join('data', 'raw', 'mit-bih')
    records = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.dat')]
    
    annotation_counts = {}
    print(f"Found {len(records)} records.", flush=True)
    
    for rec in records:
        try:
            ann = wfdb.rdann(os.path.join(data_path, rec), 'atr')
            for symbol in ann.symbol:
                annotation_counts[symbol] = annotation_counts.get(symbol, 0) + 1
        except Exception as e:
            pass
            
    print("Annotation Symbol Counts:", flush=True)
    for sym, count in sorted(annotation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"'{sym}': {count}", flush=True)

def analyze_chapman_classes():
    print("\n--- Chapman-Shaoxing Class Analysis ---", flush=True)
    data_path = os.path.join('data', 'raw', 'chapman-shaoxing')
    mapping_file = os.path.join(data_path, 'ConditionNames_SNOMED-CT.csv')
    
    code_to_name = {}
    if os.path.exists(mapping_file):
        try:
            df = pd.read_csv(mapping_file)
            df['Snomed_CT'] = df['Snomed_CT'].astype(str)
            code_to_name = pd.Series(df['Acronym Name'].values, index=df['Snomed_CT']).to_dict()
        except Exception:
            pass
    
    diagnosis_counts = {}
    file_count = 0
    print("Parsing headers (this may take a moment)...", flush=True)
    
    # LIMIT to first 1000 files for speed if full scan is too slow
    # limit = 1000 
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.hea'):
                file_count += 1
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        content = f.read()
                        for line in content.split('\n'):
                            if line.startswith('#Dx:'):
                                codes = line.split(':')[1].strip().split(',')
                                for code in codes:
                                    code = code.strip()
                                    if code:
                                        name = code_to_name.get(code, code) # Use code if name not found
                                        diagnosis_counts[name] = diagnosis_counts.get(name, 0) + 1
                except Exception:
                    pass
                
                if file_count % 1000 == 0:
                     print(f"Processed {file_count} files...", flush=True)

    print(f"Total Processed: {file_count} records.", flush=True)
    print("Top 20 Diagnoses:", flush=True)
    sorted_counts = sorted(diagnosis_counts.items(), key=lambda x: x[1], reverse=True)
    for name, count in sorted_counts[:20]:
        print(f"'{name}': {count}", flush=True)

if __name__ == "__main__":
    analyze_mit_bih_classes()
    analyze_chapman_classes()
