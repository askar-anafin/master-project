import os

def inspect_chapman_header():
    base_path = os.path.join('data', 'raw', 'chapman-shaoxing')
    
    # Find a .hea file
    found_file = False
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.hea') and not found_file:
                full_path = os.path.join(root, file)
                print(f"--- Header File: {full_path} ---")
                try:
                    with open(full_path, 'r') as f:
                        print(f.read())
                except Exception as e:
                    print(f"Error reading header: {e}")
                found_file = True
                break
        if found_file: break

if __name__ == "__main__":
    inspect_chapman_header()
