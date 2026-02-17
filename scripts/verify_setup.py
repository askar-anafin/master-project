import sys
print(f"Python version: {sys.version}")

try:
    import pandas
    print(f"pandas: {pandas.__version__}")
    import numpy
    print(f"numpy: {numpy.__version__}")
    import wfdb
    print(f"wfdb: {wfdb.__version__}")
    import torch
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    import matplotlib
    print(f"matplotlib: {matplotlib.__version__}")
    import seaborn
    print(f"seaborn: {seaborn.__version__}")
    import sklearn
    print(f"sklearn: {sklearn.__version__}")
    
    print("\nAll core dependencies imported successfully!")
except ImportError as e:
    print(f"\nImportError: {e}")
    sys.exit(1)
