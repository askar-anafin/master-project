# Data Upload Instructions

Since you are manually uploading the datasets, please ensure they are extracted into the following directories to match the project structure:

## 1. PTB-XL ECG Database

- **Source**: [https://physionet.org/content/ptb-xl/](https://physionet.org/content/ptb-xl/)
- **Target Directory**: `data/raw/ptb-xl/`
- **Expected Structure**:

  ```
  data/raw/ptb-xl/
  ├── ptbxl_database.csv
  ├── scp_statements.csv
  ├── records100/
  └── records500/
  ```

## 2. MIT-BIH Arrhythmia Database

- **Source**: [https://physionet.org/content/mitdb/](https://physionet.org/content/mitdb/)
- **Target Directory**: `data/raw/mit-bih/`
- **Expected Structure**:

  ```
  data/raw/mit-bih/
  ├── 100.dat
  ├── 100.hea
  ├── ...
  └── ANNOTATORS
  ```

## 3. Chapman-Shaoxing Database

- **Source**: [https://physionet.org/content/chapman-shaoxing/](https://physionet.org/content/chapman-shaoxing/)
- **Target Directory**: `data/raw/chapman-shaoxing/`
- **Expected Structure**:

  ```
  data/raw/chapman-shaoxing/
  ├── Diagnostics.xlsx (or similar metadata)
  ├── WFDBRecords/
  └── ...
  ```

**Note**: If your extracted folders have a top-level directory (e.g., `ptb-xl-1.0.3`), please move the *contents* into the target directory or update the path in your code accordingly.
