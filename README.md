# Heart Disease Diagnosis using ECG Signals

## Project Overview

This project aims to develop a machine learning/deep learning model to diagnose heart diseases using Electrocardiogram (ECG) signals. The goal is to classify ECG readings into at least 3 diagnostic classes (e.g., Normal, Myocardial Infarction, etc.).

## Dataset

We use the **PTB-XL ECG Database**, a large publicly available dataset containing:

- 21,837 clinical 12-lead ECG records (10 seconds length) from 18,885 patients.
- Comprehensive annotations including 71 distinct ECG statements.
- 5 superclasses: Normal (NORM), Myocardial Infarction (MI), ST/T Change (STTC), Conduction Disturbance (CD), and Hypertrophy (HYP).

## Project Structure

- `data/`: Contains raw and processed data (not included in version control).
- `notebooks/`: Jupyter notebooks for EDA and experimentation.
- `src/`: Source code modules.
- `scripts/`: Utility scripts (e.g., data download).
- `models/`: Trained models and checkpoints.

## Setup

1. Clone the repository.
2. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .\.venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset:

   ```bash
   python scripts/download_data.py
   ```

## Quick Start: Run the Demo 🚀

You can test the trained models on random samples from the test set using the `src/demo.py` script.

### 1. Diagnose Arrhythmias (MIT-BIH)

Detects: Normal (N), Supraventricular (S), Ventricular (V), Fusion (F), Unknown (Q).

```bash
python src/demo.py --mode arrhythmia
```

### 2. Diagnose Heart Disease (PTB-XL)

Detects: Normal (NORM), Myocardial Infarction (MI), Ischemia (STTC), Conduction Disturbance (CD), Hypertrophy (HYP).

```bash
python src/demo.py --mode disease
```
