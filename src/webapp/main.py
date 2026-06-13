import torch
import numpy as np
import os
import sys
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.getcwd())

from src.train_unified import get_model
from src.ptbxl.loader import PTBXLDataset

app = FastAPI(title="Heart Disease Diagnosis API")

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global dependencies
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
MODELS = {}
DATASET = None

def load_models():
    print(f"Loading models to {DEVICE}...")
    model_paths = {
        'cnn': ('cnn', 'experiments/cnn_filtered/best_model.pth'),
        'gnn': ('gnn', 'experiments/gnn_filtered/best_model.pth'),
        'vit': ('vit', 'experiments/vit_filtered/best_model.pth'),
        'hybrid': ('hybrid', 'experiments/hybrid_filtered/best_model.pth')
    }
    
    for name, (type_name, path) in model_paths.items():
        if os.path.exists(path):
            try:
                model = get_model(type_name)
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                model.eval()
                model.to(DEVICE)
                MODELS[name] = model
                print(f"Loaded {name}")
            except Exception as e:
                print(f"Failed to load {name}: {e}")
        else:
            print(f"Model path not found: {path} (skipping {name})")

@app.on_event("startup")
async def startup_event():
    global DATASET
    print("Starting up API...")
    try:
        DATASET = PTBXLDataset()
    except Exception as e:
        print(f"Failed to load dataset: {e}")
    load_models()

@app.get("/api/sample")
async def get_random_sample():
    if DATASET is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded.")
        
    try:
        splits = DATASET.get_train_val_test_split()
        test_indices = splits['test_idx']
        idx = int(np.random.choice(test_indices))
        
        # Get raw signal data
        X_all = splits['X']
        y_all = splits['y']
        
        signal = X_all[idx].copy() # shape (5000, 12)
        label = y_all[idx].copy() # shape (5,)
        
        # Downsample signal by a factor of 5 for frontend performance (5000 -> 1000 pts)
        downsampled_signal = signal[::5, :]
        
        # Get true diagnoses
        true_diagnoses = [CLASSES[i] for i, val in enumerate(label) if val == 1]
        
        return {
            "index": idx,
            "signal": downsampled_signal.tolist(), # List of 1000 lists of 12 floats
            "true_diagnoses": true_diagnoses
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class PredictionResult(BaseModel):
    model: str
    probabilities: Dict[str, float]
    predicted_classes: List[str]

@app.post("/api/predict/{sample_idx}", response_model=List[PredictionResult])
async def predict_sample(sample_idx: int):
    if DATASET is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded.")
        
    if not MODELS:
        raise HTTPException(status_code=500, detail="No models loaded.")
    
    try:
        splits = DATASET.get_train_val_test_split()
        X_all = splits['X']
        
        if sample_idx < 0 or sample_idx >= len(X_all):
            raise HTTPException(status_code=400, detail=f"Invalid sample index {sample_idx}")
            
        signal = X_all[sample_idx].copy()
        input_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        results = []
        model_probs = {}
        for model_name, model in MODELS.items():
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                
            probs_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
            predicted_classes = [CLASSES[i] for i in range(len(CLASSES)) if probs[i] > 0.5]
            model_probs[model_name] = probs
            
            results.append(PredictionResult(
                model=model_name,
                probabilities=probs_dict,
                predicted_classes=predicted_classes
            ))
            
        if MODELS:
            # Optimal ensemble: cnn + gnn + hybrid
            ensemble_candidates = [model_probs[name] for name in ['cnn', 'gnn', 'hybrid'] if name in model_probs]
            if not ensemble_candidates:
                ensemble_candidates = list(model_probs.values())
            
            probs_ensemble = np.mean(ensemble_candidates, axis=0)
            probs_ensemble_dict = {CLASSES[i]: float(probs_ensemble[i]) for i in range(len(CLASSES))}
            
            # Calibrated validation-set thresholds
            ensemble_thresholds = {'NORM': 0.45, 'MI': 0.36, 'STTC': 0.42, 'CD': 0.52, 'HYP': 0.35}
            predicted_ensemble_classes = [
                CLASSES[i] for i in range(len(CLASSES))
                if probs_ensemble[i] > ensemble_thresholds.get(CLASSES[i], 0.5)
            ]
            
            results.append(PredictionResult(
                model="ensemble",
                probabilities=probs_ensemble_dict,
                predicted_classes=predicted_ensemble_classes
            ))
            
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class UploadPredictionResponse(BaseModel):
    signal: List[List[float]]
    predictions: List[PredictionResult]

@app.post("/api/predict_upload", response_model=UploadPredictionResponse)
async def predict_upload(file: UploadFile = File(...)):
    if not MODELS:
        raise HTTPException(status_code=500, detail="No models loaded.")
        
    try:
        contents = await file.read()
        filename = file.filename
        
        # Parse data based on file type
        if filename.endswith('.json'):
            import json
            data = json.loads(contents.decode('utf-8'))
            signal = np.array(data, dtype=np.float32)
        elif filename.endswith('.csv'):
            import io
            import pandas as pd
            df = pd.read_csv(io.BytesIO(contents))
            # If the CSV has column headers, df.values will exclude them, which is perfect.
            signal = df.values.astype(np.float32)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a .json or .csv file.")
            
        # Standardize shape to (5000, 12)
        if signal.shape == (12, 5000):
            signal = signal.T
            
        if signal.shape != (5000, 12):
            raise HTTPException(status_code=400, detail=f"Invalid signal shape {signal.shape}. Expected (5000, 12) or (12, 5000).")
            
        # Run inference
        input_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        results = []
        model_probs = {}
        for model_name, model in MODELS.items():
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                
            probs_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
            predicted_classes = [CLASSES[i] for i in range(len(CLASSES)) if probs[i] > 0.5]
            model_probs[model_name] = probs
            
            results.append(PredictionResult(
                model=model_name,
                probabilities=probs_dict,
                predicted_classes=predicted_classes
            ))
            
        if MODELS:
            # Optimal ensemble: cnn + gnn + hybrid
            ensemble_candidates = [model_probs[name] for name in ['cnn', 'gnn', 'hybrid'] if name in model_probs]
            if not ensemble_candidates:
                ensemble_candidates = list(model_probs.values())
            
            probs_ensemble = np.mean(ensemble_candidates, axis=0)
            probs_ensemble_dict = {CLASSES[i]: float(probs_ensemble[i]) for i in range(len(CLASSES))}
            
            # Calibrated validation-set thresholds
            ensemble_thresholds = {'NORM': 0.45, 'MI': 0.36, 'STTC': 0.42, 'CD': 0.52, 'HYP': 0.35}
            predicted_ensemble_classes = [
                CLASSES[i] for i in range(len(CLASSES))
                if probs_ensemble[i] > ensemble_thresholds.get(CLASSES[i], 0.5)
            ]
            
            results.append(PredictionResult(
                model="ensemble",
                probabilities=probs_ensemble_dict,
                predicted_classes=predicted_ensemble_classes
            ))
            
        # Downsample signal by a factor of 5 for frontend performance
        downsampled_signal = signal[::5, :]
        
        return UploadPredictionResponse(
            signal=downsampled_signal.tolist(),
            predictions=results
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

# Root redirect to index.html
from fastapi.responses import RedirectResponse
@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")
