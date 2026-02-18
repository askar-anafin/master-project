# Comparative Analysis of Deep Learning Architectures for Multi-Label ECG Classification

## 1. Methodology

### A. Hardware & Environment

All experiments were conducted on a high-performance workstation optimized for deep learning.

* **GPU**: NVIDIA GeForce RTX 3060 (12GB VRAM) - Utilized for mixed-precision (FP16) training.
* **CPU**: Multi-core Processor (e.g., Intel Core i7-12700K or equivalent) handling data loading and augmentation.
* **RAM**: 32 GB DDR4 System Memory.
* **Software Stack**:
  * **OS**: Windows 11 Pro
  * **Framework**: PyTorch 2.5 (CUDA 12.4 backend)
  * **Libraries**: NumPy, Pandas, Scikit-Learn, WFDB (PhysioNet).

### B. Dataset Description (PTB-XL)

The **PTB-XL ECG Database** is used, comprising **21,837** clinical 12-lead ECG records from 18,885 patients.

* **Signal Properties**: 10 seconds duration, sampled at 500 Hz (5000 time steps).
* **Lead Configuration**: Standard 12-lead system (I, II, III, aVR, aVL, aVF, V1-V6).
* **Labels**: 71 different ECG statements, mapped to 5 diagnostic superclasses:
    1. **NORM**: Normal Sinus Rhythm (9528 records)
    2. **MI**: Myocardial Infarction (5486 records)
    3. **STTC**: ST/T Change (5250 records)
    4. **CD**: Conduction Disturbance (4907 records)
    5. **HYP**: Hypertrophy (2655 records)
* **Class Balance**: The dataset is imbalanced, necessitating robust evaluation metrics like F1-Score and AUPRC.

### C. Data Preprocessing

1. **Bandpass Filtering**: 0.5 - 50 Hz (Butterworth filter) to remove baseline wander and powerline noise.
2. **Z-Score Normalization**: Applied per lead ($\mu=0, \sigma=1$) to standardize amplitude.
3. **Splitting**: Stratified Split ensuring patient separation:
    * **Training**: 80% (~17,400 records)
    * **Validation**: 10% (~2,100 records)
    * **Testing**: 10% (~2,100 records)

### D. Model Architectures

#### 1. CNN: ResNet-18 (1D Adapted)

A deep convolutional network adapted for 1D time-series data.

* **Input**: $(Batch, 12, 5000)$
* **Initial Block**: Conv1d ($k=15, s=2$) $\rightarrow$ BN $\rightarrow$ ReLU $\rightarrow$ MaxPool ($k=3, s=2$). Output channels: 64.
* **Residual Stages**:
  * **Stage 1**: 2 blocks (64 filters, stride 1)
  * **Stage 2**: 2 blocks (128 filters, stride 2)
  * **Stage 3**: 2 blocks (256 filters, stride 2)
  * **Stage 4**: 2 blocks (512 filters, stride 2)
* **Aggregation**: Global Average Pooling $\rightarrow$ Flatten.
* **Classifier**: Linear Layer ($512 \rightarrow 5$ classes).
* **Total Parameters**: ~3.8 Million.

#### 2. GNN: Spatio-Temporal Graph (ST-ReGE)

A hybrid architecture combining temporal feature extraction with graph-based spatial reasoning.

* **Nodes**: 12 (One per ECG lead).
* **Temporal Backbone**: Shared ResNet-like 1D CNN applied to each node independently. Extracts a 256-dim feature vector per lead.
* **Graph Construction**: A **Learnable Adjacency Matrix** ($12 \times 12$) initialized randomly and optimized via gradient descent to learn inter-lead correlations.
* **Graph Convolution**: Two GCN layers (Hidden Dim = 256) with residual connections ($X' = \sigma(AXW) + X$).
* **readout**: Mean pooling over the 12 nodes $\rightarrow$ Linear Classifier.
* **Why it works**: Captures "Infarct Localization" (e.g., Lead V1-V4 correlation for Anterior MI).

#### 3. ViT: 1D Vision Transformer

A pure attention-based architecture treating the signal as a sequence of tokens.

* **Patch Embedding**: Non-overlapping convolution ($k=50, s=50$). Splits 5000 time steps into **100 patches**.
* **Dimensionality**: $D_{model} = 128$.
* **Encoder**:
  * **Layers**: 4
  * **Heads**: 4 (Multi-head Self-Attention)
  * **Feed-Forward Network**: Dim 256
* **Positional Embedding**: Learnable parameters added to patch tokens.
* **Classifier**: CLS token $\rightarrow$ LayerNorm $\rightarrow$ Linear Layer.

### E. Evaluation Metrics

To rigorously assess model performance on the imbalanced multi-label PTB-XL dataset, we employed the following metrics:

1. **Macro F1-Score**: The harmonic mean of Precision and Recall, calculated per class and then averaged.
    * *Why*: It is the primary metric for imbalanced datasets, ensuring that rare classes (like Hypertrophy) contribute equally to the final score as common classes (like Normal).
2. **Macro AUROC (Area Under Receiver Operating Characteristic)**: Measures the model's ability to discriminate between positive and negative classes across all decision thresholds.
    * *Why*: Indicates the global ranking quality of the model. A score of 0.5 is random grading, while 1.0 is perfect.
3. **Macro AUPRC (Area Under Precision-Recall Curve)**: Summarizes the trade-off between precision and recall for different thresholds.
    * *Why*: More informative than AUROC for imbalanced data where the "positive" class is rare, as it focuses heavily on the minority class performance.
4. **Exact Match Accuracy**: The percentage of samples where the predicted set of labels *exactly matches* the true set of labels.
    * *Why*: A strict metric for multi-label classification. Partial correctness (getting 1 out of 2 diseases right) is counted as a failure here.
5. **PhysioNet Challenge Score**: A specialized metric from the PhysioNet/Computing in Cardiology Challenges.
    * *Why*: It enables a clinical risk-based evaluation. Misclassifying a sick patient as healthy (False Negative) is penalized significantly more than a false alarm (False Positive), reflecting real-world medical priorities.

---

## 2. Experimental Results

Training was performed for 20 epochs with early stopping.

**Table 1: Final Test Set Performance**

| Metric | **CNN (ResNet-18)** | **GNN (ST-ReGE)** | **ViT (Transformer)** |
| :--- | :--- | :--- | :--- |
| **Macro F1-Score** | **0.7245** | 0.7226 | 0.6503 |
| **Macro AUROC** | **0.9271** | 0.9236 | 0.8830 |
| **Macro AUPRC** | 0.8054 | **0.8066** | 0.7216 |
| **Exact Match Acc** | 63.21% | **63.76%** | 58.76% |
| **PhysioNet Score** | **0.7228** | 0.7016 | 0.6810 |
| **Training Time** | ~12 mins | ~14 mins | ~14 mins |

### Key Findings

1. **CNN Efficiency**: The ResNet-18 converged fastest and achieved the highest overall AUROC (0.927). Its local receptive fields are ideal for detecting the sharp transitions in QRS complexes.
2. **GNN Precision**: The ST-ReGE model achieved the best **AUPRC** and **Exact Match Accuracy**. This suggests that while it may miss some subtle cases (lower recall), its positive predictions are highly reliable (high precision), likely due to cross-referencing leads via the graph structure.
3. **ViT Challenges**: The Transformer suffered from overfitting on this "small" dataset (relative to ViT requirements). Without the inductive bias of convolutions, it struggled to generalize high-frequency noise vs. signal, resulting in a 7% drop in F1.

---

## 3. Conclusion & Recommendations

The **CNN (ResNet-18)** is recommended as the primary candidate for deployment due to its superior balance of speed, accuracy, and detailed feature extraction capability. However, the **GNN** approach validates the importance of spatial lead modeling and could be used as a secondary "expert" model to verify difficult cases, particularly for complex arrhythmias where lead correlation is diagnostic.

---

## Appendix: Case Study (Live Inference)

**Sample ID**: 1567
**Diagnosis**: **Normal Sinus Rhythm (NORM)**

| Model | Prediction | Confidence (NORM) | Correct? |
| :--- | :--- | :--- | :--- |
| **CNN** | NORM | 75% | ✅ |
| **GNN** | NORM | **89%** | ✅ |
| **ViT** | NORM | 78% | ✅ |

*The GNN demonstrated the highest confidence in this clear-cut case, showcasing its robustness.*
