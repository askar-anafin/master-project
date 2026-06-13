# SUPERVISOR REVIEW OF MASTER'S THESIS

**Master's student:** Anafin Askar  
**Educational program:** Applied Data Analytics (7M06103)  
**Thesis title:** Development of a Heart Disease Diagnosis Technique Using ECG (Electrocardiogram) Signals  

---

### 1. Relevance of the Research Topic
The topic of this master's thesis is highly relevant. Cardiovascular diseases (CVDs) remain the leading cause of death globally, claiming approximately 17.9 million lives each year. Electrocardiography (ECG) is the primary non-invasive diagnostic standard. However, manual interpretation is subjective, slow, and shows a high inter-observer variability exceeding 20% in emergency and primary care settings. Computer-aided diagnostic tools powered by deep learning offer a viable solution for automated triage and clinical decision support. 

Existing machine learning approaches are either fragile due to their reliance on manual feature extraction (e.g. peak detection) or treat multi-lead ECGs as independent channels, ignoring the physical spatial geometry of electrode placements. This research addresses this gap by designing, training, and benchmarking architectures that model both temporal morphologies and spatial inter-lead relations, and packages these models into a real-time web microservice.

---

### 2. General Characteristics of the Thesis
The master's thesis is logically structured and consists of an abstract, acknowledgements, declarations, list of figures and tables, table of contents, 4 main chapters, a conclusion, a list of references (99 citations), and six appendices. 

A notable strength of the thesis is the inclusion of complete, fully reproducible PyTorch and FastAPI source code listings in Appendices A–F. The student has organized the repository cleanly, ensuring that preprocessing, model training, evaluation, ensembling, and local deployment can be executed from a single workspace, which reflects excellent software engineering discipline.

---

### 3. Scientific Novelty
The scientific novelty of the work is substantiated by several original contributions:
1. **Dynamic Spatial Lead Geometry:** Integrating a learnable adjacency matrix $A = \text{Sigmoid}(W_{adj}) \in \mathbb{R}^{12 \times 12}$ inside a Spatio-Temporal Graph Neural Network (ST-ReGE) to dynamically capture correlations between anatomically contiguous leads (septal, anterior, lateral, inferior).
2. **Standardized Deep Learning Benchmark:** Conducting a component-wise comparative evaluation on the large-scale PTB-XL database, directly contrasting CNN (morphology-focused 1D ResNet-18), GNN (space-focused ST-ReGE), Transformer (sequence-focused ViT-1D), and Spatio-Temporal Hybrid architectures under identical validation conditions.
3. **Post-Hoc Optimization Framework:** Proposing validation-set threshold grid search and soft voting ensembling to address class imbalance and multi-label co-occurrence, significantly boosting performance without expensive model retraining.

---

### 4. Brief Description of the Thesis
* **Chapter 1: Introduction** defines the clinical context, problem statement, research aim, objectives, methodological framework, scientific novelty, and practical significance. It clearly formulates three research hypotheses regarding inductive biases, spatial graphs, and model explainability.
* **Chapter 2: Literature Review and Theoretical Foundations** establishes the electrophysiological basis of the heart, details the pathophysiology of the five target diagnostic superclasses (NORM, MI, STTC, CD, HYP), and reviews the history of ECG analysis from rule-based systems to deep neural networks.
* **Chapter 3: Digital Signal Processing and Deep Learning Model Design** outlines the preprocessing pipeline (a 4th-order Butterworth bandpass filter from 0.5 to 50 Hz using zero-phase forward-backward filtering to prevent phase distortion and lead-wise Z-score normalization) and details the structural specifications of the trained CNN, GNN, ViT, and Hybrid models.
* **Chapter 4: Experimental Evaluation and Prototype Implementation** presents training hyperparameters, benchmark results on the PTB-XL test partition, ablation studies, qualitative case studies, 1D Grad-CAM feature visualizations, threshold optimization, ensembling, and the deployment details of the FastAPI web server.

---

### 5. Student Performance Evaluation
During the preparation of the thesis, the student, Askar Anafin, demonstrated a high level of research independence, technical capability, and methodological discipline. Confronted with a challenging multi-label classification problem on clinical signal sequences, the student initially faced challenges with the training instability of the graph neural network and the severe overfitting of the 1D Vision Transformer baseline due to its lack of local inductive biases. 

The student showed excellent research initiative and engineering discipline in debugging these issues, which led to:
* Implementing the **Spatio-Temporal Hybrid (ResNet-Transformer)** model to combine convolutional morphological compression with transformer global attention.
* Designing and implementing the **validation-set threshold grid search** to calibrate model decisions, raising the Macro F1-score of the voting ensemble from 0.7398 to **0.7439** and Macro AUROC to **0.9323**.
* Developing a functional **FastAPI web microservice and triage GUI** that simulates real-time patient diagnosis.

The student responded constructively to scientific feedback, systematically addressed class imbalances, and remained objective and transparent about the performance limits of the data-hungry Vision Transformer baseline. The work demonstrates the candidate's readiness for professional and research practice in applied data analytics.

---

### 6. Conclusion
The master's thesis has been completed in full, meets all academic integrity and reporting standards of the Applied Data Analytics curriculum, and is recommended for defense. The work deserves the grade «A».

**Scientific Supervisor:**  
Minsoo Hahn, PhD, Professor,  
Department of Computational and Data Sciences,  
School of Artificial Intelligence and Data Science,  
Astana IT University LLP  

**Signature:** \_\_\_\_\_\_\_\_\_\_\_\_  
**Date:** \_\_\_\_\_\_\_\_\_\_\_\_  
