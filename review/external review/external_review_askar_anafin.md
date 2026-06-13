# EXTERNAL REVIEW OF MASTER'S THESIS

**Master's student:** Anafin Askar  
**Educational program:** Applied Data Analytics (7M06103)  
**Thesis title:** Development of a Heart Disease Diagnosis Technique Using ECG (Electrocardiogram) Signals  

---

### 1. Relevance of the Research Topic
Automated classification of standard 12-lead electrocardiogram (ECG) signals is a high-priority clinical challenge. Cardiovascular diseases are the leading cause of mortality globally, and diagnostic errors due to subjective manual interpretation or cognitive fatigue in emergency triage can lead to critical delays. 

While deep learning has bypassed traditional handcrafted feature engineering, standard neural networks often fail to model the physical, anatomical geometry of the 12-lead placement, treating them merely as independent channels. Furthermore, challenges such as multi-label co-occurrences and severe class imbalances complicate clinical adoption. By developing, optimizing, and comparing temporal and spatial architectures (CNN, GNN, Transformer, and Hybrid) alongside post-hoc calibration strategies, this thesis addresses a key research gap and provides a practical framework for clinical decision support.

---

### 2. General Characteristics of the Thesis
The master's thesis is written at a high academic level and follows a rigorous, logical structure that meets the academic standards of Astana IT University. The document is well-organized, starting from clear physiological principles and progressing through data preprocessing, model architectures, rigorous validation benchmarks, explainability modules, and web microservice implementation. 

The thesis relies on a solid bibliography (99 references) of contemporary peer-reviewed research in machine learning and cardiology. The inclusion of complete, modular, and reproducible PyTorch/FastAPI codebase listings in the appendices ensures full reproducibility of all reported experiments.

---

### 3. Structure and Content of the Thesis
* **The introductory chapter** establishes the background, defines a clear research aim, outlines the object, subject, and methodological framework, and states three verifiable research hypotheses.
* **The literature review** covers cardiac electrophysiology, the pathology of the target superclasses (NORM, MI, STTC, CD, HYP), and traces the historical development of computer-aided ECG interpretation.
* **The methodology chapter** details the signal preprocessing pipeline (4th-order Butterworth bandpass filtering, zero-phase double-pass filtering, and lead-wise Z-score normalization) and the structural configurations of the four evaluated neural networks.
* **The experimental results chapter** documents the training protocol on the PTB-XL database, presents a detailed quantitative benchmark, outlines post-hoc ensembling and validation-set threshold optimization, illustrates local explainability via 1D Grad-CAM, analyzes qualitative case studies, and describes the web microservice prototype.
* **The conclusion** summarizes the findings against the objectives, confirms the verification status of the three hypotheses, and outlines future deployment challenges.

---

### 4. Strengths of the Work
1. **Multi-Perspective Benchmark:** By comparing models representing different inductive biases (CNN for local morphology, GNN for spatial propagation, ViT for global sequence attention, and a Spatio-Temporal Hybrid), the study provides a comprehensive evaluation of architectural trade-offs.
2. **Dynamic Lead Modeling:** The introduction of a learnable adjacency matrix $A \in \mathbb{R}^{12 \times 12}$ inside the ST-ReGE GNN represents a methodologically sound way to dynamically discover anatomical lead relationships rather than relying on static, physical distance metrics.
3. **Calibrated Performance Gains:** The implementation of validation-set threshold grid search significantly improves model metrics under class imbalance. The soft voting ensemble under optimized thresholds achieves a Macro F1-score of **0.7439** and a Macro AUROC of **0.9323**, demonstrating practical, non-retraining calibration benefits.
4. **Physiological Explainability:** The 1D Grad-CAM implementation successfully localizes the elevated ST-segment and T-wave onset on Lead V2 for Myocardial Infarction predictions, validating that the models align with standard cardiological guidelines.
5. **Microservice Translation:** Packaging the checkpoints into a FastAPI microservice with a working dashboard mock-up bridges the gap between theoretical deep learning and practical clinical triage.

---

### 5. Comments and Suggestions
Despite the high quality of the work, the following minor comments are made to improve the research:
1. **Model Benchmark Paradox:** Although the thesis emphasizes the physiological value of spatio-temporal graph modeling, the baseline 1D ResNet-18 model (Macro F1 = 0.7273) still slightly outperforms the proposed ST-ReGE GNN (Macro F1 = 0.7258) and Hybrid (Macro F1 = 0.7063) models. The author should address why the added structural complexity of learnable graphs and self-attention did not yield a clear benchmark advantage over simple 1D convolutions on the PTB-XL dataset.
2. **Qualitative Web Evaluation:** While the FastAPI dashboard is a valuable practical contribution, its evaluation is purely qualitative. The thesis would be strengthened by including a quantitative latency benchmark (such as average inference time per sample on CPU and GPU configurations) to demonstrate the microservice's real-time feasibility.

---

### 6. Conclusion
Overall, the master's thesis is completed at a high academic level. It addresses a well-defined clinical research gap using a methodologically sound, reproducible design, and provides findings that are both statistically interpretable and practically applicable. The thesis meets all the criteria of the Applied Data Analytics curriculum and deserves the grade «A».

**Reviewer:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_  
**Signature:** \_\_\_\_\_\_\_\_\_\_\_\_ \hfill **Date:** \_\_\_\_\_\_\_\_\_\_\_\_  

