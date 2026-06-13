# Master's Thesis Defense: 12-Minute Speech Script

**Project Title**: Development of a Heart Disease Diagnosis Technique Using ECG (Electrocardiogram) Signals  
**Candidate**: Askar Anafin  
**Scientific Supervisor**: Minsoo Hahn, PhD  
**Degree**: Master of Science in Applied Data Analytics (7M06103)  
**Institution**: Astana IT University, School of Artificial Intelligence and Data Science  
**Target Duration**: Exactly 12 Minutes (720 Seconds)

---

## ⏱️ Speech Timing Overview

| Slide # | Slide Title | Category | Target Time | Word Count | Speaking Pace |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | Title Slide | Title | 30s | 65 words | Calm & Formal |
| **2** | Relevance of Thesis Topic | Introduction | 35s | 75 words | Clear & Urgent |
| **3** | Research Aim | Introduction | 20s | 48 words | Steady & Decisive |
| **4** | Objectives | Introduction | 30s | 65 words | Structured & Fast |
| **5** | Literature Review | Introduction | 35s | 75 words | Comparative |
| **6** | Novelty of the Research | Introduction | 35s | 75 words | Confident & Proud |
| **7** | Problem Statement: Input/Output Tensors | Methodology | 30s | 65 words | Technical & Exact |
| **8** | ECG Signal Data Table | Methodology | 20s | 40 words | Clear & Tabular |
| **9** | Initial Dataset Overview | Methodology | 30s | 65 words | Informative |
| **10** | Zero-Phase Signal Preprocessing | Methodology | 30s | 65 words | Detailed |
| **11** | Diagnostic Label Filtering | Methodology | 25s | 55 words | Analytical |
| **12** | Baseline Models: CNN & ViT | Methodology | 30s | 76 words | Brief Baselines |
| **13** | Spatio-Temporal ReGE GNN | Methodology | 35s | 75 words | Core Focus |
| **14** | Spatio-Temporal Hybrid Model | Methodology | 35s | 75 words | Core Focus |
| **15** | Core Outcomes: ST-ReGE & Hybrid | Methodology | 35s | 70 words | Synthesis |
| **16** | Ensemble & Boundary Calibration | Methodology | 30s | 67 words | Core Focus |
| **17** | Why Multi-Label Evaluation Metrics Matter | Methodology | 30s | 74 words | Core Focus |
| **18** | Quantitative Model Benchmarks | Experiments | 40s | 85 words | Confident & clear |
| **19** | Model Interpretability (1D Grad-CAM) | Results | 35s | 75 words | Analytical & Visual |
| **20** | Clinical Error Analysis (Confusion Matrix) | Results | 35s | 75 words | Clinical & Logical |
| **21** | FastAPI Diagnostic Triage Dashboard | Integration | 35s | 75 words | Practical |
| **22** | Scholarly Publication | Conclusions | 15s | 31 words | Clear & Brief |
| **23** | Conclusion | Conclusions | 30s | 65 words | Summarizing |
| **24** | Future Directions | Conclusions | 25s | 53 words | Visionary |
| **25** | Thank You | Closing | 15s | 37 words | Warm & Open |
| **Total**| **25 Slides** | | **720s (12.0m)**| **1,586 words**| **~132 words / min** |

---

## 🎙️ Slide-by-Slide Defense Script

### Slide 1: Title Slide
* **Visuals**: Left-aligned Georgia serif title emphasizing *Heart Disease Diagnosis* in rust italics, candidate/supervisor details in split columns.
* **Timing**: **35 Seconds** | **Word Count**: 75 words
* **Speech**:
  > "Good afternoon, esteemed members of the State Examination Committee, scientific supervisor, and colleagues. My name is Askar Anafin, and today I present my Master’s thesis: *'Development of a Heart Disease Diagnosis Technique Using ECG Signals'*. This research was completed under the guidance of Dr. Minsoo Hahn at Astana IT University. In this defense, I will explain how we leverage spatio-temporal deep learning to automate multi-label cardiovascular screening."
* **Presenter's Tip**: Speak slowly, stand tall, and establish eye contact with the committee members.

---

### Slide 2: Relevance of Thesis Topic
* **Visuals**: Left list of motivations and research gaps, right card highlighting "The Research Need".
* **Timing**: **35 Seconds** | **Word Count**: 71 words
* **Speech**:
  > "Cardiovascular diseases are the leading cause of death worldwide, causing eighteen million deaths annually. ECG is the gold standard, but manual reading is slow, subjective, and carries a twenty percent error rate in emergency settings. Traditional tools rely on fragile peak detection, and standard CNNs treat leads independently, ignoring the anatomical geometry of electrode placement. We need models that capture both wave morphology and lead propagation."
* **Presenter's Tip**: Emphasize the "twenty percent error rate" and "spatial lead gap" as key motivators.

---

### Slide 3: Research Aim
* **Visuals**: Bold central block for formal statement, bottom details card on impact, scope, and deployment.
* **Timing**: **25 Seconds** | **Word Count**: 55 words
* **Speech**:
  > "Our research aim is directly addressed to this gap: to develop and evaluate deep learning methods for automated multi-lead ECG classification to provide real-time clinical decision support and patient triage. Practically, this targets reducing observer errors in ER rooms using standard twelve-lead signals, packaged into a lightweight web dashboard."
* **Presenter's Tip**: Keep this slide punchy; state the aim with clear confidence.

---

### Slide 4: Objectives
* **Visuals**: Two columns of rounded container cards (four on the left, three on the right) mapping the 7 objectives.
* **Timing**: **30 Seconds** | **Word Count**: 65 words
* **Speech**:
  > "To achieve this aim, we set seven key objectives: first, build a zero-phase filtering pipeline. Second, establish a ResNet-18 morphological baseline. Third, design a learnable GNN model for spatial lead relationships. Fourth, develop a temporal ViT. Fifth, build a Spatio-Temporal Hybrid. Sixth, calibrate decision boundaries using post-hoc validation search. And seventh, deploy an interactive FastAPI dashboard."
* **Presenter's Tip**: Read these sequentially but quickly, pointing out the progression from filtering to model design, and finally deployment.

---

### Slide 5: Literature Review
* **Visuals**: Benchmark table comparing historical methods (Kiranyaz, Rajpurkar, Ribeiro, Ribeiro, Zhang, Vaid, Strodthoff) with this thesis highlighted in cream.
* **Timing**: **35 Seconds** | **Word Count**: 75 words
* **Speech**:
  > "A review of current literature highlights that existing models either focus on single-lead datasets, rely on fixed spatial graphs, or require massive datasets like Vaid's Transformer. Our work places itself directly on the standardized PTB-XL benchmark. By combining a learnable spat-temporal GNN with validation threshold calibration, we achieve a Macro F1 of point-seven-five-zero-six, outperforming standard static convolutional benchmarks."
* **Presenter's Tip**: Point to the bottom row of the table, showing the performance gain of this thesis.

---

### Slide 6: Novelty of the Research
* **Visuals**: Three-column card layout (Methodology, Evaluation, Calibration) and bottom italicized callout box.
* **Timing**: **35 Seconds** | **Word Count**: 75 words
* **Speech**:
  > "The scientific novelty of this thesis is three-fold. First, we model the standard leads as graph nodes, dynamically learning anatomical pathways with a sigmoid-activated adjacency matrix. Second, we present a fair, standardized comparison of CNN, GNN, ViT, and Hybrid architectures on a single filtered dataset. Third, we introduce validation-set threshold grid search to calibrate decision boundaries, optimizing recall without the need for expensive model retraining."
* **Presenter's Tip**: Emphasize "dynamically learning anatomical pathways" as the chief methodology innovation.

---

### Slide 7: Problem Statement: Input/Output Tensors
* **Visuals**: Side-by-side cards: Input Data (ECG signal tensor) and Output Data (diagnostic labels).
* **Timing**: **30 Seconds** | **Word Count**: 65 words
* **Speech**:
  > "Let us define the mathematical boundaries of the system. The input is a high-dimensional digital signal tensor, X, recording twelve chest and limb leads over ten seconds sampled at five-hundred Hertz. The output is a binary vector mapping to the five diagnostic superclasses: Normal, Infarction, ST/T changes, Conduction block, and Hypertrophy, where multiple diagnoses are allowed to co-occur."
* **Presenter's Tip**: Pronounce the tensor dimensions clearly: "B by twelve by five-thousand."

---

---

### Slide 8: ECG Signal Data Table (12 Leads)
* **Visuals**: Table showing the first 5 time steps (rows) and standard 12 leads (columns) with mV values. Explanatory card below.
* **Timing**: **20 Seconds** | **Word Count**: 40 words
* **Speech**:
  > "To make the high-dimensional input tensor concrete, this slide displays the actual digital ECG signal values in millivolts for the first five time steps of a record. Each row is a sample separated by two milliseconds. A full ten-second recording contains five thousand such rows across the twelve columns."
* **Presenter's Tip**: Point to the columns and explain that this tabular format is how the signals are fed into our models.

---

### Slide 9: Initial Dataset Overview
* **Visuals**: Left bullet list on dataset composition (with raw count, sex coding, and stratified folds), right card showing counts.
* **Timing**: **30 Seconds** | **Word Count**: 83 words
* **Speech**:
  > "We utilize the PTB-XL database, containing twenty-one thousand seven-hundred and ninety-nine records from eighteen thousand eight-hundred and sixty-nine patients. Demographically, fifty-two percent are male and forty-eight percent are female, with a median age of sixty-two. To prevent data leakage, we use standard patient-stratified splits, folds one to ten. This groups all records of a single patient together, maintaining diagnostic class ratios. However, we face a severe class imbalance where normal records outnumber hypertrophy four-to-one, requiring post-hoc decision boundary calibration."
* **Presenter's Tip**: Explain that patient-stratification is vital because multiple ECGs from the same patient must not cross training and test sets.

---

### Slide 10: Zero-Phase Signal Preprocessing
* **Visuals**: Left details on noise filters, right card showing side-by-side plot comparing raw and preprocessed ECG waveforms.
* **Timing**: **30 Seconds** | **Word Count**: 55 words
* **Speech**:
  > "Raw ECGs contain baseline drift from breathing and high-frequency muscle noise. Our preprocessing maps raw voltages to Z-score standardized values, limits the spectrum to point-five to fifty Hertz using a fourth-order Butterworth filter, and applies zero-phase forward-backward filtering. This removes wave displacement, scaling all twelve leads into standardized, noise-free five-thousand-sample vectors."
* **Presenter's Tip**: Highlight the comparison plot on the right, pointing out the elimination of baseline wander.

---

### Slide 11: Diagnostic Label Filtering
* **Visuals**: Left text explaining all-zeros target bias, right counts table showing before/after filtering.
* **Timing**: **30 Seconds** | **Word Count**: 65 words
* **Speech**:
  > "A crucial data cleaning step was filtering out four-hundred and eleven records that only contained non-diagnostic codes, resulting in all-zero label vectors. Leaving these empty vectors in the test set artificially inflates precision-sensitive metrics. By removing them, we ensure an unbiased benchmark of twenty-one thousand three-hundred and eighty-eight records, with zero effect on class positive totals."
* **Presenter's Tip**: Emphasize that this step aligns with standard benchmarking protocols.

---

### Slide 12: Baseline Models: CNN & ViT
* **Visuals**: Side-by-side cards: 1D Adapted ResNet-18 and 1D Vision Transformer (ViT-1D) layout details.
* **Timing**: **35 Seconds** | **Word Count**: 75 words
* **Speech**:
  > "To establish a baseline, we first evaluated two standard paradigms without modifying their settings: a one-D adapted ResNet-eighteen for temporal wave morphology, and a one-D Vision Transformer to model long-range cardiac rhythms. The CNN baseline uses weight sharing and translation invariance to achieve data efficiency, while the ViT struggles on our dataset due to a lack of spatial and temporal inductive biases."
* **Presenter's Tip**: Briefly introduce these as standard models to set the stage for our custom architectures.

---

### Slide 13: Spatio-Temporal ReGE GNN
* **Visuals**: Left details on lead-spatial dynamic graph, right card detailing GCN formula, learnable adjacency matrix concept, and social network analogy.
* **Timing**: **35 Seconds** | **Word Count**: 72 words
* **Speech**:
  > "To capture lead spatial correlations, we designed the Spatio-Temporal Relation Graph GNN. We model the twelve leads as graph nodes and extract shapes with a shared CNN. We introduce a learnable adjacency matrix, A. Instead of using rigid physical distances, the model dynamically learns functional connections directly from data. It acts like a social network: leads learn to share electrical 'news' across the heart wall dynamically."
* **Presenter's Tip**: Explain that the learnable adjacency matrix allows the model to map functional electrical pathways.

---

### Slide 14: Spatio-Temporal Hybrid Model
* **Visuals**: Left details on morphology & rhythm fusion, right card detailing synergy concept (spellchecker vs. speed-reader) and tuning.
* **Timing**: **35 Seconds** | **Word Count**: 75 words
* **Speech**:
  > "We also designed the Spatio-Temporal Hybrid model. Pure Transformers struggle to learn raw beat shapes because they lack a locality bias. We solve this by stacking a CNN front-end to extract local wave shapes, followed by Squeeze-and-Excitation attention to calibrate lead channels. Then, a Transformer Encoder models the sequence. This blends local morphology and global rhythm. Think of it like a team: the CNN is a spellchecker verifying beat shapes, and the Transformer speed-reads overall rhythm flow."
* **Presenter's Tip**: Use the spellchecker and speed-reader analogy to explain how the hybrid model integrates local shape and global rhythm.

---

### Slide 15: Core Outcomes: ST-ReGE & Hybrid
* **Visuals**: Side-by-side cards: ST-ReGE GNN Outcomes and Spatio-Temporal Hybrid Outcomes.
* **Timing**: **35 Seconds** | **Word Count**: 70 words
* **Speech**:
  > "Let's synthesize the core outcomes. The ST-ReGE GNN models leads as graph nodes, learning a dynamic adjacency matrix. This resolves the spatial lead propagation gap, grouping relevant leads together. Meanwhile, the Spatio-Temporal Hybrid uses a ResNet front-end and channel attention before Transformer sequence modeling, solving the data-hunger constraint of pure Transformers. Both are theoretically superior but have higher computational complexity than baseline convolutions."
* **Presenter's Tip**: Emphasize how each model directly addresses a different gap (spatial propagation versus temporal rhythm) and highlight their respective limitations.

---

### Slide 16: Ensemble & Boundary Calibration
* **Visuals**: Left card outlining soft voting and validation search, right card detailing clinical benefits.
* **Timing**: **30 Seconds** | **Word Count**: 67 words
* **Speech**:
  > "To address severe class imbalance, we implement validation-set adjustments. First, we use a soft-voting ensemble averaging CNN, GNN, and Hybrid predictions to stabilize minority class scores. Second, we run a validation grid search to replace default point-five boundaries with class-specific thresholds. This calibration preserves rank-ordering, runs in seconds on CPU without retraining, and allows on-the-fly threshold tuning to favor recall."

---

### Slide 17: Why Multi-Label Evaluation Metrics Matter
* **Visuals**: Left column explaining imbalance and Macro F1/AUPRC metrics, right card detailing clinical triage trade-offs and PhysioNet score.
* **Timing**: **30 Seconds** | **Word Count**: 74 words
* **Speech**:
  > "In imbalanced clinical tasks, standard classification accuracy is misleading: predicting all negative for hypertrophy yields eighty-eight percent accuracy but is useless. We use Macro F1-score to calculate the balanced harmonic mean of precision and recall for each superclass independently, ensuring rare diseases are weighted equally. AUPRC and AUROC evaluate threshold-independent ranking and positive pathology detection, allowing clinicians to tune recall for triage and precision to avoid false alarms."
* **Presenter's Tip**: Explain that Macro F1 ensures that the model cannot achieve high scores by ignoring minority conditions like Hypertrophy.
* **Presenter's Tip**: Read the optimized thresholds: "such as point-thirty-four for ST/T changes and point-thirty-nine for Hypertrophy, which are adjusted downward to boost recall."

---

### Slide 18: Quantitative Model Benchmarks
* **Visuals**: Performance table highlighting the calibrated ensemble in cream, bottom result summary.
* **Timing**: **40 Seconds** | **Word Count**: 85 words
* **Speech**:
  > "The experimental results confirm our hypotheses. The ST-ReGE GNN achieved the highest individual Macro F1 of point-seven-four-four-four, validating the power of learnable spatial lead message passing. The CNN baseline and Spatio-Temporal Hybrid also performed strongly, proving the effectiveness of local convolutional morphology. Finally, ensembling and calibrating decision boundaries pushed the Macro F1 to point-seven-five-zero-six and AUROC to point-ninety-three-forty-four, yielding our best overall diagnostic system."
* **Presenter's Tip**: Point to the ensemble column and read the exact numbers proudly: "point-seven-five-zero-six."

---

### Slide 19: Model Interpretability (1D Grad-CAM)
* **Visuals**: Left bullet list on Grad-CAM methodology, right card showing 1D heatmaps overlaid on ECG waveforms.
* **Timing**: **35 Seconds** | **Word Count**: 75 words
* **Speech**:
  > "To verify that our models use medically sound features rather than background noise, we applied 1D Grad-CAM. By computing gradients of target class scores relative to activation maps of the last convolutional layer, we generate activation heatmaps. For Myocardial Infarction predictions, the heatmap peaks are localized directly on the ST-segment elevation in Lead V2. This direct alignment with standard cardiology criteria confirms the model acts as an interpretable diagnostic assistant."
* **Presenter's Tip**: Explain that high activation matches the clinical diagnostic criteria for acute ischemia.

---

### Slide 20: Clinical Error Analysis (Confusion Matrix)
* **Visuals**: Left bullet list on sensitivity and confusion justification, right card showing the class-wise confusion matrix.
* **Timing**: **35 Seconds** | **Word Count**: 75 words
* **Speech**:
  > "We also conducted a clinical error analysis. The model successfully filters healthy patients, identifying Normal Sinus Rhythms with ninety-one percent sensitivity. However, we observe that fifteen percent of Hypertrophy cases are misclassified as ST/T changes. This confusion is clinically logical: Left Ventricular Hypertrophy causes physical strain on the heart wall, resulting in lateral ST depression and T-wave inversion. These strain patterns mimic primary ischemic changes, creating a natural diagnostic overlap in morphology."
* **Presenter's Tip**: Highlight that this overlap represents a known clinical mimicry, not a random failure of the network.

---

### Slide 21: FastAPI Diagnostic Triage Dashboard
* **Visuals**: Left text on async microservice and GUI features, right card showing dashboard.
* **Timing**: **35 Seconds** | **Word Count**: 75 words
* **Speech**:
  > "For practical clinical utility, we deployed the models as an asynchronous FastAPI service. We designed a clean, light-theme triage dashboard to prevent clinical eye strain. Waveforms are plotted dynamically on familiar pink ECG thermal grids. The dashboard handles file uploads in JSON and CSV, runs the soft ensemble under calibrated thresholds in milliseconds, and highlights high-risk triage categories."
* **Presenter's Tip**: Mention that the dashboard operates in real-time, delivering predictions in under forty milliseconds.

---

### Slide 22: Scholarly Publication
* **Visuals**: Left card containing the article citation and right card showing CAJ paper side-by-side screenshots.
* **Timing**: **15 Seconds** | **Word Count**: 31 words
* **Speech**:
  > "Finally, the findings of this thesis have been published as an article in the Central Asian Journal, showing that our work contributes directly to regional clinical AI research."
* **Presenter's Tip**: Point to the screenshots on the right to show the published article.

---

### Slide 23: Conclusion
* **Visuals**: Numbered list of 7 key findings matching the 7 objectives.
* **Timing**: **30 Seconds** | **Word Count**: 65 words
* **Speech**:
  > "To conclude, all seven research objectives were successfully met: we built a zero-phase Butterworth preprocessor, established a strong ResNet morphological baseline, developed the learnable ST-ReGE GNN to capture lead correlations, evaluated Transformers, formulated the Spatio-Temporal Hybrid, calibrated decision boundaries to reach point-seven-five F1, and deployed an interactive FastAPI triage prototype."
* **Presenter's Tip**: Draw a clear line connecting this slide to the Objectives slide shown earlier.

---

### Slide 24: Future Directions
* **Visuals**: Bullet list outlining ensembling models, self-supervised pre-training, and multi-modal integration.
* **Timing**: **25 Seconds** | **Word Count**: 53 words
* **Speech**:
  > "Looking ahead, our future work will focus on three key directions. First, ensembling our models to combine morphological features, spatial pathways, and temporal contexts. Second, investigating self-supervised pre-training on larger databases like MIMIC-IV-ECG to enhance Transformer performance. And third, integrating demographic inputs for a complete multi-modal clinical decision support system."
* **Presenter's Tip**: Point out that these steps aim to translate our research into robust, hospital-ready tools.

---

### Slide 25: Thank You
* **Visuals**: Centered serif thank you text, split columns with candidate and supervisor names.
* **Timing**: **15 Seconds** | **Word Count**: 37 words
* **Speech**:
  > "This concludes my presentation. My sincere gratitude goes to Astana IT University and my supervisor, Dr. Minsoo Hahn, for their invaluable support. Thank you for your attention. I now welcome your questions."
* **Presenter's Tip**: Bow or nod slightly, speak with a warm, open posture, and prepare for the Q&A.
