# Technical Briefing: ST-ReGE GNN and Spatio-Temporal Hybrid Models

This document provides a detailed, technical, and mathematical explanation of the two core custom models designed in this research: the **Spatio-Temporal Relation Graph Neural Network (ST-ReGE GNN)** and the **Spatio-Temporal Hybrid Model**. These notes are structured to serve as a comprehensive guide for explaining the models during your Master's thesis defense.

---

## Part 1: Spatio-Temporal Relation Graph GNN (ST-ReGE)

The **ST-ReGE GNN** is designed to capture **spatial inter-lead relationships** across standard 12-lead ECG electrode projections. Traditional deep learning models stack the 12 leads as independent parallel channels, ignoring the physical spatial geometry of electrode placements on the body. ST-ReGE resolves this by modeling the 12 leads as nodes in a graph and dynamically learning their anatomical/functional connections.

### 1. Mathematical Formulation

#### Node Feature Extraction (Weight-Shared 1D CNN)
Each raw ECG lead signal is processed by a weight-shared 1D CNN backbone ($f_{\text{CNN}}$) to extract high-level morphological features:
$$h_i^{(0)} = f_{\text{CNN}}(x_i), \quad h_i^{(0)} \in \mathbb{R}^{256}$$
Where $x_i \in \mathbb{R}^{5000}$ represents the $10$-second signal ($500\text{ Hz}$ sampling rate) of lead $i$ ($i \in \{1, \dots, 12\}$).

#### Learnable Adjacency Matrix ($A$)
Instead of using a static adjacency matrix based on physical Euclidean distances, the network dynamically learns functional connections via a trainable weight matrix $A_{\text{param}} \in \mathbb{R}^{12 \times 12}$ activated by a Sigmoid function:
$$A = \text{Sigmoid}(A_{\text{param}}), \quad A_{ij} \in (0, 1)$$
This ensures that the connection strength between Lead $i$ and Lead $j$ is dynamically adjusted via gradient descent during backpropagation, capturing co-occurring electrical abnormalities (e.g. inferior MI affecting leads II, III, and aVF simultaneously).

#### Residual Graph Convolutional Networks (GCN)
We stack two GCN layers with residual shortcut connections to propagate spatial features across leads:
* **Layer 1**:
  $$H^{(1)} = \text{BatchNorm}\left(\text{ReLU}\left(A \cdot H^{(0)} \cdot W^{(1)}\right)\right) + H^{(0)}$$
* **Layer 2**:
  $$H^{(2)} = \text{BatchNorm}\left(\text{ReLU}\left(A \cdot H^{(1)} \cdot W^{(2)}\right)\right) + H^{(1)}$$
Where $H^{(l)} \in \mathbb{R}^{B \times 12 \times 256}$ represents the lead-wise feature tensor, $W^{(l)} \in \mathbb{R}^{256 \times 256}$ is a trainable graph convolution weight matrix, and $B$ is the batch size. 

> [!IMPORTANT]
> The residual additions ($+ H^{(0)}$ and $+ H^{(1)}$) prevent **over-smoothing**, a common problem in GNNs where repeated feature propagation causes node representations to become completely identical and lose localized details.

---

### 2. Input and Output Tensor Shapes
* **Input Tensor**: $X \in \mathbb{R}^{B \times 12 \times 5000}$
* **Reshaping for Backbone**: $X_{\text{flat}} \in \mathbb{R}^{(B \times 12) \times 1 \times 5000}$ (Each lead of each record is treated as a separate sequence)
* **Backbone Output**: $X_{\text{feat\_flat}} \in \mathbb{R}^{(B \times 12) \times 256 \times 1}$ (Reduced via 1D convolutions and Adaptive Average Pooling)
* **Node Feature Representation ($H^{(0)}$)**: $H^{(0)} \in \mathbb{R}^{B \times 12 \times 256}$ (The 12 leads are now 256-dimensional node vectors)
* **Graph Convolution Output ($H^{(2)}$)**: $H^{(2)} \in \mathbb{R}^{B \times 12 \times 256}$ (Features propagated across lead networks)
* **Global Node Pooling**: $H_{\text{pooled}} \in \mathbb{R}^{B \times 256}$ (Mean pooled across the 12 leads: `out.mean(dim=1)`)
* **FC Classifier Head**: Output prediction probabilities $y \in \mathbb{R}^{B \times 5}$ (mapped to the 5 diagnostic superclasses)

---

### 3. Core Coding Components (`src/models/gnn.py`)
* **`LearnableAdjacency`**: Declares `self.A = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.01)` and returns `Sigmoid(self.A)` in the forward pass.
* **`GraphConvolution`**: Performs the spatial multiplication:
  ```python
  support = torch.matmul(x, self.weight)
  output = torch.matmul(adj, support)
  ```
* **`STReGE`**: Orchestrates the pipeline, permuting dimensions, executing the backbone, computing graph convolutions with residuals, and classification.

---

### 4. Simple Analogy for the Jury
> **The Social Network Analogy:**
> "Think of traditional CNNs as a group of doctors standing in a single straight line, where each doctor can only talk to their immediate neighbor on their left or right. Our ST-ReGE GNN acts like a **social network**: the electrodes dynamically learn who to add as 'friends' based on clinical data. For instance, leads II, III, and aVF on the inferior wall learn to share cardiac abnormality news with each other immediately, skipping physical distance boundaries."

---
---

## Part 2: Spatio-Temporal Hybrid Model (ResNet-Transformer)

The **Spatio-Temporal Hybrid Model** blends the local feature extraction strengths of a **Convolutional Neural Network (CNN)** with the global sequence modeling capabilities of a **Vision Transformer (ViT)**.

### 1. Motivation and Inductive Biases
* **The ViT Constraint**: Pure 1D Vision Transformers lack **locality bias** and **translation invariance**. Because they process sequences as a set of flat patches, they require massive datasets (millions of records) to learn local signal morphologies (like the slope of an ST-segment or the height of a P-wave).
* **The Hybrid Solution**: Stacking a CNN front-end acts as a morphological filter that handles local spatial and temporal details, downsampling the raw 5000-sample signal to 313 feature steps. This allows the Transformer encoder back-end to focus entirely on learning **long-range temporal rhythms** (such as rate variations or block structures).

---

### 2. Mathematical Formulation & Architecture Components

#### 1D ResNet Front-End (Morphology Filter)
The raw ECG input $X \in \mathbb{R}^{B \times 12 \times 5000}$ is transposed to $X \in \mathbb{R}^{B \times 12 \times 5000}$ (where channels are the leads). It passes through a 1D Convolution with stride 2 and max pooling, then 3 residual stages (1D ResNet blocks):
$$X_{\text{cnn}} = \text{Layer}_3(\text{Layer}_2(\text{Layer}_1(\text{Pool}(\text{Conv}(X))))), \quad X_{\text{cnn}} \in \mathbb{R}^{B \times 256 \times 313}$$
This extracts morphological shapes and downsamples the sequence length from $5000$ to $313$.

#### Squeeze-and-Excitation (SE) Block (Channel/Lead Attention)
The SE block recalibrates channel/lead feature importances. It squeezes spatial features via Global Average Pooling to a channel descriptor vector $z \in \mathbb{R}^{256}$:
$$z_c = \frac{1}{L} \sum_{i=1}^{L} x_{c}(i)$$
This vector passes through a 2-layer MLP bottleneck to generate scaling weights $s = \text{Sigmoid}(W_2 \cdot \text{ReLU}(W_1 \cdot z))$, which scale the original channels:
$$\tilde{X}_c = s_c \cdot X_c$$
This acts as a lead channel selection mechanism, emphasizing channels highlighting abnormalities.

#### Transformer Encoder Back-End (Rhythm Modeling)
* **Projection**: The 256 channels are projected to $d_{\text{model}} = 256$ using a $1 \times 1$ convolution, yielding a sequence tensor $X_{\text{seq}} \in \mathbb{R}^{B \times 313 \times 256}$.
* **CLS Token and Positional Embedding**: A learnable class token $[CLS]$ is prepended to the sequence, and 1D learnable positional embeddings ($PE \in \mathbb{R}^{314 \times 256}$) are added to retain temporal ordering:
  $$Z_0 = \left[ [CLS]; X_{\text{seq}} \right] + PE, \quad Z_0 \in \mathbb{R}^{B \times 314 \times 256}$$
* **Self-Attention Propagation**: The sequence passes through 3 standard Transformer Encoder layers with Multi-Head Self-Attention (8 heads, feedforward dimension 512):
  $$Z_l = \text{TransformerEncoderLayer}(Z_{l-1})$$
* **Classification**: The output feature of the $[CLS]$ token ($z_{\text{cls}} = Z_3[0] \in \mathbb{R}^{256}$) is normalized and fed into a 2-layer MLP classifier head with $0.2$ dropout.

---

### 3. Input and Output Tensor Shapes
* **Input Tensor**: $X \in \mathbb{R}^{B \times 5000 \times 12}$ (transposed inside the model to $B \times 12 \times 5000$)
* **CNN Output (after Stage 3)**: $X_{\text{cnn}} \in \mathbb{R}^{B \times 256 \times 313}$ (Sequence length compressed by factor of $\sim16$, channels increased to 256)
* **SE Block Output**: $X_{\text{se}} \in \mathbb{R}^{B \times 256 \times 313}$ (Lead importance scaled)
* **Transformer Input Projection**: $X_{\text{proj}} \in \mathbb{R}^{B \times 313 \times 256}$
* **Sequence with $[CLS]$ and $PE$**: $Z_0 \in \mathbb{R}^{B \times 314 \times 256}$
* **Transformer Output**: $Z_3 \in \mathbb{R}^{B \times 314 \times 256}$
* **extracted Class Token**: $z_{\text{cls}} \in \mathbb{R}^{B \times 256}$
* **FC Classifier Head**: Output prediction probabilities $y \in \mathbb{R}^{B \times 5}$

---

### 4. Simple Analogy for the Jury
> **The Spellchecker and Speed-Reader Analogy:**
> "Think of our Hybrid model as a text review team. The CNN acts as a **spellchecker**: it reads small, local groups of letters to check individual words (checking localized wave morphology, like QRS height or ST elevation). Once the spellchecker standardizes the words, it passes them to the Transformer, which acts as a **speed-reader**: it checks the overall grammar and flow of the entire page to understand the global rhythm and context (identifying long-term blocks or irregular intervals)."
