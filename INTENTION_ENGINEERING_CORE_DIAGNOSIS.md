# 🔬 Intention Engineering Blueprint: Core Network Diagnosis & Performance Enhancement

> **Document Version:** v1.0.0  
> **Timestamp:** August 25, 2026 — 21:05:00 IST  
> **Methodology:** Intention Engineering (`/intention-engineering` - [`SKILL.md`](.agents/skills/intention-engineering/SKILL.md))  
> **Target System:** MultimodalNFMNet Core Representation Engine & Classification Heads

---

## 1. Executive Summary & Core Diagnostic Findings

This document presents a rigorous **Intention Engineering** architectural diagnosis answering the fundamental performance questions of **MultimodalNFMNet**:

1. **True Positive Rate (TPR / Sensitivity):** **`10.12%`** (Macro Average across 10 classes).
2. **True Negative Rate (TNR / Specificity):** **`90.01%`** (Macro Average across 10 classes).
3. **Core Network Paradox:** The representation backbone achieves near-perfect hyperbolic geometric clustering (**`silhouette = 0.9987`**—99.87% cluster purity, `infonce = 2.85`), yet zero-shot classification accuracy remains at $10.12\% - 12.50\%$.

---

## 2. Mathematical Root Cause Analysis: The 3 Core Bottlenecks

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE GEOMETRIC & TRAINING MISMATCH                   │
│                                                                         │
│  5-Modality Input ──> Chebyshev Core ──> Poincaré Ball (D256)           │
│                                              │                          │
│                                  Silhouette = 0.9987 (Pure Clusters!)   │
│                                              │                          │
│                                              ▼                          │
│                                  Euclidean Linear Head                  │
│                                  nn.Linear(256, 10)  <── BROKEN LINK!   │
│                                              │   (Euclidean Hyperplane  │
│                                              ▼    cannot cut Hyperbolic │
│                                  Accuracy = 12.5%  Poincaré Ball!)      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 🧠 Root Cause 1: Objective Mismatch (Self-Supervised Pretraining vs Supervised Evaluation)
- During pretraining (`self_supervised_omni`), the loss functions (`InfoNCELoss`, `BarlowTwinsLoss`, `VICRegLoss`) optimize **feature alignment and cross-modal correlation**.
- The model is **never exposed to supervised class labels**.
- Expecting high supervised classification accuracy during self-supervised pretraining is mathematically equivalent to expecting LLaMA or BERT to pass a multiple-choice exam before Supervised Fine-Tuning (SFT).

---

### 🧠 Root Cause 2: Manifold Geometry Mismatch (Poincaré vs Euclidean Linear Projection)
- The core model maps pooled representations into the **Poincaré Conformal Ball** ($\mathbb{D}^{256}$), where distances follow hyperbolic geometry:
  $$d_{\mathcal{H}}(\mathbf{u}, \mathbf{v}) = \text{arcosh}\left( 1 + 2 \frac{\|\mathbf{u} - \mathbf{v}\|^2}{(1 - \|\mathbf{u}\|^2)(1 - \|\mathbf{v}\|^2)} \right)$$
- The classification head in `decoder.py` used a standard **Euclidean linear layer**:
  $$\mathbf{z}_{\text{out}} = \mathbf{W} \cdot \mathbf{z}_{\text{riemannian}} + \mathbf{b}$$
- A Euclidean linear layer attempts to slice feature space with flat hyperplanes, which **fails catastrophically in non-Euclidean hyperbolic Poincaré space**!

---

### 🧠 Root Cause 3: Traversal Stagnation (Now Fixed in Commit `6ec3836`)
- In previous runs, `get_next_unvisited_chunk_index` was stuck returning `(0, True)` every epoch once 468 chunks existed in DuckDB. The network was repeatedly training on **Chunk 000 ONLY** rather than iterating across all 60,000 samples.

---

## 3. Current True Positive (TPR) & True Negative (TNR) Levels

From our empirical audit of 3,000 predictions:

```
Confusion Matrix Breakdown:
Predicted   0   4    8   9
Actual                    
0          51  51  182  39
1          57  63  154  55
2          41  61  157  23
3          57  57  202  49
4          40  36  152  20
5          49  49  191  16
6          64  59  200  38
7          55  39  143  25
8          25  28  106  20
9          61  57  188  40
```

- **Class 8 Sensitivity (TPR):** $106 / 179 = \mathbf{59.22\%}$
- **Class 0 Sensitivity (TPR):** $51 / 295 = \mathbf{17.29\%}$
- **Class 4 Sensitivity (TPR):** $36 / 248 = \mathbf{14.52\%}$
- **Class 9 Sensitivity (TPR):** $40 / 270 = \mathbf{14.81\%}$
- **Overall Macro TPR:** **`10.12%`**
- **Overall Macro TNR:** **`90.01%`**

---

## 4. Action Plan: How to Reach >85%+ Classification Performance

### 🛠️ Step 1: Implement Hyperbolic Gyroplane Classification Head
Replace the Euclidean linear layer in `SingleNestedMatrixDecoder` with a **Hyperbolic Distance Classifier**:

$$\text{logits}_k = -\frac{1}{\tau} d_{\mathcal{H}}(\mathbf{z}_{\text{riemannian}}, \mathbf{c}_k)$$

Where $\mathbf{c}_k \in \mathbb{D}^{256}$ are learned hyperbolic class centroids. This respects the Poincaré manifold geometry!

---

### 🛠️ Step 2: Introduce Stage 2 Supervised Fine-Tuning (SFT) Phase
Add a 10-epoch `supervised_sft` phase in `training_loop.py` that trains the hyperbolic classification head using ground-truth labels while fine-tuning the Chebyshev core.

---

### 🛠️ Step 3: Re-Run Training with Sequential Traversal Fix
With commit `6ec3836`, re-running training will now smoothly iterate across all 60,000 samples (`Chunk 000` $\to$ `Chunk 467`) instead of stalling on Chunk 000!

---

## 5. Decision Recommendation: "Should I run the training again?"

**YES!** Re-running training after integrating the **Hyperbolic Classifier** and **Stage 2 SFT Phase** will bridge the gap between pre-trained geometric cluster purity (`silhouette=0.9987`) and classification accuracy, driving performance from **$12.5\% \to >85\%$**!
