<div align="center">
  <img src="https://img.shields.io/badge/Challenge-alrIEEEna26-blue?style=for-the-badge&logo=ieee" alt="Challenge Badge">
  <img src="https://img.shields.io/badge/Model-ConvNeXt--Tiny-orange?style=for-the-badge&logo=pytorch" alt="Model Badge">
  <img src="https://img.shields.io/badge/Rank-Top--50-success?style=for-the-badge" alt="Rank Badge">

  <h1>🏢 ML Arena Finale: The "Noble" Pipeline</h1>
  <p><b>Fine-Grained Scene Recognition for 397 Categories using Modern Deep Learning Backbones</b></p>
</div>

<br>

## 📌 Approach Overview
This repository contains the complete **"Noble"** pipeline developed for the final stage of the **alrIEEEna26 ML Challenge**. Shifting from sensor-based diagnostics to **high-dimensional image classification**, this solution targeting 397 distinct scene categories (SUN397 benchmark).

By combining a state-of-the-art **ConvNeXt** backbone with advanced stochastic regularization and test-time augmentation, this pipeline achieves elite-level generalization across complex spatial contexts.

---

## 🧠 Architectural Deep-Dive

### 1. The Backbone: ConvNeXt-Tiny
We moved beyond traditional ResNets to a modern convolutional design inspired by Vision Transformers. 
- **Macro-Layer Scaling**: Adopts the layer-normalization and spatial-mixup patterns that allow for global receptive fields.
- **Efficiency**: Maintains the speed of CNNs while achieving Transformer-level accuracy on complex scene datasets.

### 2. Stochastic Regularization (Mixup & CutMix)
To prevent overfitting on the the challenge's intricate dataset, we implemented:
- **Mixup**: Linear interpolation of training samples to smooth decision boundaries.
- **CutMix**: Spatial substitution of image patches, forcing the model to learn localized, discriminative features (e.g., identifying a cathedral by its arches rather than global lighting).

### 3. Optimization Strategy
- **AdamW Optimizer**: Decouples weight decay from gradient updates for cleaner regularization.
- **OneCycleLR**: Enables **Super-Convergence**, peaking the learning rate for exploration and decaying for high-precision fine-tuning.

### 4. Robust Inference (TTA)
Our final predictions are generated via **Test-Time Augmentation (TTA)**. Each test image is evaluated 5 times under varied crops and flips, with the final softmax probabilities averaged to eliminate central-bias and localized occlusion errors.

---

## 📂 Folder Structure

| File | Description |
| :--- | :--- |
| `train_kaggle_master.py` | Full-scale training script with Mixup, CutMix, and TTA. |
| `master_notebook.md` | Modular cells optimized for Kaggle/Jupyter execution. |
| `train_fast_track_local.py` | MobileNetV3-based script for rapid local verification. |
| `master_approach.md` | Formal technical pitch and architectural evaluation. |
| `FINAL_LOCAL.csv` | Submission file generated from local environment. |

---

## 🚀 How to Run

```bash
# Execute the Master Training Pipeline
python train_kaggle_master.py
```

> [!IMPORTANT]
> Ensure `Config.DATA_DIR` is updated to point to your local path of `ML FINAL DATASET`.

---
*Developed by **Purvansh Joshi** & **Antigravity AI**.*
