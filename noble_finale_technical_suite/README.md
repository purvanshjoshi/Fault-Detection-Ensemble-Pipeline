<div align="center">
  <img src="https://img.shields.io/badge/Finale-Noble--Pipeline-blue?style=for-the-badge&logo=ieee" alt="Finale Badge">
  <img src="https://img.shields.io/badge/Architecture-ConvNeXt--Tiny-orange?style=for-the-badge&logo=pytorch" alt="Model Badge">
  <img src="https://img.shields.io/badge/Precision-High--Dimension-success?style=for-the-badge" alt="Precision Badge">

  <h1>🏢 Technical Finale: The "Noble" Pipeline</h1>
  <p><b>State-of-the-Art Scene Recognition for 397 Categories using Modern Deep Learning Backbones</b></p>
</div>

<br>

---

## 📌 Strategic Approach

This suite represents the **"Noble"** pipeline, the technical culmination of our engagement in the Machine Learning Challenge. Transitioning from 1D sensor telemetry to **high-dimensional spatial intelligence**, this solution targets the recognition of 397 distinct scene categories (SUN397 benchmark).

By synthesizing a **ConvNeXt** backbone with advanced stochastic regularization and multi-pass test-time augmentation, this pipeline achieves superior convergence and elite-level generalization.

---

## 🧠 Architectural Deep-Dive

### 1. The Backbone: ConvNeXt-Tiny
We leverage a modern convolutional design that bridges the gap between traditional CNNs and Vision Transformers.
- **Macro-Layer Scaling**: Implements layer-normalization and spatial-mixup patterns for expansive receptive fields.
- **Inverted Bottlenecks**: Optimized computational blocks for maximum feature preservation.

### 2. Stochastic Regularization
To combat overfitting on intricate scene textures, we deployed:
- **Mixup**: Synthesizes new training samples through linear interpolation of feature-target pairs.
- **CutMix**: Employs spatial substitution to force the model to focus on localized, discriminative features.

### 3. Optimization & Convergence
- **AdamW Optimizer**: Provides decoupled weight decay for surgical regularization.
- **OneCycleLR**: Drives **Super-Convergence**, enabling rapid exploration of the loss landscape followed by high-precision fine-tuning.

### 4. Robust Inference (TTA)
Our final entropy-minimized predictions are generated via **Test-Time Augmentation (TTA)**. Each image undergoes 5 transformations (varied crops/flips), with the softmax probabilities harmonized to eliminate central-bias.

---

## 📂 Suite Infrastructure

| Resource | Description |
| :--- | :--- |
| `train_kaggle_master.py` | Full-scale training script with Mixup, CutMix, and TTA orchestration. |
| `master_notebook.md` | Optimized modular segments for interactive Kaggle/Jupyter environments. |
| `train_fast_track_local.py` | Lightweight MobileNetV3-based script for immediate local validation. |
| `master_approach.md` | Detailed technical exposition and architectural rationale. |

---

## 🚀 Execution

```bash
# Execute the Master Training Pipeline
python train_kaggle_master.py
```

> [!IMPORTANT]
> Verify that `Config.DATA_DIR` in the training script points correctly to your local `ML FINAL DATASET` path.

---

<div align="center">
  <p><i>Developed with precision by <b>Purvansh Joshi</b> & <b>Antigravity AI</b>.</i></p>
</div>
