# ML Arena Finale - Top 50 Teams Approach
This folder contains the complete "Noble" pipeline developed for the alrIEEEna26 ML Challenge.

## Contents
- **`train_kaggle_master.py`**: The full-scale training script for the 397-class ConvNeXt model.
- **`master_notebook.md`**: Modular cells optimized for Kaggle/Jupyter environments.
- **`train_fast_track_local.py`**: The localized MobileNetV3 script for rapid environment verification.
- **`master_approach.md`**: Formal technical pitch and architectural deep-dive for evaluators.
- **`FINAL_LOCAL.csv`**: The submission file generated from the local training run.

## Key Techniques
- **Architecture**: ConvNeXt-Tiny (Master) / MobileNetV3 (Local)
- **Regularization**: Mixup, CutMix, Label Smoothing
- **Optimization**: AdamW + OneCycleLR
- **Inference**: Test-Time Augmentation (TTA) with 5 versions per image.

---
*Developed by Purvansh Joshi & Antigravity AI.*
