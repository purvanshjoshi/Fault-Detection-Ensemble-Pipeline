<div align="center">
  <img src="https://img.shields.io/badge/Suite-Noble--Ensemble-gold?style=for-the-badge&logo=icloud&logoColor=white" alt="Suite Badge">
  <img src="https://img.shields.io/badge/Status-Technical--Finale-success?style=for-the-badge" alt="Status Badge">
  <img src="https://img.shields.io/badge/Core-Advanced--ML-blue?style=for-the-badge&logo=gitbook&logoColor=white" alt="Core Badge">
  <img src="https://img.shields.io/badge/AI-Antigravity--Optimized-9C27B0?style=for-the-badge" alt="AI Badge">

  <h1>🏆 Noble-Ensemble Technical Suite</h1>
  <p><b>Unified High-Precision Intelligence for Hardware Diagnostics & Scenario Recognition</b></p>
</div>

<br>

---

## 🚀 Global Overview

Welcome to the **Noble-Ensemble Technical Suite**, a comprehensive machine learning ecosystem developed for a high-stakes **Machine Learning Challenge**. This repository represents the convergence of two distinct yet complementary intelligence pipelines: **Predictive Sensor Analytics** and **High-Dimensional Image Classification**.

By orchestrating advanced feature contextualization, multi-algorithmic ensembles, and state-of-the-art vision backbones, this suite delivers industry-grade predictive stability and generalization across heterogeneous datasets.

---

## 🧠 Integrated Intelligence

The suite is divided into two specialized pillars, each addressing a unique domain of technical failure and recognition:

### 1. Fault Diagnostics (Sensor-Based)
*   **Domain**: Binary classification of hardware health (Normal vs. Faulty).
*   **Input**: 47 anonymized sensor readings capturing real-time hardware telemetry.
*   **Strategy**: A 5-Fold Stratified Ensemble of XGBoost, LightGBM, and CatBoost, powered by PCA/ICA feature decomposition.

### 2. "Noble" Vision Suite (Image-Based)
*   **Domain**: Fine-grained scene categorization (SUN397).
*   **Input**: High-resolution spatial data across 397 distinct scene categories.
*   **Strategy**: Vision-Transformer inspired **ConvNeXt** backbone with Mixup/CutMix regularization and Test-Time Augmentation (TTA).

---

## 📂 Core Architecture

| Component | Path | Description |
| :--- | :--- | :--- |
| **Ensemble Pipeline** | [solution.py](file:///d:/ML_Arena/Fault-Detection-Ensemble-Pipeline/solution.py) | Autonomous ML pipeline for sensor-based diagnostics. |
| **Noble Suite** | [noble_finale_technical_suite/](file:///d:/ML_Arena/Fault-Detection-Ensemble-Pipeline/noble_finale_technical_suite) | The "Technical Finale" image classification infrastructure. |
| **Generated Results** | [FINAL.csv](file:///d:/ML_Arena/Fault-Detection-Ensemble-Pipeline/FINAL.csv) | Formatted prediction matrix for evaluation. |
| **Technical Context** | [readme.txt](file:///d:/ML_Arena/Fault-Detection-Ensemble-Pipeline/readme.txt) | Original challenge constraints and dataset specifications. |

---

## 🛠️ Technical Implementation

### **Pillar I: Sensor Diagnostics**
We utilize **Dynamic Feature Contextualization** to capture subtle drift across multiple sensors:
- 📊 **Row-Wise Statistics**: Generates mean, std, skew, and kurtosis to identify systemic stress.
- 📉 **Dimensionality Reduction**: PCA & ICA isolate noise from genuine mechanical fault signals.
- 🤝 **Soft Voting**: Harmonizes XGBoost (minority penalty), LightGBM (leaf-wise growth), and CatBoost (overfit resistance).

### **Pillar II: "Noble" Image Classification**
A modern deep-learning approach inspired by recent architectural breakthroughs:
- 🏢 **ConvNeXt Backbone**: Combines CNN efficiency with Transformer global fields.
- 🔄 **Stochastic Regularization**: Mixup and CutMix for superior generalization.
- 📈 **OneCycleLR**: Enables super-convergence for high-precision fine-tuning.

---

## 🚀 Execution Guide

Ensure your environment is configured with `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `torch`, and `torchvision`.

### Running Fault Diagnostics
```bash
python solution.py
```

### Running the Noble Vision Suite
```bash
cd noble_finale_technical_suite
python train_kaggle_master.py
```

---

<div align="center">
  <p><i>Developed with precision by <b>Purvansh Joshi</b> & <b>Antigravity AI</b>.</i></p>
  <img src="https://img.shields.io/badge/Engineered-with-E0E0E0?style=flat-square&logo=visual-studio-code&logoColor=007ACC" alt="Engineered">
</div>
