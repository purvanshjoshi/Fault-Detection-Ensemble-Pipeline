<div align="center">
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" alt="Status Badge">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn Badge">
  <img src="https://img.shields.io/badge/IEEE-00629B?style=for-the-badge&logo=ieee&logoColor=white" alt="IEEE Badge">

  <h1>🏭 Fault Diagnostics Ensemble Pipeline</h1>
  <p><b>An Advanced Machine Learning Architecture for High-Precision Hardware Anomaly Detection</b></p>
</div>

<br>

## 📌 Project Overview
Developed for the **IEEE SB, GEHU ML Challenge**, this repository houses a highly robust machine learning pipeline designed to monitor and diagnose embedded subsystem health. 

The core objective is binary classification: determining whether a device is operating under **Normal (0)** or **Faulty (1)** conditions based on an anonymized dataset of 47 numerical sensor readings. These readings capture real-time performance metrics, internal states, and environmental interactions.

Facing challenges like heavy class imbalance and isolated sensor noise, this solution pushes beyond standard classification. By engineered deeply contextual meta-features and deploying a triad of gradient-boosted models, this pipeline maximizes predictive stability and recall.

---

## 🧠 Core Methodology & Architecture

Our approach is built upon two distinct pillars: **Dynamic Feature Contextualization** and **Multi-Algorithmic Variance Reduction**.

### 1. Dynamic Feature Engineering
Standard decision trees evaluate singular sensors at precise thresholds. However, subsystem failures often manifest as subtle drift *across* multiple sensors simultaneously. To capture these cascading anomalies, we dynamically expanded the dataset:

*   📊 **Row-Wise Statistical Signatures**: 
    For every single operational cycle recorded, the pipeline computes the `mean`, `standard deviation`, `minimum`, `maximum`, `skewness`, and `kurtosis` across all 47 raw sensors. This creates a "global behavioral fingerprint" that allows the model to instantly detect if the hardware’s overall vibration, temperature, or voltage variance has spiked, regardless of individual sensor values.
*   📉 **Dimensionality Reduction (PCA & ICA)**: 
    We isolate macro-level variance by passing the standardized feature set through Principal Component Analysis (PCA) and Independent Component Analysis (ICA). This condenses the 47 sensors into 10 highly distinct meta-signals. PCA highlights the axes of maximum systemic stress, while ICA separates independent noise sources (like environmental interference) from genuine mechanical fault signals.

### 2. The 5-Fold Stratified Ensemble
Sensor data is notoriously noisy, and the provided dataset contained a significant class imbalance (approx. 60% Normal vs. 40% Faulty). To prevent our models from becoming biased toward the majority "Normal" class, we implemented a rigorously structured ensemble:

*   🔄 **Stratified K-Fold Cross Validation (K=5)**: 
    The training data is partitioned into 5 independent splits. The ratio of Normal to Faulty rows is strictly preserved in every split. The models train on 4 folds and validate on the 5th, ensuring they are only evaluated on data they have entirely never seen. This prevents data leakage and guarantees highly accurate performance metrics.
*   ⚙️ **Tri-Algorithmic Blending**: 
    No single algorithm is perfect. We train three distinct, state-of-the-art tree architectures simultaneously to compensate for each other's blind spots:
    1.  **XGBoost (Extreme Gradient Boosting)**: Heavily regularized (Depth=6) and configured with explicit `scale_pos_weight` to aggressively penalize the model when it misclassifies minority fault events.
    2.  **LightGBM**: Utilized for its highly efficient leaf-wise growth, which excels at finding optimal splits within the dense matrix of purely statistical PCA/ICA features.
    3.  **CatBoost**: Applied as the ensemble's stabilizing anchor. Its symmetric tree structures and automated target-based weighting make it exceptionally resistant to the overfitting that traditional gradient boosters suffer from.
*   🤝 **Soft Voting Mechanism**: 
    The models do not output hard 0s and 1s during evaluation. Instead, they output the exact mathematical *probability* of a fault occurring. The pipeline averages the probabilities of XGBoost, LightGBM, and CatBoost together before making the final 0/1 decision.

---

## 📂 Repository Structure

| File | Type | Description |
| :--- | :--- | :--- |
| `solution.py` | `<Python Script>` | The complete autonomous ML pipeline (Preprocessing -> PCA/ICA -> K-Fold Training -> Blending -> Inference). |
| `readme.txt`  | `<Documentation>` | The official ML Challenge dataset instructions and problem context guidelines. |
| `FINAL.csv` | `<Data File>` | The successfully generated prediction matrix formatted precisely for competition evaluation (`ID, CLASS`). |
| `.gitignore` | `<Config>` | Prevents tracking of the massive `TRAIN.csv` and `TEST.csv` files to keep the repository lightweight. |

> [!NOTE] 
> Due to GitHub file size limits and challenge policies, the raw training and testing datasets (`TRAIN.csv`, `TEST.csv`) are exclusively retained locally and are not hosted in this repository.

---

## 🚀 Execution Instructions

To execute the pipeline locally and reproduce the final prediction metrics, ensure you have a Python environment configured with the prerequisite machine learning libraries:

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm catboost

# 2. Place datasets
# Ensure TRAIN.csv and TEST.csv are placed securely in the repository's root directory.

# 3. Execute the Ensemble
python solution.py
```

The script is entirely autonomous. It will systematically output the training loss of each fold, calculate final Out-Of-Fold (OOF) evaluation metrics (Accuracy, ROC AUC, F1), and instantly generate the compliant `FINAL.csv` file.
