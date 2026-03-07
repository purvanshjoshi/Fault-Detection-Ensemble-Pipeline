<div align="center">
  <h1>🏭 Fault Detection Ensemble Pipeline</h1>
  <p><b>An Advanced Machine Learning Architecture for High-Precision Sub-System Fault Diagnostics</b></p>
</div>

<br>

## 📌 Project Overview
The **Fault Detection Ensemble Pipeline** is a robust, state-of-the-art machine learning solution engineered for the **IEEE SB, GEHU ML Challenge**. The core objective of this project is to determine the precise operational status of an embedded device—classifying it as either **Normal (0)** or **Faulty (1)**—by analyzing 47 anonymized numeric features representing real-time internal state, performance metrics, and environmental interactions.

By leveraging advanced dimensionality reduction, row-wise statistical profiling, and a multi-algorithmic ensemble architecture, this pipeline achieves a highly stable **Out-of-Fold (OOF) Accuracy and F1 Score of ~0.985**.

---

## 🧠 Methodology & Architecture

Our approach moves beyond basic classification by dynamically generating a holistic "behavioral profile" for each operational cycle, feeding this enriched data into a triad of specialized Gradient Boosted Models.

### 1. Dynamic Feature Engineering
Instead of relying solely on individual sensor readings, the system engineers deep, localized context for every single operational event:
*   **Row-Wise Statistical Signatures**: For every row (device cycle), we compute the `mean`, `std`, `min`, `max`, `skew`, and `kurtosis` across all 47 features. This mathematically captures the overall variation, asymmetry, and extremes of the hardware's operating state in that exact moment.
*   **Dimensionality Reduction (PCA & ICA)**: Complex failure modes often manifest as subtle anomalies spanning *multiple* sensors simultaneously. To detect this, we apply Principal Component Analysis (PCA) and Independent Component Analysis (ICA) to the scaled feature set. We extract 10 highly-compressed, orthogonal meta-features that isolate global variance patterns and independent noise signals that standard decision trees would struggle to split on individually.

### 2. The 5-Fold Stratified Ensemble
Single models are prone to high variance and overfitting, especially given the class imbalance present in tabular sensor data (60% Normal / 40% Faulty). To maximize generalization on unseen test data, we deployed an ensemble:
*   **Stratified K-Fold Cross Validation (K=5)**: The training data is split into 5 distinct folds, ensuring the ratio of Normal to Faulty devices remains perfectly consistent across every split. The models are trained on 4 folds and validated on the 5th, guaranteeing that out-of-fold metrics are rigorous and reliable.
*   **Tri-Algorithmic Blending**: Three distinct architectures are trained in parallel to capture different patterns in the structured data:
    1.  **XGBoost (Extreme Gradient Boosting)**: Tuned with severe Depth regularization (`max_depth=6`), Subsampling (`0.8`), and `scale_pos_weight` to aggressively penalize misclassified minority faults.
    2.  **LightGBM**: Chosen for its rapid, leaf-wise growth strategy, handling the dense matrix of statistical and PCA features highly efficiently. Weighted using internal `class_weight='balanced'`.
    3.  **CatBoost**: Applied for its exceptional handling of oblivious trees and automated regularization. It acts as the stabilizing force within the ensemble.
*   **Soft Voting Mechanism**: The final operational status prediction is derived not by taking a hard majority vote, but by **averaging the probability logits** of all three models. 

<br>

---

## 📂 Repository Structure

| File | Description |
| ---- | ----------- |
| `solution.py` | The complete ML pipeline encompassing preprocessing, PCA/ICA extraction, model training, K-Fold loops, and inference logic. |
| `readme.txt`  | Official dataset instructions and problem statement as provided by the ML Challenge. |
| `FINAL.csv` | The generated prediction file formatted precisely for the competition evaluation (`ID, CLASS`). |
| `.gitignore` | Excludes extremely large `TRAIN.csv`/`TEST.csv` dataset blobs from version control. |

*Note: Due to size constraints, the raw training and testing datasets (`TRAIN.csv`, `TEST.csv`) are not included in this repository. They must be placed in the project root to run the pipeline.*

---

## 🚀 Execution & Usage

To reproduce the ~0.985 F1-Score results, ensure your environment has the following modern data science libraries:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost
```

Once the `TRAIN.csv` and `TEST.csv` files are in your directory, execute the autonomous pipeline:
```bash
python solution.py
```
The script will sequentially output the training progress of each fold, calculate final ensemble validation metrics, and automatically generate the compliant `FINAL.csv` output.
