# IEEE ML CHALLENGE: Fault Detection

This repository contains the solution pipeline for the **ML CHALLENGE by IEEE SB, GEHU**. The objective of this challenge is to build a highly accurate machine learning model to determine the operational status of an embedded device (Normal vs. Faulty) based on 47 numerical features derived from sensor measurements.

## 🚀 Novel Approach

To maximize the predictive power and achieve an Out-Of-Fold (OOF) accuracy and F1 score of **~0.985**, this solution utilizes a robust **Stacked Ensemble Pipeline** enriched with dynamic feature engineering. 

### 1. Advanced Feature Engineering
Instead of relying solely on the raw sensor inputs (`F01–F47`), we engineer deep relationships between the sensors:
*   **Row-wise Statistical Signatures:** We generate `mean`, `std`, `min`, `max`, `skew`, and `kurtosis` across all 47 features for each device. This captures the "global behavioral profile" of a device during its operational cycle.
*   **Dimensionality Reduction (PCA & ICA):** We apply Principal Component Analysis (PCA) and Independent Component Analysis (ICA) to compress the scaled 47 features into 10 new components. This technique excels at capturing holistic, multi-sensor anomalies that a single decision tree split might miss.

### 2. 5-Fold Stratified Ensemble
To combat variance and prevent overfitting on the imbalanced dataset (60% Normal, 40% Faulty):
*   We utilize a **Stratified K-Fold (5 Splits)** approach to guarantee the model is trained and validated on a statistically representative sample of faults.
*   **Tri-Model Blending:** The pipeline trains three State-Of-The-Art Gradient Boosting algorithms simultaneously:
    1.  **XGBoost:** Hyper-tuned with scaling parameters for class imbalance.
    2.  **LightGBM:** Optimized for rapid, leaf-wise tree growth.
    3.  **CatBoost:** Utilized for its superior handling of symmetric trees and automatic class weighting.
*   The final predictions are generated via **Soft-Voting (Averaging)** the validation probabilities of all three models, resulting in an incredibly stable threshold classification.

---

## 📂 Repository Structure
*   `TRAIN.csv`: Original training dataset (features `F01-F47` + `Class`). *(Excluded in git repo)*
*   `TEST.csv`: Original testing dataset. *(Excluded in git repo)*
*   `readme.txt`: Official challenge guidelines.
*   `solution.py`: The complete machine learning pipeline (Data Proc, Feature Engineering, K-Fold Ensembling, Evaluation, and Submission generation).
*   `submission.csv`: The final predicted output formatted as `ID, CLASS`.

## ⚙️ How to Run
Ensure you have the required Python libraries installed:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost
```

Place the `TRAIN.csv` and `TEST.csv` files in the root directory, then simply execute the pipeline:
```bash
python solution.py
```
This will automatically train the 5-Fold Ensemble, print out the OOF Metrics (AUC, Accuracy, F1), and generate the final `submission.csv` file.
