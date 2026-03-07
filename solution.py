import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
train = pd.read_csv('TRAIN.csv')
test = pd.read_csv('TEST.csv')

features = [c for c in train.columns if c != 'Class']
target = 'Class'

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Concatenate for uniform feature engineering
all_data = pd.concat([train[features], test[features]], axis=0).reset_index(drop=True)

# 1. Feature Engineering: Row-wise statistics
print("Generating row-wise statistical features...")
all_data['std'] = all_data[features].std(axis=1)
all_data['mean'] = all_data[features].mean(axis=1)
all_data['max'] = all_data[features].max(axis=1)
all_data['min'] = all_data[features].min(axis=1)
all_data['skew'] = all_data[features].skew(axis=1)
all_data['kurt'] = all_data[features].kurtosis(axis=1)

# 2. Scaling
print("Scaling features...")
scaler = StandardScaler()
scaled_features = scaler.fit_transform(all_data)
scaled_df = pd.DataFrame(scaled_features, columns=all_data.columns)

# 3. Dimensionality Reduction Features (PCA & ICA)
print("Generating PCA and ICA features...")
pca = PCA(n_components=5, random_state=42)
pca_features = pca.fit_transform(scaled_features)
for i in range(5):
    scaled_df[f'pca_{i}'] = pca_features[:, i]

ica = FastICA(n_components=5, random_state=42)
ica_features = ica.fit_transform(scaled_features)
for i in range(5):
    scaled_df[f'ica_{i}'] = ica_features[:, i]

# Split back to train and test
X_train = scaled_df.iloc[:len(train)].copy()
y_train = train[target].values
X_test = scaled_df.iloc[len(train):].copy()

test_ids = test['ID'].values

print(f"Final Train shape: {X_train.shape}, Final Test shape: {X_test.shape}")

# 4. Model Training with K-Fold Cross Validation
folds = 5
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

# Initializing Out-Of-Fold and Test prediction arrays
xgb_preds = np.zeros(len(X_test))
lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

xgb_oof = np.zeros(len(X_train))
lgb_oof = np.zeros(len(X_train))
cat_oof = np.zeros(len(X_train))

# Compute class weight for imbalance
pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

print(f"Starting {folds}-Fold Training...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"--- Fold {fold+1} ---")
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    # 1. XGBoost
    xgb_model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=42 + fold,
        early_stopping_rounds=50,
        n_jobs=-1
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_oof[val_idx] = xgb_model.predict_proba(X_va)[:, 1]
    xgb_preds += xgb_model.predict_proba(X_test)[:, 1] / folds
    
    # 2. LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42 + fold,
        n_jobs=-1
    )
    # LGBM early stopping handles differently
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[]) # simple fit, LGBM early stopping needs callbacks but let's just train
    lgb_oof[val_idx] = lgb_model.predict_proba(X_va)[:, 1]
    lgb_preds += lgb_model.predict_proba(X_test)[:, 1] / folds

    # 3. CatBoost
    cat_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        auto_class_weights='Balanced',
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    cat_oof[val_idx] = cat_model.predict_proba(X_va)[:, 1]
    cat_preds += cat_model.predict_proba(X_test)[:, 1] / folds

print("Evaluating OOF (Out-of-Fold) Predictions...")
for name, oof in zip(['XGB', 'LGB', 'CAT'], [xgb_oof, lgb_oof, cat_oof]):
    score = roc_auc_score(y_train, oof)
    acc = accuracy_score(y_train, (oof > 0.5).astype(int))
    print(f"{name} -> ROC AUC: {score:.5f} | Accuracy: {acc:.5f}")

# Ensemble (Average Blending)
ensemble_oof = (xgb_oof + lgb_oof + cat_oof) / 3
ensemble_auc = roc_auc_score(y_train, ensemble_oof)
ensemble_acc = accuracy_score(y_train, (ensemble_oof > 0.5).astype(int))
f1 = f1_score(y_train, (ensemble_oof > 0.5).astype(int))
print(f"ENSEMBLE -> ROC AUC: {ensemble_auc:.5f} | Accuracy: {ensemble_acc:.5f} | F1: {f1:.5f}")

# Generate Final Predictions
ensemble_test_preds = (xgb_preds + lgb_preds + cat_preds) / 3
final_classes = (ensemble_test_preds > 0.5).astype(int)

# Create Submission File
sub = pd.DataFrame({
    'ID': test_ids,
    'CLASS': final_classes
})
sub.to_csv('FINAL.csv', index=False)
print("Submission saved to 'FINAL.csv'!")
