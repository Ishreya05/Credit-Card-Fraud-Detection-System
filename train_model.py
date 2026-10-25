"""
FraudShield — Model Training Script
=====================================
Run this ONCE to train and save the fraud detection model.

Usage:
    python train_model.py

Requirements:
    pip install scikit-learn pandas numpy imbalanced-learn joblib

Dataset:
    Download from https://www.kaggle.com/mlg-ulb/creditcardfraud
    Place 'creditcard.csv' in the same folder as this script.
    
    If you don't have the CSV yet, the script generates a
    synthetic stand-in so the app still works immediately.
"""

import os, joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline

os.makedirs("model", exist_ok=True)
CSV_PATH = "creditcard.csv"

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
if os.path.exists(CSV_PATH):
    print("📂 Loading Kaggle creditcard.csv …")
    df = pd.read_csv(CSV_PATH)
    print(f"   Rows: {len(df):,}  |  Fraud: {df['Class'].sum():,}  ({df['Class'].mean()*100:.3f}%)")

    # Features used by FraudShield UI
    FEATURES = ["V14", "V17", "V1", "V3", "Amount"]
    X = df[FEATURES].values
    y = df["Class"].values

else:
    print("⚠️  creditcard.csv not found — generating synthetic data for development.")
    print("    Download the real dataset from Kaggle and re-run this script.\n")

    rng = np.random.default_rng(42)
    n_legit = 28_000
    n_fraud = 500

    legit = np.column_stack([
        rng.normal(0, 1, n_legit),    # V14
        rng.normal(0, 1, n_legit),    # V17
        rng.normal(0, 1, n_legit),    # V1
        rng.normal(0, 1, n_legit),    # V3
        rng.exponential(100, n_legit) # Amount
    ])

    fraud = np.column_stack([
        rng.normal(-6, 1.5, n_fraud),  # V14 very negative
        rng.normal(-6, 1.5, n_fraud),  # V17 very negative
        rng.normal(-2, 1.0, n_fraud),  # V1
        rng.normal(-1, 1.0, n_fraud),  # V3
        rng.uniform(1, 5000, n_fraud)  # Amount varied
    ])

    X = np.vstack([legit, fraud])
    y = np.array([0]*n_legit + [1]*n_fraud)
    print(f"   Synthetic rows: {len(X):,}  |  Fraud: {n_fraud}")


# ─────────────────────────────────────────────────────────────
# 2. HANDLE CLASS IMBALANCE WITH SMOTE
# ─────────────────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    print("\n⚖️  Applying SMOTE oversampling …")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"   After SMOTE — Legit: {(y_res==0).sum():,}  Fraud: {(y_res==1).sum():,}")
except ImportError:
    print("\n⚠️  imbalanced-learn not installed — skipping SMOTE (using class_weight='balanced')")
    X_res, y_res = X, y


# ─────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)
print(f"\n📊 Train: {len(X_train):,}  |  Test: {len(X_test):,}")


# ─────────────────────────────────────────────────────────────
# 4. TRAIN MODELS & PICK BEST
# ─────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, C=1.0, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    ),
}

best_name, best_auc, best_model = None, -1, None
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

for name, clf in models.items():
    print(f"\n🔧 Training {name} …")
    clf.fit(X_train_s, y_train)
    y_prob = clf.predict_proba(X_test_s)[:, 1]
    y_pred = clf.predict(X_test_s)
    auc    = roc_auc_score(y_test, y_prob)
    print(f"   ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], digits=4))
    cm = confusion_matrix(y_test, y_pred)
    print(f"   Confusion Matrix:\n{cm}\n")
    if auc > best_auc:
        best_auc, best_name, best_model = auc, name, clf

print(f"\n🏆 Best model: {best_name}  (AUC={best_auc:.4f})")


# ─────────────────────────────────────────────────────────────
# 5. SAVE PIPELINE (scaler + model)
# ─────────────────────────────────────────────────────────────
pipeline = Pipeline([("scaler", scaler), ("clf", best_model)])
# Refit on full training set via pipeline
pipeline.fit(X_train, y_train)  # scaler already fit, but pipeline handles it

# Save artefacts
joblib.dump(pipeline, "model/fraud_model.pkl")
joblib.dump({
    "model_name":  best_name,
    "roc_auc":     round(best_auc, 4),
    "features":    ["V14", "V17", "V1", "V3", "Amount"],
    "trained_on":  len(X_train),
    "test_size":   len(X_test),
    "smote_used":  "imblearn" in dir(),
}, "model/model_info.json")

print("\n✅  Saved  →  model/fraud_model.pkl")
print("✅  Saved  →  model/model_info.json")
print("\nRestart app.py to use the trained model.")
