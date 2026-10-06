"""
Trains all 5 candidate models on the identical held-out test split, scores
them, promotes the best one by ROC-AUC, and saves every artifact the API
needs at inference time.

Run:
    python -m app.ml.train

Produces (in backend/models/):
    logistic_regression.pkl, decision_tree.pkl, random_forest.pkl,
    xgboost.pkl, svm.pkl
    scaler.pkl, impute_values.json, outlier_caps.json
    metrics.json          <- leaderboard shown on the Overview tab
    best_model.json        <- which model is currently promoted + why
    shap_global.json       <- global feature importance for the Explainability tab
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from app.ml.data import load_dataset, FEATURE_COLUMNS, TARGET_COLUMN
from app.ml.preprocess import fit_preprocessing, apply_preprocessing, fit_scaler, scale_features
from app.ml.explain import compute_global_shap

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False


def build_xgb_model():
    if HAS_XGBOOST:
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    # Fallback so the pipeline still runs end-to-end without the xgboost package
    print("  [!] xgboost not installed -- substituting sklearn GradientBoostingClassifier "
          "for the ensemble slot. `pip install xgboost` for the real model.")
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42)


def evaluate(model, X, y_true) -> dict:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 1),
        "precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 1),
        "recall": round(recall_score(y_true, y_pred, zero_division=0) * 100, 1),
        "f1": round(f1_score(y_true, y_pred, zero_division=0) * 100, 1),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 3),
    }


def main():
    t0 = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("1/5  Loading dataset...")
    df, is_real = load_dataset()
    print(f"     {'REAL Kaggle' if is_real else 'SYNTHETIC'} data: {len(df):,} rows, "
          f"default rate {df[TARGET_COLUMN].mean()*100:.2f}%")

    print("2/5  Preprocessing (impute -> cap outliers -> drop invalid ages -> 80/20 split)...")
    train_df, test_df = train_test_split(
        df, test_size=0.20, stratify=df[TARGET_COLUMN], random_state=42
    )
    prep = fit_preprocessing(train_df)
    train_df = apply_preprocessing(train_df, prep, drop_invalid_age=True)
    test_df = apply_preprocessing(test_df, prep, drop_invalid_age=True)

    X_train_raw, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test_raw, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    scaler = fit_scaler(X_train_raw)
    X_train_scaled = scale_features(X_train_raw, scaler)
    X_test_scaled = scale_features(X_test_raw, scaler)

    print(f"     Train: {len(train_df):,} | Test: {len(test_df):,}")

    metrics = {}

    print("3/5  Training 5 candidate models...")

    print("     - Logistic Regression (scaled features)")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_scaled, y_train)
    metrics["Logistic Regression"] = evaluate(lr, X_test_scaled, y_test)
    joblib.dump(lr, MODELS_DIR / "logistic_regression.pkl")

    print("     - Decision Tree (raw features)")
    dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, class_weight="balanced", random_state=42)
    dt.fit(X_train_raw, y_train)
    metrics["Decision Tree"] = evaluate(dt, X_test_raw, y_test)
    joblib.dump(dt, MODELS_DIR / "decision_tree.pkl")

    print("     - Random Forest (raw features)")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train_raw, y_train)
    metrics["Random Forest"] = evaluate(rf, X_test_raw, y_test)
    joblib.dump(rf, MODELS_DIR / "random_forest.pkl")

    print("     - XGBoost (raw features)")
    xgb_model = build_xgb_model()
    if HAS_XGBOOST:
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        xgb_model.set_params(scale_pos_weight=scale_pos_weight)
    xgb_model.fit(X_train_raw, y_train)
    metrics["XGBoost"] = evaluate(xgb_model, X_test_raw, y_test)
    joblib.dump(xgb_model, MODELS_DIR / "xgboost.pkl")

    print("     - SVM / RBF (scaled features, stratified subsample: full SVM on 150k rows is impractical)")
    svm_train_df, _ = train_test_split(
        train_df, train_size=min(8000, len(train_df)), stratify=train_df[TARGET_COLUMN], random_state=42
    ) if len(train_df) > 8000 else (train_df, None)
    X_svm_raw = svm_train_df[FEATURE_COLUMNS]
    y_svm = svm_train_df[TARGET_COLUMN]
    X_svm_scaled = scale_features(X_svm_raw, scaler)
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=42)
    svm.fit(X_svm_scaled, y_svm)
    metrics["SVM (RBF)"] = evaluate(svm, X_test_scaled, y_test)
    joblib.dump(svm, MODELS_DIR / "svm.pkl")

    print("4/5  Selecting best model by ROC-AUC...")
    best_name = max(metrics, key=lambda k: metrics[k]["roc_auc"])
    print(f"     Best model: {best_name}  (ROC-AUC {metrics[best_name]['roc_auc']})")

    best_model_map = {
        "Logistic Regression": (lr, "linear"),
        "Decision Tree": (dt, "tree"),
        "Random Forest": (rf, "forest"),
        "XGBoost": (xgb_model, "xgboost"),
        "SVM (RBF)": (svm, "kernel"),
    }
    best_model, best_kind = best_model_map[best_name]

    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODELS_DIR / "best_model.json").write_text(json.dumps({
        "name": best_name,
        "kind": best_kind,
        "roc_auc": metrics[best_name]["roc_auc"],
        "file": {
            "Logistic Regression": "logistic_regression.pkl",
            "Decision Tree": "decision_tree.pkl",
            "Random Forest": "random_forest.pkl",
            "XGBoost": "xgboost.pkl",
            "SVM (RBF)": "svm.pkl",
        }[best_name],
        "uses_scaled_features": best_name in ("Logistic Regression", "SVM (RBF)"),
        "trained_on_real_data": is_real,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "default_rate_pct": round(float(df[TARGET_COLUMN].mean()) * 100, 2),
    }, indent=2))

    print("5/5  Computing global SHAP feature importance for the promoted model...")
    if best_kind in ("tree", "forest", "xgboost"):
        X_for_shap = X_train_raw
    else:
        X_for_shap = pd.DataFrame(X_train_scaled, columns=FEATURE_COLUMNS)
    shap_ranking = compute_global_shap(best_model, X_for_shap, best_kind)
    (MODELS_DIR / "shap_global.json").write_text(json.dumps(shap_ranking, indent=2))

    print(f"\nDone in {time.time()-t0:.1f}s. Artifacts written to {MODELS_DIR}")
    print("\nLeaderboard:")
    for name, m in sorted(metrics.items(), key=lambda kv: -kv[1]["roc_auc"]):
        star = "  <-- promoted" if name == best_name else ""
        print(f"  {name:<22} acc={m['accuracy']:>5}%  prec={m['precision']:>5}%  "
              f"rec={m['recall']:>5}%  f1={m['f1']:>5}%  auc={m['roc_auc']}{star}")


if __name__ == "__main__":
    main()
