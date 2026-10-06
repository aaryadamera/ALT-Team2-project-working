"""
Explainability layer: SHAP for global feature importance, LIME for a single
prediction. Both are computed against the promoted (best) model.

If shap / lime aren't installed in the current environment, this module
falls back to a permutation-importance-based approximation so the rest of
the app (API, DB, frontend) can still be developed and demoed offline. Once
`pip install shap lime` succeeds, the real explainers are used automatically
-- no other code changes needed.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from app.ml.data import FEATURE_COLUMNS

try:
    import shap as _shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    HAS_LIME = True
except ImportError:
    HAS_LIME = False


# ---------------------------------------------------------------- SHAP (global)

def compute_global_shap(model, X_background_raw: pd.DataFrame, model_kind: str) -> list[dict]:
    """Returns a list of {feature, mean_abs_shap, direction} sorted by importance,
    for the model promoted to production (tree-based -> TreeExplainer)."""
    if HAS_SHAP and model_kind in ("tree", "xgboost", "forest"):
        explainer = _shap.TreeExplainer(model)
        sample = X_background_raw[FEATURE_COLUMNS].sample(
            n=min(2000, len(X_background_raw)), random_state=42
        )
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):  # binary-classifier output shape
            shap_values = shap_values[1]
        mean_abs = np.abs(shap_values).mean(axis=0)
        mean_signed = shap_values.mean(axis=0)
    else:
        # Permutation-importance fallback (no shap dependency required)
        from sklearn.inspection import permutation_importance
        sample = X_background_raw.sample(n=min(3000, len(X_background_raw)), random_state=42)
        y_proxy = (model.predict_proba(sample[FEATURE_COLUMNS])[:, 1] > 0.5).astype(int)
        result = permutation_importance(
            model, sample[FEATURE_COLUMNS], y_proxy, n_repeats=3, random_state=42, scoring="roc_auc"
        )
        mean_abs = np.clip(result.importances_mean, 0, None)
        mean_signed = mean_abs  # sign unknown in fallback; treat as risk-increasing

    ranking = []
    for i, feat in enumerate(FEATURE_COLUMNS):
        ranking.append({
            "feature": feat,
            "mean_abs_shap": round(float(mean_abs[i]), 5),
            "direction": "positive" if mean_signed[i] >= 0 else "negative",
        })
    ranking.sort(key=lambda r: r["mean_abs_shap"], reverse=True)
    return ranking


# ----------------------------------------------------------------- LIME (local)

def build_lime_explainer(X_train_raw: pd.DataFrame) -> "LimeTabularExplainer | None":
    if not HAS_LIME:
        return None
    return LimeTabularExplainer(
        training_data=X_train_raw[FEATURE_COLUMNS].values,
        feature_names=FEATURE_COLUMNS,
        class_names=["Repaid", "Default"],
        mode="classification",
        discretize_continuous=True,
    )


def explain_instance(model, explainer, instance_row: pd.DataFrame, predict_fn) -> list[dict]:
    """Returns a ranked list of {feature, value, contribution, direction} for one applicant."""
    if HAS_LIME and explainer is not None:
        exp = explainer.explain_instance(
            instance_row[FEATURE_COLUMNS].values[0],
            predict_fn,
            num_features=len(FEATURE_COLUMNS),
        )
        factors = []
        for condition, weight in exp.as_list():
            factors.append({
                "feature": condition,
                "contribution": round(float(weight), 4),
                "direction": "positive" if weight > 0 else "negative",
            })
        factors.sort(key=lambda f: abs(f["contribution"]), reverse=True)
        return factors

    # Fallback: local perturbation sensitivity (finite-difference "poor man's LIME")
    base_pred = predict_fn(instance_row[FEATURE_COLUMNS].values)[0][1]
    factors = []
    for feat in FEATURE_COLUMNS:
        perturbed = instance_row.copy()
        std = max(abs(instance_row[feat].values[0]) * 0.1, 0.01)
        perturbed[feat] = perturbed[feat] + std
        new_pred = predict_fn(perturbed[FEATURE_COLUMNS].values)[0][1]
        delta = new_pred - base_pred
        factors.append({
            "feature": f"{feat} = {instance_row[feat].values[0]:.2f}",
            "contribution": round(float(delta), 4),
            "direction": "positive" if delta > 0 else "negative",
        })
    factors.sort(key=lambda f: abs(f["contribution"]), reverse=True)
    return factors
