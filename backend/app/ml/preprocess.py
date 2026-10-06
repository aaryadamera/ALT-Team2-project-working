"""
Preprocessing pipeline.

This module is imported by BOTH train.py and the /predict API route, so a
live applicant is guaranteed to go through exactly the same transformation
the models were trained on:

  1. Median-impute missing MonthlyIncome / NumberOfDependents
  2. Cap outliers at the 99.5th percentile (per feature)
  3. Drop rows with an invalid age (age <= 0) -- training only
  4. Standard-scale a copy of the features for Logistic Regression & SVM;
     Decision Tree / Random Forest / XGBoost use the raw, unscaled features
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.ml.data import FEATURE_COLUMNS

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
IMPUTE_PATH = MODELS_DIR / "impute_values.json"
CAPS_PATH = MODELS_DIR / "outlier_caps.json"
SCALER_PATH = MODELS_DIR / "scaler.pkl"


def fit_preprocessing(df: pd.DataFrame) -> dict:
    """Learn imputation medians and outlier caps from the TRAINING split only,
    save them, and return them for immediate use."""
    impute_values = {
        "MonthlyIncome": float(df["MonthlyIncome"].median()),
        "NumberOfDependents": float(df["NumberOfDependents"].median()),
    }
    df_imputed = df.copy()
    df_imputed["MonthlyIncome"] = df_imputed["MonthlyIncome"].fillna(impute_values["MonthlyIncome"])
    df_imputed["NumberOfDependents"] = df_imputed["NumberOfDependents"].fillna(impute_values["NumberOfDependents"])

    caps = {col: float(df_imputed[col].quantile(0.995)) for col in FEATURE_COLUMNS}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    IMPUTE_PATH.write_text(json.dumps(impute_values, indent=2))
    CAPS_PATH.write_text(json.dumps(caps, indent=2))
    return {"impute_values": impute_values, "caps": caps}


def load_preprocessing_artifacts() -> dict:
    impute_values = json.loads(IMPUTE_PATH.read_text())
    caps = json.loads(CAPS_PATH.read_text())
    return {"impute_values": impute_values, "caps": caps}


def apply_preprocessing(df: pd.DataFrame, artifacts: dict, drop_invalid_age: bool = False) -> pd.DataFrame:
    """Apply impute -> cap -> (optional) invalid-age drop. Does NOT scale --
    scaling is a separate, model-specific step (see fit_scaler / scale_features)."""
    out = df.copy()
    out["MonthlyIncome"] = out["MonthlyIncome"].fillna(artifacts["impute_values"]["MonthlyIncome"])
    out["NumberOfDependents"] = out["NumberOfDependents"].fillna(artifacts["impute_values"]["NumberOfDependents"])

    for col, cap in artifacts["caps"].items():
        out[col] = out[col].clip(upper=cap)

    if drop_invalid_age:
        out = out[out["age"] > 0].reset_index(drop=True)

    return out


def fit_scaler(X_train_raw: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train_raw[FEATURE_COLUMNS])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    return scaler


def load_scaler() -> StandardScaler:
    return joblib.load(SCALER_PATH)


def scale_features(X_raw: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    scaled = scaler.transform(X_raw[FEATURE_COLUMNS])
    return pd.DataFrame(scaled, columns=FEATURE_COLUMNS, index=X_raw.index)
