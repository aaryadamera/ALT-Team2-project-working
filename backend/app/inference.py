"""
Inference layer used by the API. Loads every artifact train.py produced
exactly once at startup, then serves predictions + LIME explanations without
re-reading disk on every request.

This is the module a faculty reviewer should be pointed at to see "where the
logic lives": preprocess -> best model's predict_proba -> LIME -> DB log.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from app.ml.data import FEATURE_COLUMNS
from app.ml.preprocess import load_preprocessing_artifacts, apply_preprocessing, load_scaler, scale_features
from app.ml.explain import build_lime_explainer, explain_instance
from app.ml.data import load_dataset

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

RISK_THRESHOLD = 0.40  # default probability at/above this => High Risk

# Map the API's human-friendly field names to the dataset's original column names
FIELD_TO_COLUMN = {
    "age": "age",
    "monthly_income": "MonthlyIncome",
    "revolving_utilization": "RevolvingUtilizationOfUnsecuredLines",
    "debt_ratio": "DebtRatio",
    "open_credit_lines": "NumberOfOpenCreditLinesAndLoans",
    "real_estate_loans": "NumberRealEstateLoansOrLines",
    "dependents": "NumberOfDependents",
    "late_30_59_days": "NumberOfTime30-59DaysPastDueNotWorse",
    "late_60_89_days": "NumberOfTime60-89DaysPastDueNotWorse",
    "late_90_plus_days": "NumberOfTimes90DaysLate",
}

FRIENDLY_NAME = {
    "RevolvingUtilizationOfUnsecuredLines": "Credit utilization ratio",
    "age": "Applicant age",
    "NumberOfTime30-59DaysPastDueNotWorse": "Times 30-59 days late",
    "DebtRatio": "Debt-to-income ratio",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines/loans",
    "NumberOfTimes90DaysLate": "Times 90+ days late",
    "NumberRealEstateLoansOrLines": "Real-estate loans/lines",
    "NumberOfTime60-89DaysPastDueNotWorse": "Times 60-89 days late",
    "NumberOfDependents": "Number of dependents",
}


class InferenceEngine:
    """Loads once, reused across requests. Raises a clear error if train.py
    hasn't been run yet -- there is nothing to silently fall back to."""

    def __init__(self):
        if not (MODELS_DIR / "best_model.json").exists():
            raise RuntimeError(
                "No trained model found. Run `python -m app.ml.train` from the backend/ "
                "directory once before starting the API."
            )
        self.best_meta = json.loads((MODELS_DIR / "best_model.json").read_text())
        self.metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
        self.shap_global = json.loads((MODELS_DIR / "shap_global.json").read_text())
        self.model = joblib.load(MODELS_DIR / self.best_meta["file"])
        self.prep_artifacts = load_preprocessing_artifacts()
        self.scaler = load_scaler()

        # Background data for LIME needs to be preprocessed the same way as training
        df, _ = load_dataset()
        df = apply_preprocessing(df, self.prep_artifacts, drop_invalid_age=True)
        self._background_raw = df[FEATURE_COLUMNS]
        self._background = (
            scale_features(self._background_raw, self.scaler)
            if self.best_meta["uses_scaled_features"]
            else self._background_raw
        )
        self.lime_explainer = build_lime_explainer(self._background)

    def _predict_proba_fn(self, X):
        """LIME calls this with a raw numpy array of perturbed samples."""
        df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
        return self.model.predict_proba(df)

    def predict(self, applicant: dict) -> dict:
        row = {FIELD_TO_COLUMN[k]: v for k, v in applicant.items() if k in FIELD_TO_COLUMN}
        raw_df = pd.DataFrame([row])[FEATURE_COLUMNS]

        # same impute + cap pipeline as training (no invalid-age drop for live inference)
        processed = apply_preprocessing(raw_df, self.prep_artifacts, drop_invalid_age=False)

        model_input = (
            scale_features(processed, self.scaler)
            if self.best_meta["uses_scaled_features"]
            else processed
        )

        proba = float(self.model.predict_proba(model_input)[0][1])
        risk_label = "High Risk / Likely Default" if proba >= RISK_THRESHOLD else "Approved / Low Risk"
        decision = "Refer for manual review" if proba >= RISK_THRESHOLD else "Approve"

        factors_raw = explain_instance(
            self.model, self.lime_explainer, processed, self._predict_proba_fn
        )
        factors = []
        for f in factors_raw[:6]:
            feat_label = f["feature"]
            for col, friendly in FRIENDLY_NAME.items():
                feat_label = feat_label.replace(col, friendly)
            factors.append({
                "feature": feat_label,
                "contribution": f["contribution"],
                "direction": f["direction"],
            })

        explanation_text = self._plain_language_explanation(
            applicant, proba, risk_label, factors, self.best_meta["name"]
        )

        return {
            "applicant_name": applicant.get("applicant_name", "Applicant"),
            "model_used": self.best_meta["name"],
            "default_probability": round(proba, 4),
            "risk_label": risk_label,
            "decision": decision,
            "top_factors": factors,
            "explanation_text": explanation_text,
        }

    @staticmethod
    def _plain_language_explanation(applicant: dict, proba: float, risk_label: str, factors: list, model_name: str) -> str:
        name = applicant.get("applicant_name", "The applicant")
        pct = round(proba * 100, 1)
        top_pos = next((f for f in factors if f["direction"] == "positive"), None)
        top_neg = next((f for f in factors if f["direction"] == "negative"), None)
        text = f"{name} was scored at a {pct}% probability of default by {model_name}. "
        if risk_label.startswith("High"):
            text += f"The largest risk driver was {top_pos['feature'].lower()}. " if top_pos else ""
            if top_neg:
                text += f"{top_neg['feature']} partially offsets this, but not enough to clear the approval threshold. "
            text += "We recommend a manual review before a final decision."
        else:
            text += f"The application clears the approval threshold, supported most by {top_neg['feature'].lower()}. " if top_neg else "The application clears the approval threshold. "
            if top_pos:
                text += f"{top_pos['feature']} is noted as a minor watch-item but is not disqualifying."
        return text


_engine: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine
