from pydantic import BaseModel, Field


class ApplicantInput(BaseModel):
    applicant_name: str = Field(default="Applicant", max_length=120)
    age: int = Field(ge=18, le=100)
    monthly_income: float = Field(ge=0, le=1_000_000)
    revolving_utilization: float = Field(ge=0, le=5, description="Fraction of available revolving credit in use")
    debt_ratio: float = Field(ge=0, le=10)
    open_credit_lines: int = Field(ge=0, le=60)
    real_estate_loans: int = Field(ge=0, le=30)
    dependents: int = Field(ge=0, le=20)
    late_30_59_days: int = Field(ge=0, le=30)
    late_60_89_days: int = Field(ge=0, le=30)
    late_90_plus_days: int = Field(ge=0, le=30)


class FactorOut(BaseModel):
    feature: str
    contribution: float
    direction: str


class PredictionOut(BaseModel):
    applicant_name: str
    model_used: str
    default_probability: float
    risk_label: str
    decision: str
    top_factors: list[FactorOut]
    explanation_text: str


class LeaderboardEntry(BaseModel):
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    is_best: bool


class ShapFeature(BaseModel):
    feature: str
    mean_abs_shap: float
    direction: str
