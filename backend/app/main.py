"""
CreditTrust AI backend.

Routes (all under /api):
    GET  /api/health                -> liveness + which model is promoted
    GET  /api/leaderboard           -> all 5 models' metrics (Model Comparison tab)
    GET  /api/shap/global           -> global SHAP feature importance (Explainability tab)
    POST /api/predict               -> score one applicant, explain it, log it to SQLite
    GET  /api/predictions/recent    -> audit trail of recent predictions (from the DB)

Run from the backend/ directory:
    python -m app.ml.train        # one-time: trains models, writes artifacts
    uvicorn app.main:app --reload # starts the API + serves the frontend at "/"
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database import init_db, log_prediction, get_recent_predictions
from app.inference import get_engine
from app.schemas import ApplicantInput, PredictionOut, LeaderboardEntry, ShapFeature

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

app = FastAPI(title="CreditTrust AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    init_db()
    get_engine()  # fail fast at startup if train.py hasn't been run yet


@app.get("/api/health")
def health():
    engine = get_engine()
    return {
        "status": "ok",
        "promoted_model": engine.best_meta["name"],
        "roc_auc": engine.best_meta["roc_auc"],
        "trained_on_real_data": engine.best_meta["trained_on_real_data"],
        "train_rows": engine.best_meta.get("train_rows"),
        "test_rows": engine.best_meta.get("test_rows"),
        "default_rate_pct": engine.best_meta.get("default_rate_pct"),
    }


@app.get("/api/leaderboard", response_model=list[LeaderboardEntry])
def leaderboard():
    engine = get_engine()
    best_name = engine.best_meta["name"]
    return [
        LeaderboardEntry(
            model=name,
            accuracy=m["accuracy"],
            precision=m["precision"],
            recall=m["recall"],
            f1=m["f1"],
            roc_auc=m["roc_auc"],
            is_best=(name == best_name),
        )
        for name, m in engine.metrics.items()
    ]


@app.get("/api/shap/global", response_model=list[ShapFeature])
def shap_global():
    engine = get_engine()
    return [ShapFeature(**f) for f in engine.shap_global]


@app.post("/api/predict", response_model=PredictionOut)
def predict(applicant: ApplicantInput):
    engine = get_engine()
    try:
        result = engine.predict(applicant.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    log_prediction(
        applicant_name=result["applicant_name"],
        inputs=applicant.model_dump(),
        model_name=result["model_used"],
        default_probability=result["default_probability"],
        risk_label=result["risk_label"],
        top_factors=result["top_factors"],
    )
    return PredictionOut(**result)


@app.get("/api/predictions/recent")
def recent_predictions(limit: int = 20):
    return get_recent_predictions(limit=min(limit, 100))


# Serve the frontend last, so /api/* routes above take precedence
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
