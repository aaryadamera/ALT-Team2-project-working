"""
Lightweight SQLite persistence layer.

Every prediction served by the API is written here: the applicant inputs,
the model's output, and which model/version produced it. This is what makes
the system's decisions auditable, per the "Decision Support System" stage of
the pipeline -- a loan officer (or a faculty reviewer) can open
credittrust.db directly, or hit GET /api/predictions/recent, and see exactly
what the model was shown and what it decided.

Plain sqlite3 (stdlib) is used deliberately -- no ORM -- so the whole data
layer is inspectable in a few lines of code, which matters when the goal is
to explain it to someone, not just run it.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "credittrust.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    applicant_name TEXT,
    inputs_json TEXT NOT NULL,
    model_name TEXT NOT NULL,
    default_probability REAL NOT NULL,
    risk_label TEXT NOT NULL,
    top_factors_json TEXT NOT NULL
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(SCHEMA)


def log_prediction(applicant_name: str, inputs: dict, model_name: str,
                    default_probability: float, risk_label: str, top_factors: list) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO predictions
               (created_at, applicant_name, inputs_json, model_name,
                default_probability, risk_label, top_factors_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                applicant_name,
                json.dumps(inputs),
                model_name,
                default_probability,
                risk_label,
                json.dumps(top_factors),
            ),
        )
        return cur.lastrowid


def get_recent_predictions(limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [
        {
            "id": r["id"],
            "created_at": r["created_at"],
            "applicant_name": r["applicant_name"],
            "inputs": json.loads(r["inputs_json"]),
            "model_name": r["model_name"],
            "default_probability": r["default_probability"],
            "risk_label": r["risk_label"],
            "top_factors": json.loads(r["top_factors_json"]),
        }
        for r in rows
    ]


def count_predictions() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
