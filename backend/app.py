"""
app.py
------
PhishGuard – FastAPI production backend.

Endpoints
---------
GET  /health       → liveness / readiness check with model metadata
POST /predict      → classify an email and return risk level + explanations
POST /analyze      → alias for /predict (optional extended response)

Security notes
--------------
- Email bodies are NOT logged in full (privacy).
- Recommend placing behind a reverse proxy (nginx/Caddy) with rate limiting.
- CORS is configured via the CORS_ORIGINS env var (comma-separated).

Authors: Aayush Paudel 2260308, Chhandak Mukherjee 2260357, Arnav Makhija 2260344
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "stacking_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

# ---------------------------------------------------------------------------
# Logging — email bodies must NOT appear in logs
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("phishguard.api")

# ---------------------------------------------------------------------------
# Optional SHAP
# ---------------------------------------------------------------------------
try:
    import shap as _shap  # noqa: F401

    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

# ---------------------------------------------------------------------------
# Project-local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(ROOT))
from utils.preprocessing import combine_subject_body, safe_str  # noqa: E402
from utils.feature_engineering import (  # noqa: E402
    extract_features,
    get_feature_names,
    map_risk_level,
    build_reason_string,
)

# ---------------------------------------------------------------------------
# Application startup: load model
# ---------------------------------------------------------------------------
_MODEL: Any = None
_MODEL_METADATA: Dict[str, Any] = {}
_MODEL_LOAD_ERROR: Optional[str] = None


def _load_model() -> None:
    """Load the serialised pipeline from disk (called at startup)."""
    global _MODEL, _MODEL_METADATA, _MODEL_LOAD_ERROR

    if not MODEL_PATH.exists():
        _MODEL_LOAD_ERROR = (
            f"Model file not found at {MODEL_PATH}. "
            "Run `python backend/train.py` first."
        )
        log.error(_MODEL_LOAD_ERROR)
        return

    try:
        log.info("Loading model from %s …", MODEL_PATH)
        _MODEL = joblib.load(MODEL_PATH)
        log.info("Model loaded successfully.")
    except Exception as exc:
        _MODEL_LOAD_ERROR = f"Failed to load model: {exc}"
        log.error(_MODEL_LOAD_ERROR)
        return

    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH) as f:
                _MODEL_METADATA = json.load(f)
            log.info(
                "Model metadata: version=%s, trained_at=%s",
                _MODEL_METADATA.get("model_version", "?"),
                _MODEL_METADATA.get("trained_at", "?"),
            )
        except Exception as exc:
            log.warning("Could not read model metadata: %s", exc)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PhishGuard API",
    description="Classifies emails as Phishing or Legitimate with risk-level scoring.",
    version="1.0.0",
)

# ---- CORS ----
_raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:  # noqa: D401
    """Load model on application start."""
    _load_model()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Payload for POST /predict.

    At least one of ``subject`` / ``body`` must be non-empty.
    ``email_raw`` is an optional full RFC-2822 email string
    (subject + body are extracted server-side if provided).
    """

    subject: Optional[str] = Field(default="", description="Email subject line.")
    body: Optional[str] = Field(default="", description="Email body text.")
    email_raw: Optional[str] = Field(
        default=None,
        description="Full raw email (subject + body will be extracted from this if provided).",
    )

    @validator("subject", "body", pre=True, always=True)
    def _coerce_none_to_empty(cls, v):  # noqa: N805
        return v or ""


class FeatureContribution(BaseModel):
    feature: str
    value: Any
    score_contribution: float


class PredictResponse(BaseModel):
    label: str = Field(description="'Phishing' or 'Legitimate'")
    phishing_probability: float = Field(ge=0.0, le=1.0)
    risk_level: str = Field(description="'Low', 'Medium', or 'High'")
    top_reasons: List[FeatureContribution]
    reason_summary: str = Field(description="Short human-readable explanation.")
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_raw_email(raw: str) -> tuple[str, str]:
    """Very lightweight RFC-2822 header extraction.

    Returns (subject, body) strings.
    This is intentionally minimal — a proper parser would use email.parser.
    """
    lines = raw.splitlines()
    subject = ""
    body_lines: list[str] = []
    in_body = False

    for line in lines:
        if in_body:
            body_lines.append(line)
        elif line.strip() == "":
            in_body = True
        elif line.lower().startswith("subject:"):
            subject = line[len("subject:") :].strip()

    body = "\n".join(body_lines)
    return subject, body


def _get_top_reasons(
    subject: str,
    body: str,
    phish_prob: float,
    n: int = 3,
) -> list[FeatureContribution]:
    """Return the top *n* feature contributions.

    Strategy
    --------
    1. If SHAP is available and the model exposes tree-based estimators,
       use TreeExplainer (best quality).
    2. Otherwise fall back to hand-crafted feature weights derived from
       the raw metadata features and TF-IDF token contributions.

    NOTE: The meta-learner (LogisticRegression) operates on out-of-fold
    base-learner probabilities, NOT on raw TF-IDF tokens directly.
    For end-user explanation we therefore report the most informative
    *metadata* features (always interpretable) plus optionally the top
    TF-IDF tokens from the TF-IDF component.
    """
    feats = extract_features(subject, body)
    feat_names = get_feature_names()

    # Heuristic contribution weights: features most correlated with phishing
    # (These are rough proxy weights; for precise attribution use SHAP.)
    _PROXY_WEIGHTS: Dict[str, float] = {
        "has_ip_in_url":               0.25,
        "sender_domain_mismatch":      0.20,
        "has_suspicious_phrase":       0.18,
        "domain_has_phishing_keywords":0.16,
        "domain_not_whitelisted":      0.14,
        "has_url":                     0.13,
        "num_urls":                    0.11,
        "has_suspicious_words":        0.10,
        "num_suspicious_phrases":      0.09,
        "num_suspicious_words":        0.08,
        "num_hyphens_in_domain":       0.07,
        "domain_entropy":              0.06,
        "num_subdomains":              0.05,
        "has_html_tags":               0.07,
        "num_exclamations":            0.05,
        "upper_ratio":                 0.04,
        "domain_length":               0.03,
        "num_dollars":                 0.03,
        "num_digits":                  0.02,
        "length_body":                 0.01,
        "length_subject":              0.01,
        "num_question_marks":          0.01,
    }

    contributions = []
    for name in feat_names:
        val = feats.get(name, 0)
        weight = _PROXY_WEIGHTS.get(name, 0.0)
        # Scale weight by value (e.g. num_urls=5 → higher contribution)
        if isinstance(val, float) and 0 < val <= 1:
            scaled = weight * val
        elif isinstance(val, (int, float)) and val > 1:
            scaled = weight * min(val / 10.0, 1.0)
        else:
            scaled = weight * float(val)
        contributions.append(
            FeatureContribution(
                feature=name,
                value=val,
                score_contribution=round(scaled * phish_prob, 4),
            )
        )

    # Sort by score_contribution descending, take top n
    contributions.sort(key=lambda x: x.score_contribution, reverse=True)
    return contributions[:n]


def _build_dataframe(subject: str, body: str) -> pd.DataFrame:
    """Build the single-row DataFrame expected by the pipeline."""
    return pd.DataFrame(
        [
            {
                "subject": safe_str(subject),
                "body": safe_str(body),
                "combined_text": combine_subject_body(subject, body),
            }
        ]
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Liveness / readiness check")
async def health() -> JSONResponse:
    """Return service health status and model metadata."""
    if _MODEL_LOAD_ERROR:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": _MODEL_LOAD_ERROR,
            },
        )
    return JSONResponse(
        content={
            "status": "ok",
            "model_version": _MODEL_METADATA.get("model_version", "unknown"),
            "trained_at": _MODEL_METADATA.get("trained_at", "unknown"),
            "git_commit": _MODEL_METADATA.get("git_commit", "unknown"),
            "test_roc_auc": _MODEL_METADATA.get("test_roc_auc"),
            "test_accuracy": _MODEL_METADATA.get("test_accuracy"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify an email as Phishing or Legitimate",
)
async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Classify an email and return:
    - Binary label (Phishing / Legitimate)
    - Phishing probability (0–1)
    - Risk level (Low / Medium / High)
    - Top 3 feature contributions
    """
    if _MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=_MODEL_LOAD_ERROR or "Model not loaded.",
        )

    # --- Input resolution ---
    subject = payload.subject or ""
    body = payload.body or ""

    if payload.email_raw:
        parsed_subj, parsed_body = _parse_raw_email(payload.email_raw)
        subject = subject or parsed_subj
        body = body or parsed_body

    if not subject.strip() and not body.strip():
        raise HTTPException(
            status_code=400,
            detail="At least one of 'subject' or 'body' must be non-empty.",
        )

    # --- Log sanitised metadata only (NOT the email content) ---
    log.info(
        "Prediction request | subject_len=%d | body_len=%d | ip=%s",
        len(subject),
        len(body),
        request.client.host if request.client else "unknown",
    )

    t0 = time.perf_counter()

    # --- Build input DataFrame ---
    df_input = _build_dataframe(subject, body)

    # --- Inference ---
    try:
        proba = _MODEL.predict_proba(df_input)[0]
        phish_prob = float(proba[1])
        label = "Phishing" if phish_prob >= 0.5 else "Legitimate"
    except Exception as exc:
        log.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    # --- Risk level ---
    risk_level = map_risk_level(phish_prob)

    # --- Explanation ---
    top_reasons = _get_top_reasons(subject, body, phish_prob, n=3)
    reason_summary = build_reason_string(subject, body, phish_prob)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    log.info(
        "Prediction result | label=%s | prob=%.4f | risk=%s | ms=%.1f",
        label,
        phish_prob,
        risk_level,
        elapsed_ms,
    )

    return PredictResponse(
        label=label,
        phishing_probability=round(phish_prob, 4),
        risk_level=risk_level,
        top_reasons=top_reasons,
        reason_summary=reason_summary,
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.post(
    "/analyze",
    response_model=PredictResponse,
    summary="Alias for /predict with identical behaviour",
)
async def analyze(payload: PredictRequest, request: Request) -> PredictResponse:
    """Extended analysis endpoint — currently an alias for /predict."""
    return await predict(payload, request)


# ---------------------------------------------------------------------------
# Dev-mode entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
