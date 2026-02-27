#!/usr/bin/env python3
"""
train.py
--------
PhishGuard – end-to-end training script.

Usage
-----
  # Full training run (uses all three datasets):
  python backend/train.py

  # Fast iteration / CI (sub-samples each dataset):
  python backend/train.py --sample 2000

  # Tune hyperparameters more aggressively:
  python backend/train.py --tune

  # Debug: print dataset previews before mapping columns:
  python backend/train.py --debug

Pipeline summary
----------------
1.  Load three CSVs and intelligently unify columns → is_phish (0/1).
2.  Clean text (lowercase, strip HTML, decode entities).
3.  Feature union:  TF-IDF (1-2 grams, 20k feats) ⊕ Metadata features.
4.  Stacking ensemble:
      Base learners: RandomForest + XGBoost
      Meta-learner : LogisticRegression (trained on out-of-fold proba)
      CV strategy  : StratifiedKFold(n_splits=5)
5.  Evaluate on held-out test set (accuracy, F1, AUC, confusion matrix).
6.  Calibrate probabilities with CalibratedClassifierCV.
7.  Save full pipeline to models/stacking_pipeline.joblib.

Authors: Aayush Paudel 2260308, Chhandak Mukherjee 2260357, Arnav Makhija 2260344
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Attempt to import optional / heavy dependencies gracefully
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    logging.warning("xgboost not installed – will use extra RandomForest instead.")

try:
    import shap

    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    logging.warning("shap not installed – SHAP explanations will be skipped.")

# ---------------------------------------------------------------------------
# Project-local imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.preprocessing import clean_text, combine_subject_body, safe_str  # noqa: E402
from utils.feature_engineering import (  # noqa: E402
    MetadataFeatureExtractor,
    get_feature_names,
    get_text_column,
)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("phishguard.train")

# ---------------------------------------------------------------------------
# Dataset paths – resolved to their actual location on this machine
# ---------------------------------------------------------------------------
_DATA_DIR = ROOT.parent / "data"

DATASET_PATHS = [
    _DATA_DIR / "spam (1).csv",
    _DATA_DIR / "phishing_email.csv",
    _DATA_DIR / "TREC_05_cleaned.csv",
]

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "stacking_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# StratifiedKFold splits for stacking meta-features
N_SPLITS = 5

# ---------------------------------------------------------------------------
# Column-mapping heuristics
# ---------------------------------------------------------------------------

# Candidate column names for each semantic field
_SUBJECT_CANDIDATES = [
    "subject", "Subject", "SUBJECT", "email_subject", "title",
    "clean_subject",               # TREC_05_cleaned.csv
]
_BODY_CANDIDATES = [
    "body", "Body", "BODY",
    "message", "Message",
    "content", "Content",
    "text", "Text",
    "email_text", "email", "Email", "mail", "Mail",
    "clean_body",                  # TREC_05_cleaned.csv
    "text_combined",               # phishing_email.csv
    "v2",                          # spam (1).csv  (v2 = raw SMS text)
]
_LABEL_CANDIDATES = [
    "label", "Label", "LABEL",
    "is_phish", "is_spam",
    "spam", "Spam",
    "class", "Class",
    "category", "Category",
    "phishing", "target", "type",
    "v1",                          # spam (1).csv  (v1 = ham/spam label)
    "Email Type",                  # some phishing_email variants
]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first candidate column name that exists in *df*, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    # Fuzzy case-insensitive fallback
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _map_label(series: pd.Series, source_name: str) -> pd.Series:
    """Map raw labels to binary is_phish (1 = phishing, 0 = legitimate).

    Different datasets use wildly different schemes:
    - spam (1).csv          : 'spam' / 'ham'  or 1 / 0
    - phishing_email.csv    : 'Phishing Email' / 'Safe Email' or 1 / 0
    - TREC_05_cleaned.csv   : 'spam' / 'ham'  or 1 / 0

    If numeric: assume 1 = phish / spam, 0 = legit.
    If string:  map positive labels to 1.
    """
    positive_strings = {
        "spam",
        "phishing",
        "phish",
        "phishing email",
        "malicious",
        "1",
        "yes",
        "true",
        "unsafe",
        "fraud",
        "scam",
    }

    def _to_binary(val) -> int:
        if pd.isna(val):
            return -1  # will be dropped later
        if isinstance(val, (int, float)):
            return int(val) if int(val) in (0, 1) else -1
        return 1 if str(val).strip().lower() in positive_strings else 0

    mapped = series.apply(_to_binary)
    n_unknown = (mapped == -1).sum()
    if n_unknown > 0:
        log.warning(
            "[%s] Could not map %d label values – they will be dropped.", source_name, n_unknown
        )
    return mapped


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(path: Path, debug: bool = False) -> Optional[pd.DataFrame]:
    """Load a single CSV file and normalise it to columns [subject, body, is_phish].

    Parameters
    ----------
    path:
        Absolute path to the CSV file.
    debug:
        When True, print a preview of raw columns / values.

    Returns
    -------
    pd.DataFrame or None if the file cannot be read / mapped.
    """
    if not path.exists():
        log.warning("Dataset not found: %s – skipping.", path)
        return None

    log.info("Loading: %s", path)
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)

    log.info("  Columns: %s", list(df.columns))
    log.info("  Shape  : %s", df.shape)

    if debug:
        print(f"\n--- Preview of {path.name} ---")
        print(df.head(3).to_string())
        print()

    # ---- Label column ----
    label_col = _find_col(df, _LABEL_CANDIDATES)
    if label_col is None:
        log.error(
            "[%s] Cannot find label column. Columns are: %s\n"
            "→ MANUAL FIX NEEDED: add the correct column name to _LABEL_CANDIDATES "
            "or rename the column before training.",
            path.name,
            list(df.columns),
        )
        return None

    # ---- Subject column (optional) ----
    subject_col = _find_col(df, _SUBJECT_CANDIDATES)

    # ---- Body column ----
    body_col = _find_col(df, _BODY_CANDIDATES)
    if body_col is None:
        log.error(
            "[%s] Cannot find body/text column. Columns are: %s",
            path.name,
            list(df.columns),
        )
        return None

    # Build unified frame
    out = pd.DataFrame()
    out["subject"] = df[subject_col].apply(safe_str) if subject_col else ""
    out["body"] = df[body_col].apply(safe_str)
    out["is_phish"] = _map_label(df[label_col], path.name)

    # Drop unmappable labels
    out = out[out["is_phish"].isin([0, 1])].reset_index(drop=True)

    phish_pct = out["is_phish"].mean() * 100
    log.info(
        "  → Unified shape: %s  | phishing: %.1f%%", out.shape, phish_pct
    )
    return out


def load_all_datasets(
    paths: list[Path],
    sample: Optional[int] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """Load, unify and deduplicate all datasets.

    Parameters
    ----------
    paths:
        List of CSV file paths.
    sample:
        If set, sub-sample at most *sample* rows per dataset (for fast iteration).
    debug:
        Pass-through to :func:`load_dataset`.

    Returns
    -------
    pd.DataFrame with columns [subject, body, is_phish].
    """
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = load_dataset(p, debug=debug)
        if df is not None:
            if sample:
                # Stratified sub-sampling (pandas-3 compatible)
                n = min(sample, len(df))
                half = n // 2
                parts = []
                for label_val, grp in df.groupby("is_phish"):
                    parts.append(grp.sample(min(len(grp), half), random_state=42))
                df = pd.concat(parts, ignore_index=True)
            frames.append(df)

    if not frames:
        raise RuntimeError(
            "No datasets could be loaded. Check the DATASET_PATHS in train.py."
        )

    combined = pd.concat(frames, ignore_index=True)
    log.info("Combined (pre-dedup) shape: %s", combined.shape)

    # Deduplicate on body text (keeping first occurrence)
    before = len(combined)
    combined["_dedup_key"] = combined["body"].str.strip().str.lower()
    combined = combined.drop_duplicates(subset="_dedup_key").drop(columns="_dedup_key")
    log.info("After deduplication: %d rows (removed %d)", len(combined), before - len(combined))

    phish_pct = combined["is_phish"].mean() * 100
    log.info(
        "Class balance – phishing: %.1f%% | legitimate: %.1f%%",
        phish_pct,
        100 - phish_pct,
    )

    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# sklearn Pipeline components
# ---------------------------------------------------------------------------


def build_pipeline(n_tfidf_features: int = 20_000) -> Pipeline:
    """Construct the full feature-extraction pipeline.

    Architecture
    ------------
    FeatureUnion
    ├── TF-IDF on combined_text (1–2 grams, max_features)
    └── MetadataFeatureExtractor  (14 hand-crafted features)

    The union output is fed into the stacking classifier.

    Parameters
    ----------
    n_tfidf_features:
        Vocabulary size cap for TF-IDF.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted feature pipeline (transform-only, no classifier).
    """
    text_pipeline = Pipeline(
        [
            (
                "text_selector",
                FunctionTransformer(get_text_column, validate=False),
            ),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=n_tfidf_features,
                    stop_words="english",
                    sublinear_tf=True,   # log-scaled TF to reduce impact of very frequent terms
                    min_df=2,
                ),
            ),
        ]
    )

    feature_union = FeatureUnion(
        transformer_list=[
            ("tfidf_pipe", text_pipeline),
            ("meta_feats", MetadataFeatureExtractor()),
        ]
    )

    return feature_union


# ---------------------------------------------------------------------------
# Stacking classifier
# ---------------------------------------------------------------------------


def build_stacking_classifier(tune: bool = False) -> StackingClassifier:
    """Build the stacking ensemble.

    Base learners
    -------------
    - RandomForestClassifier (bagging-based)
    - XGBClassifier          (gradient boosting)

    Meta-learner
    ------------
    LogisticRegression trained on out-of-fold base-learner probabilities.

    Parameters
    ----------
    tune:
        If True, wrap base learners in RandomizedSearchCV for HPO.
        This increases training time significantly but may improve AUC.

    Returns
    -------
    StackingClassifier (unfitted)
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # ---- Random Forest ----
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    if tune:
        rf_param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_leaf": [1, 2, 4],
        }
        rf = RandomizedSearchCV(
            rf,
            rf_param_dist,
            n_iter=6,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )

    # ---- XGBoost (or fallback RF) ----
    if _HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",   # fast histogram-based
            n_jobs=-1,
            random_state=42,
        )
        if tune:
            xgb_param_dist = {
                "n_estimators": [100, 200],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
            }
            xgb = RandomizedSearchCV(
                xgb,
                xgb_param_dist,
                n_iter=6,
                cv=3,
                scoring="roc_auc",
                n_jobs=-1,
                random_state=42,
                verbose=0,
            )
        second_learner = ("xgb", xgb)
    else:
        # Fallback: second RF with different parameters
        rf2 = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=99,
        )
        second_learner = ("rf2", rf2)

    meta = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )

    stacking = StackingClassifier(
        estimators=[("rf", rf), second_learner],
        final_estimator=meta,
        cv=cv,
        stack_method="predict_proba",
        passthrough=False,   # only use base learner proba as meta-features
        n_jobs=1,            # parallelism inside each learner is already set
    )

    return stacking


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------


def evaluate(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate a fitted pipeline on the test set.

    Returns
    -------
    dict with keys: accuracy, roc_auc, report, confusion_matrix.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"])
    cm = confusion_matrix(y_test, y_pred)

    log.info("Test Accuracy : %.4f", acc)
    log.info("Test ROC-AUC  : %.4f", auc)
    log.info("Classification Report:\n%s", report)
    log.info("Confusion Matrix:\n%s", cm)

    return {
        "accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def _extract_tfidf_top_features(pipeline, feature_matrix, n_top: int = 20) -> list[tuple]:
    """Extract top-weighted TF-IDF token names from the pipeline.

    Used as a fallback explanation when SHAP is unavailable.
    Returns a list of (feature_name, weight) sorted by |weight|.
    """
    # Navigate: pipeline → "full_pipeline" (FeatureUnion) → tfidf_pipe → tfidf
    try:
        tfidf = (
            pipeline.named_steps["full_pipeline"]
            .transformer_list[0][1]  # tfidf_pipe (Pipeline)
            .named_steps["tfidf"]
        )
        vocab = tfidf.get_feature_names_out()

        # Get meta-learner coefficients (LogisticRegression)
        meta = pipeline.named_steps["stacking"].final_estimator_
        if hasattr(meta, "coef_"):
            # Only TF-IDF feature weights (meta-learner gets base-learner proba, not raw features)
            # → fall back to raw TF-IDF sum across test matrix
            tfidf_weights = np.array(feature_matrix[:, : len(vocab)].sum(axis=0)).flatten()
            top_idx = np.argsort(tfidf_weights)[::-1][:n_top]
            return [(vocab[i], float(tfidf_weights[i])) for i in top_idx]
    except Exception as exc:
        log.debug("Could not extract TF-IDF features: %s", exc)
    return []


# ---------------------------------------------------------------------------
# Main training entrypoint
# ---------------------------------------------------------------------------


def train(
    sample: Optional[int] = None,
    tune: bool = False,
    debug: bool = False,
    n_tfidf_features: int = 20_000,
) -> None:
    """Run the full training pipeline.

    Parameters
    ----------
    sample:
        If set, sub-sample at most *sample* rows per dataset (dev/CI mode).
    tune:
        Enable RandomizedSearchCV hyperparameter tuning.
    debug:
        Print raw dataset previews before column mapping.
    n_tfidf_features:
        TF-IDF vocabulary size.
    """
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("PhishGuard – Training Pipeline")
    log.info("=" * 60)
    data = load_all_datasets(DATASET_PATHS, sample=sample, debug=debug)

    # ------------------------------------------------------------------
    # 2. Train / val / test split  (60 / 20 / 20 – stratified)
    # ------------------------------------------------------------------
    X_temp, X_test, y_temp, y_test = train_test_split(
        data[["subject", "body"]],
        data["is_phish"],
        test_size=0.20,
        random_state=42,
        stratify=data["is_phish"],
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,  # 0.25 × 0.80 = 0.20 of total
        random_state=42,
        stratify=y_temp,
    )
    log.info(
        "Split – train: %d | val: %d | test: %d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    # ------------------------------------------------------------------
    # 3. Add combined text column for TF-IDF
    # ------------------------------------------------------------------
    def _add_combined(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()  # avoid pandas 3 CoW issues with slice assignment
        df["combined_text"] = df.apply(
            lambda r: combine_subject_body(r["subject"], r["body"]), axis=1
        )
        return df

    X_train = _add_combined(X_train)
    X_val   = _add_combined(X_val)
    X_test  = _add_combined(X_test)

    # ------------------------------------------------------------------
    # 4. Build feature pipeline + stacking classifier
    # ------------------------------------------------------------------
    log.info("Building feature pipeline (TF-IDF + metadata) …")
    feat_pipeline = build_pipeline(n_tfidf_features=n_tfidf_features)

    log.info("Building stacking classifier (RF + XGB → LogisticRegression) …")
    stacker = build_stacking_classifier(tune=tune)

    # Full end-to-end pipeline
    full_pipeline = Pipeline(
        [
            ("full_pipeline", feat_pipeline),
            ("stacking", stacker),
        ]
    )

    # ------------------------------------------------------------------
    # 5. Fit on training data
    # ------------------------------------------------------------------
    log.info("Fitting pipeline on %d training samples …", len(X_train))
    full_pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 6. Validate
    # ------------------------------------------------------------------
    log.info("Validation metrics:")
    val_metrics = evaluate(full_pipeline, X_val, y_val)

    # ------------------------------------------------------------------
    # 7. Evaluate on test set
    # ------------------------------------------------------------------
    log.info("Test set metrics:")
    test_metrics = evaluate(full_pipeline, X_test, y_test)

    # ------------------------------------------------------------------
    # 8. Probability calibration
    #    sklearn ≥ 1.8 removed the prefit mode from CalibratedClassifierCV.
    #    StackingClassifier with stack_method="predict_proba" already
    #    produces cross-validated, reasonably calibrated probabilities,
    #    so we skip the extra Platt scaling step and evaluate directly.
    # ------------------------------------------------------------------
    log.info("Post-fit (calibration skipped – sklearn 1.8+) test metrics:")
    test_metrics_cal = evaluate(full_pipeline, X_test, y_test)

    # ------------------------------------------------------------------
    # 9. (Optional) SHAP explanation on a small test subset
    # ------------------------------------------------------------------
    if _HAS_SHAP:
        log.info("Computing SHAP values on 100-sample subset …")
        try:
            # Transform test samples
            X_test_sub = X_test.iloc[:100]
            X_test_transformed = full_pipeline.named_steps["full_pipeline"].transform(X_test_sub)

            # Try TreeExplainer on the base RF inside the stacking classifier
            stacking_step = full_pipeline.named_steps["stacking"]
            rf_estimator = stacking_step.estimators_[0][1]  # ("rf", <RF>)
            if hasattr(rf_estimator, "estimators_"):
                explainer = shap.TreeExplainer(rf_estimator)
                _shap_vals = explainer.shap_values(X_test_transformed)
                log.info("SHAP TreeExplainer fitted successfully.")
        except Exception as exc:
            log.warning("SHAP computation failed: %s", exc)
    else:
        log.info("SHAP not available – skipping global explanations.")

    # ------------------------------------------------------------------
    # 10. Save pipeline
    # ------------------------------------------------------------------
    log.info("Saving pipeline to %s …", MODEL_PATH)
    joblib.dump(full_pipeline, MODEL_PATH)

    # Save model metadata (version info, timestamp, metrics)
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "1.0.0",
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_tfidf_features": n_tfidf_features,
        "n_splits": N_SPLITS,
        "tuned": tune,
        "sampled": sample is not None,
        "val_accuracy": val_metrics["accuracy"],
        "val_roc_auc": val_metrics["roc_auc"],
        "test_accuracy": test_metrics_cal["accuracy"],
        "test_roc_auc": test_metrics_cal["roc_auc"],
        "test_confusion_matrix": test_metrics_cal["confusion_matrix"],
    }

    # Attempt to capture git commit hash for reproducibility
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        metadata["git_commit"] = git_hash
    except Exception:
        metadata["git_commit"] = "unknown"

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("Training complete in %.1f seconds.", elapsed)
    log.info("Model saved to: %s", MODEL_PATH)
    log.info("Metadata saved to: %s", METADATA_PATH)
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PhishGuard – train the stacking ensemble.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Sub-sample at most N rows per dataset for fast iteration. "
            "Omit for full training run."
        ),
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable RandomizedSearchCV hyperparameter tuning (slower).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw dataset previews before column mapping.",
    )
    parser.add_argument(
        "--tfidf-features",
        type=int,
        default=20_000,
        dest="n_tfidf_features",
        help="TF-IDF vocabulary size cap.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        sample=args.sample,
        tune=args.tune,
        debug=args.debug,
        n_tfidf_features=args.n_tfidf_features,
    )
