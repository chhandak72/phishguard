"""
test_api.py
-----------
Integration tests for the PhishGuard FastAPI application.

Uses FastAPI TestClient (backed by httpx) to send real HTTP requests
against the app without starting a server process.

IMPORTANT: These tests assume the model is NOT loaded (so we test the
           503 / error handling paths), plus the happy-path with a
           mocked model. To run against a real trained model, ensure
           backend/models/stacking_pipeline.joblib exists before running.

Run with:
    pytest backend/tests/test_api.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure backend/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client_no_model():
    """TestClient with no model loaded (simulates cold start without training)."""
    # Import app fresh; since model likely doesn't exist in CI, this is fine
    from fastapi.testclient import TestClient
    import app as app_module

    # Override model state
    with patch.object(app_module, "_MODEL", None), patch.object(
        app_module, "_MODEL_LOAD_ERROR", "Model not loaded for testing."
    ):
        with TestClient(app_module.app) as c:
            yield c


@pytest.fixture(scope="module")
def mock_pipeline():
    """Return a mock pipeline that returns a fixed phishing probability of 0.87."""
    import numpy as np

    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[0.13, 0.87]])
    mock.predict.return_value = ["Phishing"]
    return mock


@pytest.fixture(scope="module")
def client_with_mock_model(mock_pipeline):
    """TestClient with a mocked pipeline injected."""
    from fastapi.testclient import TestClient
    import app as app_module

    with patch.object(app_module, "_MODEL", mock_pipeline), patch.object(
        app_module, "_MODEL_LOAD_ERROR", None
    ), patch.object(
        app_module,
        "_MODEL_METADATA",
        {
            "model_version": "1.0.0-test",
            "trained_at": "2026-02-27T00:00:00+00:00",
            "git_commit": "abc1234",
            "test_roc_auc": 0.99,
            "test_accuracy": 0.97,
        },
    ):
        with TestClient(app_module.app) as c:
            yield c


# ---------------------------------------------------------------------------
# /health tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_no_model_returns_503(self, client_no_model):
        resp = client_no_model.get("/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unhealthy"

    def test_health_with_model_returns_200(self, client_with_mock_model):
        resp = client_with_mock_model.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model_version" in data
        assert "trained_at" in data

    def test_health_includes_git_commit(self, client_with_mock_model):
        resp = client_with_mock_model.get("/health")
        data = resp.json()
        assert "git_commit" in data


# ---------------------------------------------------------------------------
# /predict tests
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    def test_predict_no_model_returns_503(self, client_no_model):
        resp = client_no_model.post(
            "/predict",
            json={"subject": "Win a prize", "body": "Click here to claim your reward!"},
        )
        assert resp.status_code == 503

    def test_predict_empty_input_returns_400(self, client_with_mock_model):
        resp = client_with_mock_model.post("/predict", json={"subject": "", "body": ""})
        assert resp.status_code == 400

    def test_predict_missing_body_uses_subject(self, client_with_mock_model):
        resp = client_with_mock_model.post(
            "/predict",
            json={"subject": "Important security alert"},
        )
        assert resp.status_code == 200

    def test_predict_returns_expected_fields(self, client_with_mock_model):
        resp = client_with_mock_model.post(
            "/predict",
            json={
                "subject": "Verify your account immediately",
                "body": "Dear user, click http://evil.xyz to verify your password",
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        assert "label" in data
        assert "phishing_probability" in data
        assert "risk_level" in data
        assert "top_reasons" in data
        assert "reason_summary" in data
        assert "processing_time_ms" in data

    def test_predict_label_is_phishing(self, client_with_mock_model):
        """Mock returns prob=0.87 → should be 'Phishing'."""
        resp = client_with_mock_model.post(
            "/predict",
            json={"subject": "Urgent", "body": "Click now"},
        )
        data = resp.json()
        assert data["label"] == "Phishing"
        assert data["phishing_probability"] == pytest.approx(0.87, abs=0.01)

    def test_predict_risk_level_high(self, client_with_mock_model):
        """prob=0.87 → risk=High."""
        resp = client_with_mock_model.post(
            "/predict",
            json={"subject": "Urgent", "body": "Click now"},
        )
        data = resp.json()
        assert data["risk_level"] == "High"

    def test_predict_top_reasons_are_list(self, client_with_mock_model):
        resp = client_with_mock_model.post(
            "/predict",
            json={"subject": "Prize winner", "body": "Visit http://scam.biz/win"},
        )
        data = resp.json()
        assert isinstance(data["top_reasons"], list)
        assert len(data["top_reasons"]) <= 3

    def test_predict_top_reason_has_required_fields(self, client_with_mock_model):
        resp = client_with_mock_model.post(
            "/predict",
            json={"subject": "Prize winner", "body": "Visit http://scam.biz/win"},
        )
        data = resp.json()
        for reason in data["top_reasons"]:
            assert "feature" in reason
            assert "value" in reason
            assert "score_contribution" in reason

    def test_predict_probability_in_range(self, client_with_mock_model):
        resp = client_with_mock_model.post(
            "/predict",
            json={"subject": "Hello", "body": "Meeting tomorrow"},
        )
        data = resp.json()
        assert 0.0 <= data["phishing_probability"] <= 1.0

    def test_predict_with_raw_email(self, client_with_mock_model):
        raw = (
            "From: phisher@evil.com\n"
            "To: victim@example.com\n"
            "Subject: Your account is compromised\n"
            "\n"
            "Please verify your account at http://evil.com/login"
        )
        resp = client_with_mock_model.post(
            "/predict",
            json={"email_raw": raw},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /analyze alias
# ---------------------------------------------------------------------------


class TestAnalyzeEndpoint:
    def test_analyze_alias_works(self, client_with_mock_model):
        resp = client_with_mock_model.post(
            "/analyze",
            json={"subject": "Click here", "body": "Limited offer!"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "label" in data
