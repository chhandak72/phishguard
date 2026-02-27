"""
test_preprocessing.py
---------------------
Unit tests for text cleaning and feature extraction utilities.

Run with:
    pytest backend/tests/test_preprocessing.py -v
"""

import sys
from pathlib import Path

import pytest

# Ensure backend/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.preprocessing import clean_text, combine_subject_body, safe_str
from utils.feature_engineering import (
    extract_features,
    get_feature_names,
    map_risk_level,
    build_reason_string,
    MetadataFeatureExtractor,
)
import pandas as pd


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_returns_string(self):
        assert isinstance(clean_text("Hello World"), str)

    def test_none_input(self):
        assert clean_text(None) == ""

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_lowercases(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_strips_html_tags(self):
        result = clean_text("<b>Bold</b> text <a href='x'>link</a>")
        assert "<" not in result
        assert "bold" in result
        assert "text" in result
        assert "link" in result

    def test_decodes_html_entities(self):
        result = clean_text("Price &amp; value &lt;10&gt;")
        assert "&amp;" not in result
        assert "price" in result

    def test_collapses_whitespace(self):
        result = clean_text("hello   \n\t  world")
        assert "  " not in result
        assert result == "hello world"

    def test_removes_email_headers(self):
        text = "From: sender@example.com\nSubject: Test\n\nBody here"
        result = clean_text(text)
        # Header lines should be stripped; body content should remain
        assert "body here" in result

    def test_boilerplate_removed(self):
        text = "Please unsubscribe if you do not wish to receive"
        result = clean_text(text)
        # Boilerplate pattern replaced with space
        assert "unsubscribe" not in result


# ---------------------------------------------------------------------------
# combine_subject_body
# ---------------------------------------------------------------------------


class TestCombineSubjectBody:
    def test_both_present(self):
        result = combine_subject_body("Win a prize", "Click here to claim")
        assert "win" in result
        assert "claim" in result

    def test_subject_doubled(self):
        # Subject should appear twice (upweighting trick)
        result = combine_subject_body("urgent", "please act now")
        assert result.count("urgent") >= 2

    def test_no_subject(self):
        result = combine_subject_body(None, "just a body")
        assert "just" in result

    def test_no_body(self):
        result = combine_subject_body("just a subject", None)
        assert "just" in result

    def test_both_empty(self):
        assert combine_subject_body("", "") == ""

    def test_both_none(self):
        assert combine_subject_body(None, None) == ""


# ---------------------------------------------------------------------------
# safe_str
# ---------------------------------------------------------------------------


class TestSafeStr:
    def test_none(self):
        assert safe_str(None) == ""

    def test_nan(self):
        import math
        assert safe_str(float("nan")) == ""

    def test_int(self):
        assert safe_str(42) == "42"

    def test_string(self):
        assert safe_str("hello") == "hello"


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    def test_returns_all_keys(self):
        feats = extract_features("subject", "body text")
        expected = set(get_feature_names())
        assert expected.issubset(set(feats.keys()))

    def test_detects_url(self):
        feats = extract_features("", "Visit http://example.com now")
        assert feats["has_url"] == 1
        assert feats["num_urls"] >= 1

    def test_no_url(self):
        feats = extract_features("Hello", "No links here at all.")
        assert feats["has_url"] == 0
        assert feats["num_urls"] == 0

    def test_detects_ip_url(self):
        feats = extract_features("", "Go to http://192.168.1.1/phish")
        assert feats["has_ip_in_url"] == 1

    def test_detects_html_tags(self):
        feats = extract_features("", "<b>Bold text</b>")
        assert feats["has_html_tags"] == 1

    def test_no_html_tags(self):
        feats = extract_features("", "Plain text email")
        assert feats["has_html_tags"] == 0

    def test_counts_exclamations(self):
        feats = extract_features("", "Win!!! Act now!!!")
        assert feats["num_exclamations"] == 6

    def test_detects_suspicious_words(self):
        feats = extract_features("", "Please verify your account password urgently")
        assert feats["has_suspicious_words"] == 1
        assert feats["num_suspicious_words"] >= 3

    def test_length_features(self):
        subj = "Test subject"
        body = "Some body text"
        feats = extract_features(subj, body)
        assert feats["length_subject"] == len(subj)
        assert feats["length_body"] == len(body)

    def test_sender_domain_mismatch_detected(self):
        feats = extract_features(
            "Update required",
            "Click http://evil.xyz/login",
            sender="From: admin@mybank.com",
        )
        assert feats["sender_domain_mismatch"] == 1

    def test_sender_domain_match(self):
        feats = extract_features(
            "Update required",
            "Click http://mybank.com/login",
            sender="From: admin@mybank.com",
        )
        assert feats["sender_domain_mismatch"] == 0


# ---------------------------------------------------------------------------
# map_risk_level
# ---------------------------------------------------------------------------


class TestMapRiskLevel:
    def test_low(self):
        assert map_risk_level(0.10) == "Low"
        assert map_risk_level(0.00) == "Low"
        assert map_risk_level(0.32) == "Low"

    def test_medium(self):
        assert map_risk_level(0.33) == "Medium"
        assert map_risk_level(0.50) == "Medium"
        assert map_risk_level(0.65) == "Medium"

    def test_high(self):
        assert map_risk_level(0.66) == "High"
        assert map_risk_level(0.90) == "High"
        assert map_risk_level(1.00) == "High"


# ---------------------------------------------------------------------------
# MetadataFeatureExtractor (sklearn transformer)
# ---------------------------------------------------------------------------


class TestMetadataFeatureExtractor:
    def test_transform_shape(self):
        extractor = MetadataFeatureExtractor()
        df = pd.DataFrame(
            [
                {"subject": "Win a prize!", "body": "Click http://win.xyz now"},
                {"subject": "Hello", "body": "Meeting at 3pm"},
            ]
        )
        result = extractor.fit_transform(df)
        assert result.shape == (2, len(get_feature_names()))

    def test_feature_names(self):
        extractor = MetadataFeatureExtractor()
        names = extractor.get_feature_names_out()
        assert list(names) == get_feature_names()

    def test_non_dataframe_raises(self):
        extractor = MetadataFeatureExtractor()
        with pytest.raises(ValueError):
            extractor.transform([[1, 2, 3]])
