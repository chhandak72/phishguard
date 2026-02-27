"""
feature_engineering.py
-----------------------
Metadata / structural feature extraction for the PhishGuard pipeline.

All features are computable from the raw text (subject + body) alone, so
the pipeline does not require any email header parsing to work at inference
time. Header-based features (e.g. sender domain mismatch) are extracted
where available and gracefully skipped otherwise.

Authors: Aayush Paudel 2260308, Chhandak Mukherjee 2260357, Arnav Makhija 2260344
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Compiled regex patterns (module-level for efficiency)
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"(https?://|www\.)[^\s<>\"']+",
    re.IGNORECASE,
)

_IP_IN_URL_RE = re.compile(
    r"(https?://|www\.)\d{1,3}(\.\d{1,3}){3}",
    re.IGNORECASE,
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")

_EMAIL_FROM_RE = re.compile(
    r"from:.*?@([^\s>\"']+)",
    re.IGNORECASE,
)
_EMAIL_LINK_DOMAIN_RE = re.compile(
    r"https?://([^/\s>\"']+)",
    re.IGNORECASE,
)

# Extracts the registrable domain (hostname) from a URL
_DOMAIN_FROM_URL_RE = re.compile(
    r"https?://([^/\s>\"'?#]+)",
    re.IGNORECASE,
)

# Words/phrases strongly associated with phishing
_SUSPICIOUS_WORDS = frozenset(
    [
        "verify",
        "verification",
        "click",
        "password",
        "account",
        "urgent",
        "update",
        "confirm",
        "login",
        "bank",
        "credit",
        "debit",
        "social security",
        "ssn",
        "suspend",
        "suspended",
        "limited",
        "expire",
        "expir",
        "validate",
        "unauthorized",
        "immediate",
        "alert",
        "winner",
        "prize",
        "free",
        "gift",
        "congratulations",
        "lottery",
        "claim",
        "invoice",
        "bitcoin",
        "crypto",
        "wire transfer",
        "act now",
        "dear customer",
        "dear user",
    ]
)

# Exact suspicious phrases (regex-matched against lower-cased full text)
_SUSPICIOUS_PHRASES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"unusual (login|activity|sign.?in)",
        r"verify your (account|identity|email|details|information)",
        r"review your account",
        r"account (suspended|locked|compromised|at risk|limited)",
        r"confirm your (details|information|identity|password|account)",
        r"click (here|the link|below) to (verify|confirm|update|restore|reactivate)",
        r"your (account|password|access) (will be|has been) (suspended|terminated|locked|disabled)",
        r"(immediate|urgent|immediate) (action|attention|response) (required|needed)",
        r"(update|verify|validate) your (payment|billing|credit card) (information|details)",
        r"you have (won|been selected|been chosen)",
        r"(reset|recover) your password",
        r"(sign in|log in) to (verify|confirm|restore)",
        r"your (package|delivery|shipment|order) (is|has been) (held|delayed|suspended)",
    ]
]

# Well-known legitimate domains (whitelist for domain_not_whitelisted feature)
_TRUSTED_DOMAINS = frozenset([
    "google.com", "gmail.com", "youtube.com", "googleapis.com",
    "microsoft.com", "outlook.com", "office.com", "live.com", "hotmail.com",
    "apple.com", "icloud.com",
    "amazon.com", "aws.amazon.com",
    "facebook.com", "instagram.com", "twitter.com", "x.com", "linkedin.com",
    "github.com", "stackoverflow.com",
    "paypal.com", "ebay.com",
    "netflix.com", "spotify.com",
    "dropbox.com", "zoom.us",
    "wikipedia.org", "wikimedia.org",
])

# Keywords inside a domain name that suggest phishing
_PHISHING_DOMAIN_KEYWORDS = frozenset([
    "secure", "login", "verify", "account", "update", "bank", "confirm",
    "signin", "sign-in", "webscr", "ebayisapi", "paypal", "support",
    "service", "check", "validate", "billing", "security", "recover",
    "authenticate", "password", "credential",
])


# ---------------------------------------------------------------------------
# Domain analysis helpers
# ---------------------------------------------------------------------------

def _extract_domains_from_text(text: str) -> list[str]:
    """Return list of hostnames found in URLs within text."""
    return [m.lower().strip() for m in _DOMAIN_FROM_URL_RE.findall(text)]


def _get_registrable_domain(hostname: str) -> str:
    """Strip port and return the last two labels (e.g. 'sub.evil.com' → 'evil.com')."""
    hostname = hostname.split(":")[0]   # strip port
    parts = hostname.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else hostname


def _domain_entropy(domain: str) -> float:
    """Shannon entropy of the domain string (high entropy → likely DGA/random)."""
    if not domain:
        return 0.0
    counts = Counter(domain)
    total = len(domain)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _analyze_domains(text: str) -> dict:
    """Return aggregated domain-level signals from all URLs in text."""
    hostnames = _extract_domains_from_text(text)
    if not hostnames:
        return {
            "num_hyphens_in_domain": 0,
            "domain_length": 0,
            "domain_has_phishing_keywords": 0,
            "num_subdomains": 0,
            "domain_entropy": 0.0,
            "domain_not_whitelisted": 0,
        }

    # Aggregate across all URLs, take max (most suspicious URL wins)
    max_hyphens = 0
    max_length = 0
    max_keyword_hit = 0
    max_subdomains = 0
    max_entropy = 0.0
    any_not_whitelisted = 0

    for hostname in hostnames:
        hostname = hostname.split(":")[0]  # strip port
        parts = hostname.split(".")

        hyphens = hostname.count("-")
        length = len(hostname)
        keyword_hit = int(any(kw in hostname for kw in _PHISHING_DOMAIN_KEYWORDS))
        subdomains = max(0, len(parts) - 2)
        entropy = _domain_entropy(hostname)
        reg_domain = _get_registrable_domain(hostname)
        not_whitelisted = int(reg_domain not in _TRUSTED_DOMAINS)

        max_hyphens = max(max_hyphens, hyphens)
        max_length = max(max_length, length)
        max_keyword_hit = max(max_keyword_hit, keyword_hit)
        max_subdomains = max(max_subdomains, subdomains)
        max_entropy = max(max_entropy, entropy)
        any_not_whitelisted = max(any_not_whitelisted, not_whitelisted)

    return {
        "num_hyphens_in_domain": max_hyphens,
        "domain_length": max_length,
        "domain_has_phishing_keywords": max_keyword_hit,
        "num_subdomains": max_subdomains,
        "domain_entropy": round(max_entropy, 4),
        "domain_not_whitelisted": any_not_whitelisted,
    }


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def extract_features(subject: str, body: str, sender: Optional[str] = None) -> Dict[str, Any]:
    """Extract hand-crafted metadata features from an email.

    Parameters
    ----------
    subject:
        Raw or lightly cleaned subject line.
    body:
        Raw or lightly cleaned email body.
    sender:
        Optional raw ``From:`` header value (used for domain mismatch).

    Returns
    -------
    dict
        Mapping feature_name → numeric / boolean value.
        All booleans are cast to ``int`` (0/1) for sklearn compatibility.
    """
    text_full = (subject or "") + " " + (body or "")
    text_lower = text_full.lower()

    urls = _URL_RE.findall(text_full)
    num_urls = len(urls)
    has_url = int(num_urls > 0)

    # IP-based URL (common in phishing)
    has_ip_in_url = int(bool(_IP_IN_URL_RE.search(text_full)))

    # HTML tags in body
    has_html_tags = int(bool(_HTML_TAG_RE.search(body or "")))

    # Punctuation-based features
    num_exclamations = text_full.count("!")
    num_question_marks = text_full.count("?")
    num_dollars = text_full.count("$")

    # Digit density
    num_digits = sum(c.isdigit() for c in text_full)

    # Suspicious word hits
    num_suspicious = sum(1 for w in _SUSPICIOUS_WORDS if w in text_lower)
    has_suspicious_words = int(num_suspicious > 0)

    # Length features
    length_subject = len(subject or "")
    length_body = len(body or "")

    # Ratio: uppercase to total letters (SHOUTING is a phishing signal)
    letters = [c for c in text_full if c.isalpha()]
    upper_ratio = sum(1 for c in letters if c.isupper()) / max(len(letters), 1)

    # Sender domain mismatch:
    # Compare the From: domain with link domains in the body.
    # If the From: domain differs from ALL link domains → likely spoofed.
    sender_domain_mismatch = 0
    if sender:
        from_match = _EMAIL_FROM_RE.search(sender)
        if from_match:
            from_domain = from_match.group(1).lower().strip()
            link_domains = [
                m.lower().strip() for m in _EMAIL_LINK_DOMAIN_RE.findall(text_full)
            ]
            if link_domains:
                sender_domain_mismatch = int(
                    not any(from_domain in ld or ld in from_domain for ld in link_domains)
                )

    # ---- NEW: Domain-level analysis ----
    domain_feats = _analyze_domains(text_full)

    # ---- NEW: Suspicious phrase matching ----
    has_suspicious_phrase = int(
        any(p.search(text_lower) for p in _SUSPICIOUS_PHRASES)
    )
    num_suspicious_phrases = sum(
        1 for p in _SUSPICIOUS_PHRASES if p.search(text_lower)
    )

    return {
        # Original 14 features
        "has_url": has_url,
        "num_urls": num_urls,
        "has_ip_in_url": has_ip_in_url,
        "has_html_tags": has_html_tags,
        "num_exclamations": num_exclamations,
        "num_question_marks": num_question_marks,
        "num_dollars": num_dollars,
        "num_digits": num_digits,
        "num_suspicious_words": num_suspicious,
        "has_suspicious_words": has_suspicious_words,
        "sender_domain_mismatch": sender_domain_mismatch,
        "length_subject": length_subject,
        "length_body": length_body,
        "upper_ratio": round(upper_ratio, 4),
        # New domain features (6)
        "num_hyphens_in_domain": domain_feats["num_hyphens_in_domain"],
        "domain_length": domain_feats["domain_length"],
        "domain_has_phishing_keywords": domain_feats["domain_has_phishing_keywords"],
        "num_subdomains": domain_feats["num_subdomains"],
        "domain_entropy": domain_feats["domain_entropy"],
        "domain_not_whitelisted": domain_feats["domain_not_whitelisted"],
        # New phrase features (2)
        "has_suspicious_phrase": has_suspicious_phrase,
        "num_suspicious_phrases": num_suspicious_phrases,
    }


def get_feature_names() -> list[str]:
    """Return the ordered list of metadata feature names."""
    return [
        # Original 14
        "has_url",
        "num_urls",
        "has_ip_in_url",
        "has_html_tags",
        "num_exclamations",
        "num_question_marks",
        "num_dollars",
        "num_digits",
        "num_suspicious_words",
        "has_suspicious_words",
        "sender_domain_mismatch",
        "length_subject",
        "length_body",
        "upper_ratio",
        # New domain features (6)
        "num_hyphens_in_domain",
        "domain_length",
        "domain_has_phishing_keywords",
        "num_subdomains",
        "domain_entropy",
        "domain_not_whitelisted",
        # New phrase features (2)
        "has_suspicious_phrase",
        "num_suspicious_phrases",
    ]


def get_text_column(X: "pd.DataFrame") -> "pd.Series":
    """Extract the combined_text column from a DataFrame.

    Defined here (not in train.py) so that the FunctionTransformer pickled
    inside the sklearn Pipeline can be deserialized in any module context.
    """
    return X["combined_text"]


# ---------------------------------------------------------------------------
# sklearn Transformer wrapper
# ---------------------------------------------------------------------------


class MetadataFeatureExtractor(BaseEstimator, TransformerMixin):
    """sklearn-compatible transformer that extracts metadata features.

    Input
    -----
    X : pd.DataFrame with columns ``['subject', 'body']``
        (and optionally ``'sender'``).

    Output
    ------
    np.ndarray of shape ``(n_samples, n_features)``
    """

    def fit(self, X, y=None):  # noqa: N803
        return self  # stateless

    def transform(self, X, y=None):  # noqa: N803
        """Transform a DataFrame into a metadata feature matrix."""
        if isinstance(X, pd.DataFrame):
            rows = X.to_dict(orient="records")
        else:
            raise ValueError("MetadataFeatureExtractor expects a pandas DataFrame.")

        result = []
        for row in rows:
            subject = str(row.get("subject") or "")
            body = str(row.get("body") or "")
            sender = row.get("sender")
            feats = extract_features(subject, body, sender=sender)
            result.append([feats[k] for k in get_feature_names()])

        return np.array(result, dtype=np.float32)

    def get_feature_names_out(self):
        return np.array(get_feature_names())


# ---------------------------------------------------------------------------
# Risk-level helpers (shared between train.py and app.py)
# ---------------------------------------------------------------------------

RISK_THRESHOLDS = {"Low": 0.25, "Medium": 0.60}  # High = p >= 0.60


def map_risk_level(phish_prob: float) -> str:
    """Map a phishing probability to a risk label.

    Parameters
    ----------
    phish_prob:
        Model output probability for the phishing class (0.0 – 1.0).

    Returns
    -------
    str
        One of ``'Low'``, ``'Medium'``, ``'High'``.
    """
    if phish_prob < RISK_THRESHOLDS["Low"]:
        return "Low"
    elif phish_prob < RISK_THRESHOLDS["Medium"]:
        return "Medium"
    else:
        return "High"


def build_reason_string(
    subject: str,
    body: str,
    phish_prob: float,
) -> str:
    """Return a short human-readable reason for the risk level.

    This complements the SHAP-based top_reasons returned by the API.

    Parameters
    ----------
    subject, body:
        Raw email text.
    phish_prob:
        Phishing probability from the model.

    Returns
    -------
    str
    """
    feats = extract_features(subject, body)
    parts: list[str] = []

    if feats["has_url"]:
        parts.append(f"contains {feats['num_urls']} URL(s)")
    if feats["has_ip_in_url"]:
        parts.append("URL contains a raw IP address")
    if feats["has_suspicious_phrase"]:
        parts.append(f"{feats['num_suspicious_phrases']} suspicious phrase(s) detected")
    if feats["has_suspicious_words"]:
        parts.append(f"{feats['num_suspicious_words']} suspicious word(s) detected")
    if feats["domain_has_phishing_keywords"]:
        parts.append("domain name contains phishing keywords")
    if feats["domain_not_whitelisted"] and feats["has_url"]:
        parts.append("domain is not a known trusted site")
    if feats["num_hyphens_in_domain"] >= 2:
        parts.append(f"domain has {feats['num_hyphens_in_domain']} hyphens (suspicious)")
    if feats["num_exclamations"] >= 3:
        parts.append(f"{feats['num_exclamations']} exclamation marks")
    if feats["has_html_tags"]:
        parts.append("HTML markup in body")
    if feats["sender_domain_mismatch"]:
        parts.append("sender domain does not match link domains")
    if phish_prob >= 0.9:
        parts.append("very high phishing signature")

    if not parts:
        return "No strong phishing signals detected."
    return "; ".join(parts).capitalize() + "."
