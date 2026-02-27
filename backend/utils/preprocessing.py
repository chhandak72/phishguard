"""
preprocessing.py
----------------
Text cleaning utilities for the PhishGuard pipeline.
These are applied BEFORE TF-IDF vectorisation.

Authors: Aayush Paudel 2260308, Chhandak Mukherjee 2260357, Arnav Makhija 2260344
"""

from __future__ import annotations

import html
import re
import string
import unicodedata
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Boilerplate patterns that add noise but carry little signal
_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"[\-=_]{3,}", re.IGNORECASE),   # separator lines  --- / ===
    re.compile(r"(unsubscribe|opt.?out|to be removed)", re.IGNORECASE),
    re.compile(r"\bthis (email|message) (was sent|is intended)", re.IGNORECASE),
    re.compile(r"\bconfidentiality notice\b", re.IGNORECASE),
    re.compile(r"(copyright|all rights reserved)", re.IGNORECASE),
]

# Common email header artefacts (From:, To:, Date:, MIME-Version:, …)
_HEADER_RE = re.compile(
    r"^(from|to|cc|bcc|date|subject|content-type|mime-version|"
    r"message-id|return-path|received|x-[a-z-]+):.*$",
    re.IGNORECASE | re.MULTILINE,
)

# Collapse multiple whitespace / newlines
_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)

# HTML tag removal
_HTML_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_text(text: Optional[str]) -> str:
    """Return a clean, normalised version of *text* suitable for TF-IDF.

    Steps
    -----
    1. Guard against None / non-string input.
    2. Unicode normalisation (NFKC) so ligatures are decomposed.
    3. Decode HTML entities (``&amp;`` → ``&``).
    4. Strip inline HTML tags.
    5. Remove email header artefacts.
    6. Remove boilerplate patterns.
    7. Lowercase.
    8. Collapse whitespace.

    Parameters
    ----------
    text:
        Raw email text (subject or body, or their concatenation).

    Returns
    -------
    str
        Cleaned text string (may be empty string if input was empty/None).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Unicode normalisation
    text = unicodedata.normalize("NFKC", text)

    # 2. HTML entity decoding
    text = html.unescape(text)

    # 3. Strip HTML tags
    text = _HTML_TAG_RE.sub(" ", text)

    # 4. Remove email header lines
    text = _HEADER_RE.sub(" ", text)

    # 5. Remove boilerplate
    for pat in _BOILERPLATE_PATTERNS:
        text = pat.sub(" ", text)

    # 6. Lowercase
    text = text.lower()

    # 7. Remove punctuation that is not part of a URL or word character
    #    Keep alphanumeric, spaces and a few connector chars.
    text = re.sub(r"[^\w\s\-@./:]", " ", text)

    # 8. Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text


def combine_subject_body(subject: Optional[str], body: Optional[str]) -> str:
    """Combine subject and body into a single string for vectorisation.

    The subject is weighted by prepending it twice so the TF-IDF model
    gives slightly more weight to subject tokens — a common NLP trick for
    short but highly discriminative fields.

    Parameters
    ----------
    subject:
        Email subject line.
    body:
        Email body text.

    Returns
    -------
    str
        Cleaned combined text.
    """
    subj = clean_text(subject)
    bod = clean_text(body)
    # Prepend subject twice for slight upweighting
    parts = ([subj, subj] if subj else []) + ([bod] if bod else [])
    return " ".join(parts)


def safe_str(val) -> str:
    """Convert *val* to str, returning '' for NaN / None."""
    if val is None:
        return ""
    try:
        import math
        if math.isnan(float(val)):  # type: ignore[arg-type]
            return ""
    except (TypeError, ValueError):
        pass
    return str(val)
