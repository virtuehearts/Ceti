"""PII stripper — heuristic, stdlib-only.

Runs on every query BEFORE it touches the cloud, and on every Q&A pair
BEFORE it is persisted to AIML or the memory index. The local Qwen keeps
the raw query (it's Jero's own iron — private); nothing personal ever
leaves the box or lands in a file.

This is heuristic redaction, not a guarantee. When in doubt it
over-redacts. Treat `found` as advisory.
"""
from __future__ import annotations

import re

# (label, compiled pattern) — order matters: most specific first.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")),
    ("CREDIT_CARD", re.compile(
        r"\b(?:\d[ -]?){13,19}\b")),  # loose; refined below
    ("SSN", re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(
        r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}(?!\d)")),
    ("IPV4", re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("IPV6", re.compile(
        r"\b(?:[0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}\b")),
    ("API_KEY", re.compile(
        r"\b(?:sk-|xox[bap]-|ghp_|gho_|AKIA|hf_)[A-Za-z0-9_\-]{8,}\b")),
    ("SECRET_TOKEN", re.compile(
        r"\b[A-Za-z0-9_\-]{32,}\b")),  # long high-entropy-ish token
    ("STREET_ADDRESS", re.compile(
        r"\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\s+"
        r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|"
        r"Court|Ct|Terrace|Place|Pl|Way)\b")),
    ("POSTAL_CODE", re.compile(
        r"\b[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z]\s?\d[ABCEGHJ-NPRSTV-Z]\d\b",
        re.IGNORECASE)),  # Canadian
    ("NAME_HONORIFIC", re.compile(
        r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b")),
    ("NAME_INTRO", re.compile(
        r"\b(?:my name is|i am|i'm|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        re.IGNORECASE)),
]

# signals that the text is about the user personally (used by the
# classifier to refuse promotion even when no regex fired)
_PERSONAL_HINTS = re.compile(
    r"\b(my|mine)\s+(name|address|phone|number|email|password|birthday|"
    r"dob|ssn|salary|account|location|home)\b"
    r"|\bi\s+live\s+(at|in)\b"
    r"|\bcall\s+me\b",
    re.IGNORECASE,
)


def _looks_like_card(s: str) -> bool:
    digits = re.sub(r"\D", "", s)
    return 13 <= len(digits) <= 19 and digits[0] in "3456"


def strip(text: str) -> tuple[str, list[str]]:
    """Return (redacted_text, labels_found).

    Redactions look like [REDACTED_EMAIL]. The original text is never
    modified in place — callers decide which copy to use downstream.
    """
    found: list[str] = []
    out = text
    for label, rx in _PATTERNS:
        def _rep(m: re.Match, _label: str = label) -> str:
            if _label == "CREDIT_CARD" and not _looks_like_card(m.group(0)):
                return m.group(0)  # probably just a long number
            if _label not in found:
                found.append(_label)
            return f"[REDACTED_{_label}]"
        out = rx.sub(_rep, out)
    return out, found


def looks_personal(text: str) -> bool:
    """Heuristic: is this query about the user's own private details?"""
    cleaned, found = strip(text)
    if found:
        return True
    return bool(_PERSONAL_HINTS.search(text))
