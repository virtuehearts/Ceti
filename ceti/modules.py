"""CETI Modules — AIML extended with module headers, facts, provenance.

A module file is AIML plus four extensions:

* a ``<module>`` header carrying domain / name / version / source /
  confidence;
* explicit ``<fact subject=... predicate=... object=.../>`` tags;
* provenance on everything the module contributes (source + confidence);
* domains as directories: ``<domain>/*.aiml`` under a modules root.

Plain AIML files (no ``<module>`` wrapper) still load as legacy modules
with ``legacy=True``, ``confidence=0.6`` and ``source="legacy"``.

Stdlib only. Fail-closed: malformed XML or an unreadable file makes
:func:`load_module` return ``None`` instead of raising, and
:func:`load_modules` silently skips such files.
"""
from __future__ import annotations

import gzip
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

# Confidence assigned to legacy (header-less) AIML files. They have no
# provenance of their own, so they rank below curated header modules.
LEGACY_CONFIDENCE = 0.6
LEGACY_SOURCE = "legacy"

# Default confidence for a header module that omits the attribute.
# An explicit header is a curated statement; it defaults to full weight.
HEADER_DEFAULT_CONFIDENCE = 1.0

# Keyword rules for inferring a legacy file's domain from its filename.
# Checked in order; first match wins. Subdirectory name always wins over
# this inference (see load_modules).
_DOMAIN_KEYWORDS = (
    ("bbs", ("fidonet", "bbs", "board", "sysop", "usenet", "phrack", "phreak")),
    ("computing", ("commodore", "vic-20", "pet", "c64", "morris")),
    ("demoscene", ("demoscene", "ansi")),
    ("telecom", ("modem", "phone", "hayes")),
)


@dataclass
class Module:
    """One parsed module file: header + facts + Q&A patterns."""

    domain: str
    name: str
    version: str
    source: str
    confidence: float
    facts: list = field(default_factory=list)
    patterns: list = field(default_factory=list)
    path: str = ""
    legacy: bool = False
    domain_explicit: bool = False
    # domain_explicit: True when an explicit <module domain="...">
    # attribute supplied the domain (beats the directory); False
    # when the domain came from inference (directory beats it).

    # facts: list of (subject, predicate, object) tuples.
    # patterns: list of (pattern, template) tuples.


def infer_domain(filename: str) -> str:
    """Infer a legacy file's domain from its filename (keyword rules)."""
    base = os.path.basename(filename).lower()
    for domain, keywords in _DOMAIN_KEYWORDS:
        if any(kw in base for kw in keywords):
            return domain
    return "general"


def _stem(path: str) -> str:
    """File stem without .aiml / .aiml.gz suffixes."""
    base = os.path.basename(path)
    if base.endswith(".gz"):
        base = base[: -len(".gz")]
    if base.endswith(".aiml"):
        base = base[: -len(".aiml")]
    return base


def _text(elem: ET.Element | None) -> str:
    """All text under an element (handles nested tags), stripped."""
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()


def _parse_categories(root: ET.Element) -> list:
    """Collect (pattern, template) pairs from <category> elements."""
    pairs = []
    for cat in root.iter("category"):
        pattern = _text(cat.find("pattern"))
        template = _text(cat.find("template"))
        if pattern or template:
            pairs.append((pattern, template))
    return pairs


def _parse_facts(root: ET.Element) -> list:
    """Collect (subject, predicate, object) triples from <fact> tags."""
    facts = []
    for el in root.iter("fact"):
        subject = (el.get("subject") or "").strip()
        predicate = (el.get("predicate") or "").strip()
        obj = (el.get("object") or "").strip()
        if subject and predicate and obj:
            facts.append((subject, predicate, obj))
    return facts


def _to_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def load_module(path) -> Module | None:
    """Parse one module file. Returns None (never raises) on failure.

    A file whose root element is ``<module>`` is parsed as a header
    module: its attributes supply domain/name/version/source/confidence
    and its ``<fact>`` / ``<category>`` descendants supply the content.
    Anything else is a legacy AIML file: categories are parsed, the
    domain is inferred from the filename, and provenance is
    ``source="legacy"``, ``confidence=0.6``.
    """
    try:
        path_str = os.fspath(path)
        if path_str.endswith(".gz"):
            fh = gzip.open(path_str, "rt", encoding="utf-8")
        else:
            fh = open(path_str, "r", encoding="utf-8")
        with fh:
            root = ET.parse(fh).getroot()
    except (OSError, ET.ParseError, UnicodeDecodeError, EOFError,
            gzip.BadGzipFile, ValueError):
        return None

    stem = _stem(path_str)
    if root.tag == "module":
        header_domain = (root.get("domain") or "").strip()
        domain = header_domain or infer_domain(stem)
        return Module(
            domain=domain,
            name=(root.get("name") or "").strip() or stem,
            version=(root.get("version") or "").strip() or "1",
            source=(root.get("source") or "").strip(),
            confidence=_to_float(root.get("confidence"),
                                 HEADER_DEFAULT_CONFIDENCE),
            facts=_parse_facts(root),
            patterns=_parse_categories(root),
            path=path_str,
            legacy=False,
            domain_explicit=bool(header_domain),
        )

    # No <module> wrapper: legacy AIML file.
    return Module(
        domain=infer_domain(stem),
        name=stem,
        version="1",
        source=LEGACY_SOURCE,
        confidence=LEGACY_CONFIDENCE,
        facts=_parse_facts(root),
        patterns=_parse_categories(root),
        path=path_str,
        legacy=True,
    )


def load_modules(dir) -> list:
    """Load every module under a directory, recursively.

    Top-level ``*.aiml`` files load as legacy flat cassettes; files in a
    subdirectory ``<domain>/*.aiml`` take the subdirectory name as their
    domain ONLY when the file has no explicit header domain — an explicit
    ``<module domain="...">`` always wins, the directory beats filename
    inference. ``*.aiml.gz`` files load transparently. Non-AIML files
    and malformed files are skipped. A missing/unreadable directory
    yields an empty list.
    """
    try:
        root_dir = os.fspath(dir)
    except TypeError:
        return []
    found = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for filename in sorted(filenames):
            if not (filename.endswith(".aiml") or filename.endswith(".aiml.gz")):
                continue
            full = os.path.join(dirpath, filename)
            module = load_module(full)
            if module is None:
                continue
            rel = os.path.relpath(full, root_dir)
            parts = rel.split(os.sep)
            if len(parts) > 1 and not module.domain_explicit:
                # File lives in a subdirectory and has no explicit
                # header domain: the top-level subdir name is the
                # domain (beats filename inference, loses to an
                # explicit header).
                module.domain = parts[0]
            found.append(module)
    found.sort(key=lambda m: m.path)
    return found


def module_facts(module: Module) -> list:
    """A module's facts as dicts carrying the module's provenance."""
    return [
        {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "source": module.source,
            "confidence": module.confidence,
        }
        for (subject, predicate, obj) in module.facts
    ]
