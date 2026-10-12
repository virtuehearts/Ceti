"""Minimal AIML 1.0 brain — stdlib only.

Supports the subset the framework needs:
  * <category><pattern>..</pattern><template>..</template></category>
  * pattern wildcards: * (0+ words) and _ (1+ words)
  * template tags: <random><li>, <srai>, <star/>, <br/>, <person>

Deliberately NOT a full ALICE engine (no <set>/<get>, <condition>,
<that>/<topic>). Anything unmatched returns None and the decision
layer falls through to the next tier — no catch-all wildcards here.

learn(question, answer) appends a normalized category to the learned
file and rewrites it, so the file is always valid XML.
"""
from __future__ import annotations

import os
import random
import re
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

_WS_PAT = re.compile(r"\s+")

# articles carry no intent: "what is a bbs" and "what is bbs" are the
# same question. Dropped from both sides so the instant tier is
# article-insensitive.
_FILLER = {"A", "AN", "THE"}


def _drop_filler(words: list[str]) -> list[str]:
    return [w for w in words if w not in _FILLER]


def normalize_input(text: str) -> str:
    """Normalize user input for matching (wildcards stripped)."""
    text = text.upper()
    text = re.sub(r"[^\w\s]", " ", text)
    words = _WS_PAT.sub(" ", text).strip().split(" ")
    return " ".join(_drop_filler(words))


def normalize_pattern(text: str) -> str:
    """Normalize a pattern, keeping * and _ wildcards."""
    text = text.upper()
    text = re.sub(r"[^\w\s\*\_]", " ", text)
    words = _WS_PAT.sub(" ", text).strip().split(" ")
    return " ".join(_drop_filler(words))


def _pattern_to_regex(pattern: str) -> re.Pattern:
    parts: list[str] = []
    for ch in pattern:
        if ch == "*":
            parts.append("(.*?)")
        elif ch == "_":
            parts.append("(.+?)")
        elif ch == " ":
            parts.append(r"\s+")
        else:
            parts.append(re.escape(ch))
    return re.compile("^" + "".join(parts) + "$")


def _specificity(pattern: str) -> tuple[int, int]:
    """Sort key: fewer wildcards first, then longer literal text."""
    wilds = pattern.count("*") + pattern.count("_")
    literal = len(pattern.replace("*", "").replace("_", "").replace(" ", ""))
    return (wilds, -literal)


_PERSON_MAP = [
    (re.compile(r"\bI'M\b"), "YOU ARE"),
    (re.compile(r"\bI AM\b"), "YOU ARE"),
    (re.compile(r"\bME\b"), "YOU"),
    (re.compile(r"\bMY\b"), "YOUR"),
    (re.compile(r"\bMINE\b"), "YOURS"),
    (re.compile(r"\bI\b"), "YOU"),
    (re.compile(r"\bYOU ARE\b"), "I AM"),
    (re.compile(r"\bYOUR\b"), "MY"),
    (re.compile(r"\bYOURS\b"), "MINE"),
    (re.compile(r"\bYOU\b"), "I"),
]


def _person(text: str) -> str:
    for rx, rep in _PERSON_MAP:
        text = rx.sub(rep, text)
    return text


class AimlBrain:
    # load priority: lower number wins on duplicate patterns
    PRIO_LEARNED = 0
    PRIO_CASSETTE = 1
    PRIO_SEED = 2

    def __init__(self, seed_path: str | None = None,
                 learned_path: str | None = None,
                 max_learned: int = 500,
                 extra_paths: list[str] | None = None):
        """extra_paths: cassettes — read-only AIML files loaded between
        the seed and the learned file in priority."""
        self.max_learned = max_learned
        self.learned_path = learned_path
        # pattern -> (template element, priority)
        self._exact: dict[str, tuple[ET.Element, int]] = {}
        self._wild: list[tuple[str, re.Pattern, ET.Element, int]] = []
        # learned categories in insertion order: (pattern, template_text)
        self._learned: list[tuple[str, str]] = []
        if seed_path and os.path.exists(seed_path):
            self._load_file(seed_path, learned=False,
                            priority=self.PRIO_SEED)
        for p in extra_paths or []:
            if os.path.exists(p):
                self._load_file(p, learned=False,
                                priority=self.PRIO_CASSETTE)
        if learned_path:
            if os.path.exists(learned_path):
                self._load_file(learned_path, learned=True,
                                priority=self.PRIO_LEARNED)
            else:
                self._write_learned()  # create the empty file

    # ---- loading ------------------------------------------------------
    def _load_file(self, path: str, learned: bool,
                   priority: int = PRIO_SEED) -> None:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            return  # corrupt file: skip, never crash the brain
        for cat in root.findall("category"):
            pat_el = cat.find("pattern")
            tpl_el = cat.find("template")
            if pat_el is None or tpl_el is None:
                continue
            pattern = normalize_pattern(pat_el.text or "")
            if not pattern:
                continue
            if learned:
                text = self._flatten(tpl_el)
                if pattern not in {p for p, _ in self._learned}:
                    self._learned.append((pattern, text))
            self._register(pattern, tpl_el, priority=priority)

    def _register(self, pattern: str, tpl_el: ET.Element,
                  priority: int = PRIO_SEED, override: bool = False) -> None:
        if "*" in pattern or "_" in pattern:
            rx = _pattern_to_regex(pattern)
            for i, (p, _rx, _t, pr) in enumerate(self._wild):
                if p == pattern:
                    if override or priority < pr:
                        self._wild[i] = (pattern, rx, tpl_el, priority)
                    break
            else:
                self._wild.append((pattern, rx, tpl_el, priority))
            self._wild.sort(key=lambda e: (e[3], _specificity(e[0])))
        else:
            cur = self._exact.get(pattern)
            if cur is None or override or priority < cur[1]:
                self._exact[pattern] = (tpl_el, priority)

    def _flatten(self, el: ET.Element) -> str:
        """Best-effort plain text of a template (for persistence)."""
        return _WS_PAT.sub(" ", "".join(el.itertext())).strip()

    # ---- answering ----------------------------------------------------
    def respond(self, text: str, _depth: int = 0) -> str | None:
        if _depth > 8:
            return None
        norm = normalize_input(text)
        if not norm:
            return None
        hit = self._exact.get(norm)
        tpl = hit[0] if hit else None
        stars: list[str] = []
        if tpl is None:
            for _pat, rx, t, _pr in self._wild:
                m = rx.match(norm)
                if m:
                    tpl = t
                    stars = [g.strip() for g in m.groups()]
                    break
        if tpl is None:
            return None
        return self._render(tpl, stars, _depth).strip() or None

    def _render(self, el: ET.Element, stars: list[str], depth: int) -> str:
        tag = (el.tag or "").lower()
        if tag == "random":
            options = el.findall("li")
            if not options:
                return ""
            return self._render(random.choice(options), stars, depth)
        if tag == "srai":
            inner = self._inner_text(el, stars, depth)
            return self.respond(inner, depth + 1) or ""
        if tag == "star":
            idx = el.get("index")
            try:
                i = int(idx) - 1 if idx else 0
            except (TypeError, ValueError):
                i = 0
            return stars[i] if 0 <= i < len(stars) else ""
        if tag == "br":
            return "\n"
        if tag == "person":
            return _person(self._inner_text(el, stars, depth))
        return self._inner_text(el, stars, depth)

    def _inner_text(self, el: ET.Element, stars: list[str],
                    depth: int) -> str:
        out = [el.text or ""]
        for child in el:
            out.append(self._render(child, stars, depth))
            out.append(child.tail or "")
        return "".join(out)

    # ---- learning ------------------------------------------------------
    def learn(self, question: str, answer: str) -> bool:
        """Promote a Q&A pair. Returns False if rejected/duplicate."""
        pattern = normalize_pattern(question)
        template = _WS_PAT.sub(" ", answer).strip()
        if not pattern or not template or len(template) > 2000:
            return False
        if any(p == pattern for p, _ in self._learned):
            return False  # already in the learned file
        # NOTE: a pattern that exists in the seed or a cassette is NOT
        # rejected — the learned file outranks them (priority 0), so
        # live learning overrides pre-trained content.
        self._learned.append((pattern, template))
        # cap: drop oldest first, brain stays tiny
        while len(self._learned) > self.max_learned:
            old, _ = self._learned.pop(0)
            self._exact.pop(old, None)
        tpl_el = ET.fromstring(
            "<template>" + escape(template) + "</template>")
        self._register(pattern, tpl_el,
                       priority=self.PRIO_LEARNED, override=True)
        self._write_learned()
        return True

    def _write_learned(self) -> None:
        if not self.learned_path:
            return
        lines = ['<?xml version="1.0" encoding="UTF-8"?>',
                 "<!-- learned.aiml — auto-promoted by the brain's "
                 "relevance classifier. Safe to delete. -->",
                 '<aiml version="1.0">']
        for pattern, template in self._learned:
            lines.append("<category>")
            lines.append(f"<pattern>{escape(pattern)}</pattern>")
            lines.append(f"<template>{escape(template)}</template>")
            lines.append("</category>")
        lines.append("</aiml>")
        tmp = self.learned_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, self.learned_path)

    def count(self) -> int:
        return len(self._exact) + len(self._wild)

    def learned_template(self, pattern: str) -> str | None:
        """Template stored in the learned file for a normalized pattern,
        or None. Used by dataset merging for conflict detection."""
        for p, t in self._learned:
            if p == pattern:
                return t
        return None
