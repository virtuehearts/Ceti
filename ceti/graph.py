"""Knowledge graph — a DERIVED INDEX built from module files.

Module files are the source of truth; this graph is a compiled index
over them (same relationship as AIML patterns over the fact base).
It generalizes the fact store (ceti/facts.py) with the same
semantics: conflict at (subject, predicate), confidence, supersede,
provenance.

Differences from FactStore:
- In-memory, dictionary-encoded (interned strings): each fact is
  stored as integer ids (subject_id, predicate_id, object_id) plus
  metadata. Repeated subjects/predicates/objects are stored once,
  which is what stats() proves.
- Predicates are NOT restricted to the closed facts.py vocabulary:
  module files may contribute any predicate (e.g. "located_in"),
  so the graph can chain entities (Paris -> France -> Europe).
  Facts learned via learn() still go through ceti.facts.extract, so
  they keep the closed vocabulary and its rendering.

Stdlib only. No I/O, no network, no cloud/oracle.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from ceti import facts as _facts


@dataclass
class _Entry:
    """One stored fact, dictionary-encoded."""
    subject_id: int
    predicate_id: int
    object_id: int
    confidence: float
    source: str
    created: float
    superseded: bool = False


class KnowledgeGraph:
    """Dictionary-encoded knowledge graph.

    Strings live once in ``_strings`` (``_ids`` maps string -> id);
    facts store integer ids plus metadata. Superseded facts are kept
    (audit trail) and excluded from every read path.
    """

    def __init__(self) -> None:
        self._strings: list[str] = []
        self._ids: dict[str, int] = {}
        self._entries: list[_Entry] = []
        # (subject_id, predicate_id) -> index into _entries (active only).
        self._active: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------------
    # Dictionary encoding
    # ------------------------------------------------------------------

    def intern(self, s: str) -> int:
        """Return the id for ``s``, storing it once on first sight."""
        existing = self._ids.get(s)
        if existing is not None:
            return existing
        idx = len(self._strings)
        self._strings.append(s)
        self._ids[s] = idx
        return idx

    def _decode(self, entry: _Entry) -> tuple[str, str, str]:
        return (
            self._strings[entry.subject_id],
            self._strings[entry.predicate_id],
            self._strings[entry.object_id],
        )

    def stats(self) -> dict:
        """Prove interning saves space.

        ``facts`` is the number of stored facts (including superseded
        audit-trail entries); ``strings`` is the number of distinct
        strings; ``dict_bytes`` is the approximate bytes of the
        stored strings (sum of UTF-8 lengths). With repeated subjects
        / predicates, ``strings`` is well under ``3 * facts`` — the
        naive per-fact string count.
        """
        return {
            "facts": len(self._entries),
            "strings": len(self._strings),
            "dict_bytes": sum(
                len(s.encode("utf-8")) for s in self._strings
            ),
        }

    # ------------------------------------------------------------------
    # Core insert with the FactStore conflict rule
    # ------------------------------------------------------------------

    def _insert(self, subject: str, predicate: str, object_: str,
                confidence: float, source: str) -> str:
        """Insert one fact; return "learned"/"blocked"/"superseded".

        Conflict is at (subject, predicate), mirroring
        FactStore.learn: same object (case-insensitive) is idempotent;
        a conflicting fact with confidence >= the newcomer's is
        blocked; a conflicting fact with lower confidence is marked
        superseded and replaced.
        """
        s_id = self._ids.get(subject)
        p_id = self._ids.get(predicate)
        if s_id is not None and p_id is not None:
            active_idx = self._active.get((s_id, p_id))
            if active_idx is not None:
                old = self._entries[active_idx]
                old_object = self._strings[old.object_id]
                if old_object.upper() == object_.upper():
                    return "learned"  # idempotent, do not duplicate
                if old.confidence >= confidence:
                    return "blocked"  # do not store, do not intern
                old.superseded = True
                o_id = self.intern(object_)
                new = _Entry(s_id, p_id, o_id, confidence, source,
                             time.time())
                self._entries.append(new)
                self._active[(s_id, p_id)] = len(self._entries) - 1
                return "superseded"
        s_id = self.intern(subject)
        p_id = self.intern(predicate)
        o_id = self.intern(object_)
        self._entries.append(
            _Entry(s_id, p_id, o_id, confidence, source, time.time()))
        self._active[(s_id, p_id)] = len(self._entries) - 1
        return "learned"

    def add_fact(self, subject: str, predicate: str, object: str,
                 confidence: float = 0.8,
                 source: str = "local") -> str:
        """Direct insert with the same conflict rule as learn().

        Used by build_from_modules for facts that already exist as
        triples (no question/answer extraction needed).
        """
        return self._insert(subject, predicate, object,
                            confidence, source)

    def learn(self, question: str, answer: str,
              score: float = 0.8, source: str = "local") -> str:
        """Learn a fact from a (question, answer) pair.

        Mirrors FactStore.learn semantics, returning a plain status
        string: "learned" | "blocked" | "superseded" | "pair".
        "pair" means no Fact extracted — the caller keeps the raw pair.
        """
        fact = _facts.extract(question, answer)
        if fact is None:
            return "pair"
        return self._insert(fact.subject, fact.predicate, fact.object,
                            score, source)

    # ------------------------------------------------------------------
    # Reads (active facts only)
    # ------------------------------------------------------------------

    def _active_entry(self, subject: str, predicate: str) -> _Entry | None:
        s_id = self._ids.get(subject)
        p_id = self._ids.get(predicate)
        if s_id is None or p_id is None:
            return None
        idx = self._active.get((s_id, p_id))
        if idx is None:
            return None
        return self._entries[idx]

    @staticmethod
    def _render(subject: str, predicate: str, object_: str) -> str:
        """Render a fact as an answer sentence.

        Closed-vocabulary predicates use the FactStore templates
        (identical output); anything else renders as
        "<subject> <predicate words> <object>." with underscores
        spaced out (e.g. "Paris located in France.").
        """
        if predicate in _facts._TEMPLATES:
            return _facts.Fact(subject, predicate, object_,
                               0.0, "").render()
        return f"{subject} {predicate.replace('_', ' ')} {object_}."

    def answer(self, question: str) -> str | None:
        """Answer a question from the graph (paraphrase-robust).

        Uses ceti.facts.extract_query to map the question to
        (subject, predicate), then renders the active fact.
        """
        try:
            q = _facts.extract_query(question)
        except Exception:
            return None
        if q is None:
            return None
        subject, predicate = q
        entry = self._active_entry(subject, predicate)
        if entry is None:
            return None
        s, p, o = self._decode(entry)
        return self._render(s, p, o)

    def lookup(self, subject: str, predicate: str) -> str | None:
        """The active fact's object for (subject, predicate), or None."""
        entry = self._active_entry(subject, predicate)
        if entry is None:
            return None
        return self._strings[entry.object_id]

    def lookup_by_entity(self, entity: str) -> list[tuple[str, str]]:
        """All active (predicate, object) pairs for a subject."""
        s_id = self._ids.get(entity)
        if s_id is None:
            return []
        out = []
        for (sid, _pid), idx in self._active.items():
            if sid != s_id:
                continue
            entry = self._entries[idx]
            out.append((self._strings[entry.predicate_id],
                        self._strings[entry.object_id]))
        out.sort()
        return out

    def _is_subject(self, entity: str) -> bool:
        """True if ``entity`` is the subject of any active fact."""
        s_id = self._ids.get(entity)
        if s_id is None:
            return False
        return any(sid == s_id for (sid, _pid) in self._active)

    def traverse(self, subject: str, relation: str | None = None,
                 depth: int = 2) -> list[tuple[str, str, str]]:
        """Follow entity chaining from ``subject`` up to ``depth`` steps.

        Each step is a (subject, predicate, object) fact whose object
        is itself a known subject (so the chain can continue). When
        ``relation`` is given, only facts with that predicate are
        followed; when None, every predicate is followed. Cycles are
        cut by the visited set. Returns steps in traversal order.
        """
        if depth <= 0:
            return []
        steps: list[tuple[str, str, str]] = []
        visited = {subject}
        frontier = [subject]
        for _ in range(depth):
            nxt: list[str] = []
            for current in frontier:
                for predicate, obj in self.lookup_by_entity(current):
                    if relation is not None and predicate != relation:
                        continue
                    steps.append((current, predicate, obj))
                    if obj not in visited and self._is_subject(obj):
                        visited.add(obj)
                        nxt.append(obj)
            frontier = nxt
            if not frontier:
                break
        return steps

    # ------------------------------------------------------------------
    # Bulk ingest from module files
    # ------------------------------------------------------------------

    def build_from_modules(self, modules) -> dict:
        """Ingest facts from module objects (files are source of truth).

        Each module has ``.facts`` (list of (subject, predicate,
        object) triples), ``.source`` and ``.confidence``. Dict-style
        modules ({"facts": [...], "source": ..., "confidence": ...})
        are accepted too. Reference modules (confidence 1.0) outrank
        learned ones via the normal conflict rule: a higher-confidence
        fact supersedes, a lower-or-equal one is blocked.

        Returns counts: {"learned": n, "blocked": n, "superseded": n}.
        """
        counts = {"learned": 0, "blocked": 0, "superseded": 0}
        for module in modules:
            if isinstance(module, dict):
                triples = module.get("facts", [])
                source = module.get("source", "local")
                confidence = module.get("confidence", 0.6)
            else:
                triples = getattr(module, "facts", [])
                source = getattr(module, "source", "local")
                confidence = getattr(module, "confidence", 0.6)
            for triple in triples:
                if hasattr(triple, "subject"):  # a Fact object
                    s, p, o = triple.subject, triple.predicate, triple.object
                else:
                    s, p, o = triple
                status = self._insert(s, p, o, confidence, source)
                counts[status] += 1
        return counts

    # ------------------------------------------------------------------
    # Full listings
    # ------------------------------------------------------------------

    def all_active(self) -> list[tuple[str, str, str, float, str]]:
        """All active facts as (subject, predicate, object, confidence,
        source), sorted by (subject, predicate)."""
        out = []
        for entry in self._entries:
            if entry.superseded:
                continue
            s, p, o = self._decode(entry)
            out.append((s, p, o, entry.confidence, entry.source))
        out.sort(key=lambda t: (t[0], t[1]))
        return out

    def all_facts(self) -> list[tuple[str, str, str, float, str, bool]]:
        """All stored facts including superseded ones, as
        (subject, predicate, object, confidence, source, superseded)
        — the audit trail."""
        out = []
        for entry in self._entries:
            s, p, o = self._decode(entry)
            out.append((s, p, o, entry.confidence, entry.source,
                        entry.superseded))
        out.sort(key=lambda t: (t[0], t[1], t[5]))
        return out

    def __len__(self) -> int:
        return sum(1 for e in self._entries if not e.superseded)
