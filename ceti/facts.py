"""Structured fact store — learning as triples, not flat Q&A pairs.

A Fact is (subject, predicate, object) + confidence + source +
created + superseded. The fact base is the source of truth for learned
knowledge; AIML is a compiled index over it (see docs/facts.md).

Stdlib only. SQLite for the store, in-memory for the extractor.
"""
from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Predicates: a closed vocabulary. Extraction is rule-based and
# deterministic; anything outside this set stays a raw Q&A pair.
# ---------------------------------------------------------------------------

PREDICATES = {
    "introduced", "released", "founded", "created",
    "ram", "memory", "cpu", "price", "units_sold",
    "author", "creator", "designer", "inventor",
    "is", "named_after", "successor", "predecessor",
}

# Confidence by source (see docs/facts.md).
CONFIDENCE = {
    "research": 0.9,
    "cassette": 0.9,
    "oracle": 0.8,
    "cloud": 0.8,
    "local": 0.6,
    "user": 0.9,
}

# How to render a fact as an answer sentence.
_TEMPLATES = {
    "introduced": "The {s} was introduced in {o}.",
    "released": "The {s} was released in {o}.",
    "founded": "The {s} was founded in {o}.",
    "created": "The {s} was created in {o}.",
    "ram": "The {s} has {o} of RAM.",
    "memory": "The {s} has {o} of memory.",
    "cpu": "The {s} uses the {o}.",
    "price": "The price of the {s} was {o}.",
    "units_sold": "The {s} sold {o} units.",
    "author": "The {s} was written by {o}.",
    "creator": "The {s} was created by {o}.",
    "designer": "The {s} was designed by {o}.",
    "inventor": "The {s} was invented by {o}.",
    "is": "The {s} is {o}.",
    "named_after": "The {s} was named after {o}.",
    "successor": "The {s} was succeeded by {o}.",
    "predecessor": "The {s} was preceded by {o}.",
}

# Paraphrase patterns per predicate, with {S} = subject.
# These compile to AIML patterns (normalized: upper, no punctuation).
_PATTERNS = {
    "introduced": [
        "WHEN WAS {S} INTRODUCED",
        "WHEN DID {S} COME OUT",
        "WHAT YEAR WAS {S} INTRODUCED",
        "WHEN WAS {S} LAUNCHED",
        "WHEN WAS {S} RELEASED",
    ],
    "released": [
        "WHEN WAS {S} RELEASED",
        "WHEN DID {S} COME OUT",
        "WHAT YEAR WAS {S} RELEASED",
    ],
    "founded": [
        "WHEN WAS {S} FOUNDED",
        "WHAT YEAR WAS {S} FOUNDED",
        "WHEN DID {S} START",
    ],
    "created": [
        "WHEN WAS {S} CREATED",
        "WHO CREATED {S}",
        "WHEN DID {S} COME OUT",
    ],
    "ram": [
        "HOW MUCH RAM DOES {S} HAVE",
        "WHAT IS RAM OF {S}",
        "HOW MUCH MEMORY DOES {S} HAVE",
        "WHAT IS MEMORY OF {S}",
    ],
    "memory": [
        "HOW MUCH MEMORY DOES {S} HAVE",
        "WHAT IS MEMORY OF {S}",
        "HOW MUCH RAM DOES {S} HAVE",
    ],
    "cpu": [
        "WHAT IS CPU OF {S}",
        "WHAT PROCESSOR DOES {S} USE",
        "WHAT CPU DOES {S} HAVE",
    ],
    "price": [
        "WHAT WAS PRICE OF {S}",
        "HOW MUCH DID {S} COST",
        "WHAT IS PRICE OF {S}",
    ],
    "units_sold": [
        "HOW MANY UNITS OF {S} WERE SOLD",
        "HOW MANY {S} WERE SOLD",
        "WHAT WAS TOTAL NUMBER OF {S} UNITS SOLD",
    ],
    "author": [
        "WHO WROTE {S}",
        "WHO IS AUTHOR OF {S}",
        "WHO WROTE THE BOOK {S}",
    ],
    "creator": [
        "WHO CREATED {S}",
        "WHO IS CREATOR OF {S}",
    ],
    "designer": [
        "WHO DESIGNED {S}",
        "WHO IS DESIGNER OF {S}",
    ],
    "inventor": [
        "WHO INVENTED {S}",
        "WHO IS INVENTOR OF {S}",
    ],
    "is": [
        "WHAT IS {S}",
        "WHAT IS A {S}",
        "WHAT IS THE {S}",
    ],
    "named_after": [
        "WHAT WAS {S} NAMED AFTER",
        "WHY IS {S} CALLED {S}",
    ],
    "successor": [
        "WHAT SUCCEEDED {S}",
        "WHAT WAS SUCCESSOR OF {S}",
    ],
    "predecessor": [
        "WHAT PRECEDED {S}",
        "WHAT WAS PREDECESSOR OF {S}",
    ],
}


@dataclass
class Fact:
    subject: str
    predicate: str
    object: str
    confidence: float
    source: str
    created: float = 0.0
    id: int | None = None
    superseded: bool = False

    def render(self) -> str:
        tpl = _TEMPLATES.get(self.predicate)
        if tpl is None:
            return f"{self.subject} {self.predicate} {self.object}."
        return tpl.format(s=self.subject, o=self.object)

    def patterns(self) -> list[str]:
        """AIML patterns (normalized) that should answer this fact."""
        out = []
        for pat in _PATTERNS.get(self.predicate, []):
            out.append(pat.replace("{S}", self.subject.upper()))
        return out


# ---------------------------------------------------------------------------
# Extraction: (question, answer) -> Fact | None. Rule-based, deterministic.
#
# The QUESTION determines the subject + predicate (its shape is the
# query). The ANSWER provides the object (the evidence). Pulling the
# subject from the answer is unreliable — answers start with
# "Approximately 2.5 million units were sold..." or "The book '...' was
# written by ...", neither of which has a clean subject span.
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text.strip().upper())


def _clean(text: str) -> str:
    """Strip articles / trailing punctuation, title-case a subject."""
    s = _WS.sub(" ", text.strip()).strip(" ?.,;:")
    s = re.sub(r"^(THE|A|AN)\s+", "", s, flags=re.I)
    return s


_ACRONYMS = {"VIC", "CPU", "RAM", "BBS", "SID", "PET", "DOS", "IBM",
             "C64", "C128", "CBM", "MOS", "Z80", "TCP", "IP", "DNS",
             "ARPANET", "NSFNET", "FIDO", "FIDONET", "WWIV", "PCBOARD",
             "GEOS", "BASIC", "HTML", "HTTP", "URL", "AT", "BASIC"}


def _title_word(w: str) -> str:
    """Title-case one word, keeping acronyms and punctuation."""
    core = w.strip("()[]{}\"'.,;:")
    if core.upper() in _ACRONYMS:
        return w.upper() if w == core else w.replace(core, core.upper())
    # capitalize the first letter, lowercase the rest
    i = 0
    while i < len(w) and not w[i].isalpha():
        i += 1
    if i >= len(w):
        return w
    return w[:i] + w[i].upper() + w[i + 1:].lower()


def _title(s: str) -> str:
    """Title-case, but keep known acronyms uppercase (VIC, CPU, BBS)."""
    return " ".join(_title_word(w) for w in s.split(" "))


def _subject_from_question(qn: str, predicate: str) -> str:
    """The subject span for a known predicate, from the question shape."""
    # "WHAT YEAR DID X COME OUT" / "WHEN WAS X INTRODUCED" / ...
    m = re.match(
        r"^(?:WHEN\s+WAS|WHEN\s+DID|WHAT\s+YEAR\s+DID|WHAT\s+YEAR\s+WAS)\s+"
        r"(?:THE\s+)?(.+?)\s+"
        r"(?:INTRODUCED|LAUNCHED|RELEASED|FOUNDED|CREATED|COME\s+OUT)\b",
        qn)
    if m:
        return _clean(m.group(1))
    # "HOW MUCH RAM DOES X HAVE" / "HOW MUCH MEMORY DOES X HAVE"
    m = re.match(r"^HOW\s+MUCH\s+(?:RAM|MEMORY)\s+DOES\s+(?:THE\s+)?(.+?)\s+HAVE\b", qn)
    if m:
        return _clean(m.group(1))
    # "WHAT IS THE RAM/MEMORY/CPU/PRICE OF X" / "WHAT IS X'S RAM"
    m = re.match(r"^WHAT\s+IS\s+(?:THE\s+)?"
                 r"(?:RAM|MEMORY|CPU|PROCESSOR|PRICE|COST)\s+OF\s+(?:THE\s+)?(.+)$", qn)
    if m:
        return _clean(m.group(1))
    # "WHAT CPU/PROCESSOR DOES X USE/HAVE"
    m = re.match(r"^WHAT\s+(?:CPU|PROCESSOR)\s+DOES\s+(?:THE\s+)?(.+?)\s+(?:USE|HAVE)\b", qn)
    if m:
        return _clean(m.group(1))
    # "HOW MUCH DID X COST"
    m = re.match(r"^HOW\s+MUCH\s+DID\s+(?:THE\s+)?(.+?)\s+COST\b", qn)
    if m:
        return _clean(m.group(1))
    # "HOW MANY UNITS OF X WERE SOLD" / "HOW MANY X WERE SOLD"
    m = re.match(r"^HOW\s+MANY\s+(?:UNITS?\s+OF\s+)?(?:THE\s+)?(.+?)\s+(?:WERE\s+)?SOLD\b", qn)
    if m:
        return _clean(m.group(1))
    # "WHO WROTE X" / "WHO CREATED/DESIGNED/INVENTED X"
    m = re.match(r"^WHO\s+(?:WROTE|CREATED|DESIGNED|INVENTED)\s+(?:THE\s+)?(.+)$", qn)
    if m:
        return _clean(m.group(1))
    # "WHO IS THE AUTHOR/CREATOR/DESIGNER/INVENTOR OF X"
    m = re.match(r"^WHO\s+IS\s+(?:THE\s+)?(?:AUTHOR|CREATOR|DESIGNER|INVENTOR)\s+OF\s+(?:THE\s+)?(.+)$", qn)
    if m:
        return _clean(m.group(1))
    # "WHAT WAS X NAMED AFTER" / "WHAT WAS THE NAME OF X" / "WHY IS X CALLED ..."
    m = re.match(r"^WHAT\s+WAS\s+(?:THE\s+)?(.+?)\s+NAMED\s+(?:AFTER|FOR)\b", qn)
    if m:
        return _clean(m.group(1))
    m = re.match(r"^WHAT\s+WAS\s+(?:THE\s+)?NAME\s+OF\s+(?:THE\s+)?(.+)$", qn)
    if m:
        return _clean(m.group(1))
    # "WHAT SUCCEEDED/PRECEDED X" / "WHAT WAS THE SUCCESSOR/PREDECESSOR OF X"
    m = re.match(r"^WHAT\s+(?:SUCCEEDED|PRECEDED)\s+(?:THE\s+)?(.+)$", qn)
    if m:
        return _clean(m.group(1))
    m = re.match(r"^WHAT\s+WAS\s+(?:THE\s+)?(?:SUCCESSOR|PREDECESSOR)\s+OF\s+(?:THE\s+)?(.+)$", qn)
    if m:
        return _clean(m.group(1))
    # "WHAT IS X" / "WHAT IS A X"
    m = re.match(r"^WHAT\s+IS\s+(?:THE\s+|A\s+|AN\s+)?(.+)$", qn)
    if m and predicate == "is":
        return _clean(m.group(1))
    return ""


def _predicate_of_question(qn: str) -> str | None:
    """The predicate a question is asking for, from its shape.

    A bare "when was X" has no verb — the answer's verb decides
    (released / introduced / founded / created). extract() handles
    that case by re-checking the answer.
    """
    if re.search(r"\b(INTRODUCED|LAUNCHED|RELEASED|COME\s+OUT)\b", qn) and \
            re.search(r"\b(WHEN|WHAT\s+YEAR)\b", qn):
        return "introduced"
    if re.search(r"\bFOUNDED\b", qn):
        return "founded"
    if re.search(r"\bCREATED\b", qn):
        return "created"
    if re.search(r"\b(RAM|MEMORY)\b", qn):
        return "ram"
    if re.search(r"\b(CPU|PROCESSOR)\b", qn):
        return "cpu"
    if re.search(r"\b(PRICE|COST)\b", qn):
        return "price"
    if re.search(r"\b(SOLD|UNITS)\b", qn):
        return "units_sold"
    if re.search(r"\b(WROTE|WRITTEN|AUTHOR)\b", qn):
        return "author"
    if re.search(r"\b(DESIGNED|DESIGNER)\b", qn):
        return "designer"
    if re.search(r"\b(INVENTED|INVENTOR)\b", qn):
        return "inventor"
    if re.search(r"\b(NAMED|NAME)\b", qn):
        return "named_after"
    if re.search(r"\b(SUCCEEDED|SUCCESSOR)\b", qn):
        return "successor"
    if re.search(r"\b(PRECEDED|PREDECESSOR)\b", qn):
        return "predecessor"
    if re.match(r"^WHAT\s+IS\b", qn):
        return "is"
    return None


def _object_from_answer(predicate: str, answer: str) -> str | None:
    """The object (the evidence) for a predicate, from the answer."""
    an = answer.strip()
    au = an.upper()

    if predicate in ("introduced", "released", "founded", "created"):
        m = re.search(
            r"\b(?:INTRODUCED|LAUNCHED|RELEASED|FOUNDED|CREATED)\s+(?:IN|ON)\s+"
            r"([A-Z][A-Z]+ \d{4}|\d{4}|[A-Z][A-Z]+ \d{1,2},? \d{4})", au)
        if m:
            return m.group(1).title()
        return None

    if predicate == "ram":
        m = re.search(r"\b(\d+\s*(?:KB|MB|GB))\s+(?:OF\s+)?(?:RAM|MEMORY)\b", au)
        if m:
            return m.group(1).upper()
        return None

    if predicate == "cpu":
        # "<name> CPU" — the name usually precedes CPU ("MOS 6502 CPU")
        m = re.search(r"\b([A-Z][A-Z0-9 .\-/]{2,40}?)\s+(?:CPU|PROCESSOR)\b", au)
        if m:
            obj = _WS.sub(" ", m.group(1).strip()).strip()
            # drop a leading article / verb ("featured a", "uses the")
            # greedy: drop everything up to the last verb ("PET FEATURED A ...")
            obj = re.sub(r"^.*\b(?:FEATURED|USES|USE|USING|WITH)\s+", "", obj, flags=re.I)
            obj = re.sub(r"^(?:FEATURED|USES|USE|USING|WITH|AND)\s+", "", obj, flags=re.I)
            obj = re.sub(r"^(?:THE|A|AN)\s+", "", obj, flags=re.I)
            if obj and len(obj) <= 40:
                return _title(obj)
        # "CPU is/uses <name>"
        m = re.search(r"\b(?:CPU|PROCESSOR)\b\s*(?:IS|WAS|USES|USED)?\s*"
                      r"(?:THE\s+)?([A-Z0-9][A-Z0-9 \-\./]{2,40}?)(?:\.|,|;|$)", au)
        if m:
            obj = _WS.sub(" ", m.group(1).strip()).strip()
            if obj and len(obj) <= 40:
                return _title(obj)
        return None

    if predicate == "price":
        m = re.search(r"US\$([\d,]+\.\d{2}|[\d,]+)", au)
        if m:
            return f"US${m.group(1)}"
        m = re.search(r"\$([\d,]+\.\d{2}|[\d,]+)", au)
        if m:
            return f"US${m.group(1)}"
        return None

    if predicate == "units_sold":
        m = re.search(r"\b([\d.]+\s*MILLION|[\d,]+)\s+UNITS?\b", au)
        if m:
            return m.group(1).title()
        m = re.search(r"\b([\d,]+)\s+(?:UNITS?\s+)?SOLD\b", au)
        if m:
            return m.group(1)
        return None

    if predicate in ("author", "creator", "designer", "inventor"):
        verb = {"author": r"WRITTEN", "creator": r"CREATED",
                "designer": r"DESIGNED", "inventor": r"INVENTED"}[predicate]
        # "written by Philip K. Dick." — greedy name, strip final period
        m = re.search(
            verb + r"\s+BY\s+([A-Z][A-Za-z']*(?:\s+[A-Z][A-Za-z'.]*)*)", au)
        if m:
            obj = _WS.sub(" ", m.group(1).strip()).strip(".,;")
            if obj and len(obj) <= 60:
                return _title(obj)
        return None

    if predicate == "named_after":
        m = re.search(r"\bNAMED\s+(?:AFTER|FOR)\s+(?:THE\s+)?"
                      r"([A-Z][A-Za-z .()]+?)(?:,|;|\.\s|$)", au)
        if m:
            obj = _WS.sub(" ", m.group(1).strip()).strip(".,;")
            if obj and len(obj) <= 60:
                return _title(obj)
        return None

    if predicate == "is":
        # "A BBS is a bulletin board system." — strip a leading article
        m = re.match(r"^(?:A\s+|AN\s+|THE\s+)?(.+?)\s+IS\s+(.+)$", an, flags=re.I)
        if m:
            obj = _WS.sub(" ", m.group(2).strip()).strip(".")
            if obj and len(obj) <= 200:
                return obj
        return None

    return None


def extract(question: str, answer: str) -> Fact | None:
    """Extract a Fact from a (question, answer) pair, or None."""
    qn = _norm(question)
    if not qn or not answer.strip():
        return None
    pred = _predicate_of_question(qn)
    if pred is None:
        # Bare "when was X" — the answer's verb decides the predicate.
        m = re.match(r"^WHEN\s+WAS\s+(?:THE\s+)?(.+)$", qn)
        if m:
            subj = _clean(m.group(1))
            au = answer.upper()
            for verb, p in (("RELEASED", "released"), ("INTRODUCED", "introduced"),
                            ("FOUNDED", "founded"), ("CREATED", "created")):
                if re.search(r"\b" + verb + r"\b", au):
                    obj = _object_from_answer(p, answer)
                    if obj:
                        return Fact(_title(subj), p, obj, 0.0, "", time.time())
        return None
    subj = _subject_from_question(qn, pred)
    if not subj or len(subj) > 80:
        return None
    obj = _object_from_answer(pred, answer)
    if not obj:
        return None
    return Fact(_title(subj), pred, obj, 0.0, "", time.time())


def extract_query(question: str) -> tuple[str, str] | None:
    """Map a question to (subject, predicate) for fact lookup."""
    qn = _norm(question)
    pred = _predicate_of_question(qn)
    if pred is None:
        # Bare "when was X" — the answer's verb decides the predicate.
        m = re.match(r"^WHEN\s+WAS\s+(?:THE\s+)?(.+)$", qn)
        if m:
            subj = _clean(m.group(1))
            au = answer.upper()
            for verb, p in (("RELEASED", "released"), ("INTRODUCED", "introduced"),
                            ("FOUNDED", "founded"), ("CREATED", "created")):
                if re.search(r"\b" + verb + r"\b", au):
                    obj = _object_from_answer(p, answer)
                    if obj:
                        return Fact(_title(subj), p, obj, 0.0, "", time.time())
        return None
    subj = _subject_from_question(qn, pred)
    if not subj:
        return None
    return (_title(subj), pred)


# ---------------------------------------------------------------------------
# Store: SQLite. One row per fact; superseded facts are kept (audit trail).
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL,
    created REAL NOT NULL,
    superseded INTEGER NOT NULL DEFAULT 0,
    superseded_by INTEGER
);
CREATE INDEX IF NOT EXISTS idx_facts_sp
    ON facts (subject, predicate, superseded);
"""


class FactStore:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def active(self, subject: str, predicate: str) -> Fact | None:
        """The current (non-superseded) fact for (subject, predicate)."""
        row = self.conn.execute(
            "SELECT id, subject, predicate, object, confidence, source,"
            " created, superseded FROM facts"
            " WHERE subject = ? AND predicate = ? AND superseded = 0"
            " ORDER BY confidence DESC, created DESC LIMIT 1",
            (subject, predicate)).fetchone()
        if row is None:
            return None
        return Fact(row[1], row[2], row[3], row[4], row[5], row[6],
                    id=row[0], superseded=bool(row[7]))

    def learn(self, question: str, answer: str,
              confidence: float | None = None,
              source: str = "local") -> dict:
        """Learn a fact from a (question, answer) pair.

        Returns a dict: {"status": "learned"|"blocked"|"superseded"|"pair",
        "fact": Fact|None, "existing": Fact|None}.
        "pair" means no fact extracted — the caller keeps the raw pair.
        """
        fact = extract(question, answer)
        if fact is None:
            return {"status": "pair", "fact": None, "existing": None}
        if confidence is None:
            confidence = CONFIDENCE.get(source, 0.6)
        fact.confidence = confidence
        fact.source = source
        existing = self.active(fact.subject, fact.predicate)
        if existing is not None:
            if existing.object.upper() == fact.object.upper():
                return {"status": "learned", "fact": existing,
                        "existing": existing}
            if existing.confidence >= fact.confidence:
                return {"status": "blocked", "fact": None,
                        "existing": existing}
            # New fact wins: supersede the old one.
            self.conn.execute(
                "UPDATE facts SET superseded = 1, superseded_by = ?"
                " WHERE id = ?", (-1, existing.id))
            self.conn.commit()
            cur = self.conn.execute(
                "INSERT INTO facts (subject, predicate, object,"
                " confidence, source, created, superseded)"
                " VALUES (?, ?, ?, ?, ?, ?, 0)",
                (fact.subject, fact.predicate, fact.object,
                 fact.confidence, fact.source, fact.created))
            fact.id = cur.lastrowid
            self.conn.execute(
                "UPDATE facts SET superseded_by = ? WHERE id = ?",
                (fact.id, existing.id))
            self.conn.commit()
            return {"status": "superseded", "fact": fact,
                    "existing": existing}
        cur = self.conn.execute(
            "INSERT INTO facts (subject, predicate, object,"
            " confidence, source, created, superseded)"
            " VALUES (?, ?, ?, ?, ?, ?, 0)",
            (fact.subject, fact.predicate, fact.object,
             fact.confidence, fact.source, fact.created))
        fact.id = cur.lastrowid
        self.conn.commit()
        return {"status": "learned", "fact": fact, "existing": None}

    def answer(self, question: str) -> str | None:
        """Answer a question from the fact base (paraphrase-robust)."""
        q = extract_query(question)
        if q is None:
            return None
        fact = self.active(q[0], q[1])
        if fact is None:
            return None
        return fact.render()

    def all_active(self) -> list[Fact]:
        rows = self.conn.execute(
            "SELECT id, subject, predicate, object, confidence, source,"
            " created, superseded FROM facts"
            " WHERE superseded = 0 ORDER BY subject, predicate").fetchall()
        return [Fact(r[1], r[2], r[3], r[4], r[5], r[6], id=r[0],
                     superseded=bool(r[7])) for r in rows]

    def compile_aiml(self) -> list[tuple[str, str]]:
        """(pattern, template) pairs for every active fact."""
        out = []
        for fact in self.all_active():
            for pat in fact.patterns():
                out.append((pat, fact.render()))
        return out
