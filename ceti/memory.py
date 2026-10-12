"""MemoryIndex — fuzzy Q&A store over promoted answers. stdlib only.

Catches "the same question asked differently" three ways:
  1. alias table — learns "BBS <-> bulletin board system" from promoted
     answers ("BBS (Bulletin Board System)", "BBS, or Bulletin Board
     System"), applied at query time in both directions
  2. typo-tolerant token similarity (Damerau-Levenshtein, so
     transpositions like "dofferent" still match "different")
  3. contraction normalization ("what's" -> "what is")

No embeddings, no numpy: TF-IDF-weighted fuzzy cosine over token sets,
persisted in sqlite3. When a real vector backend exists it can replace
this class behind the same interface:

    add(question, answer, score) -> None
    add_alias(phrase, expansion) -> None
    learn_aliases_from(answer_text) -> int
    search(query, top_k=3) -> [(question, answer, score, sim)]
"""
from __future__ import annotations

import math
import os
import re
import sqlite3
import time

_TOKEN = re.compile(r"[a-z0-9]+")

_CONTRACTIONS = {
    "what's": "what is", "that's": "that is", "it's": "it is",
    "don't": "do not", "can't": "can not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "didn't": "did not",
    "won't": "will not", "wouldn't": "would not", "couldn't": "could not",
    "shouldn't": "should not", "i'm": "i am", "you're": "you are",
    "we're": "we are", "they're": "they are",
}
_CONTRACTION_RX = re.compile(
    r"\b(" + "|".join(sorted(_CONTRACTIONS, key=len, reverse=True)) + r")\b")

# "BBS (Bulletin Board System)" / "Bulletin Board System (BBS)" /
# "BBS, or Bulletin Board System"
_ACRONYM_RX = re.compile(r"\b([A-Z]{2,}[A-Z0-9]*)\s*\(([^()]{3,80})\)")
_ACRONYM_RX2 = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\s*\(([A-Z]{2,}[A-Z0-9]*)\)")
_OR_RX = re.compile(
    r"\b([A-Z]{2,}[A-Z0-9]*)\s*,\s*or\s+"
    r"([A-Z][a-zA-Z]+(?:\s+[a-zA-Z]+){1,5})")

_FUZZY_FLOOR = 0.6  # token similarity below this earns no credit


def tokenize(text: str) -> list[str]:
    text = _CONTRACTION_RX.sub(
        lambda m: _CONTRACTIONS[m.group(1)], text.lower())
    return _TOKEN.findall(text)


def _osa(a: str, b: str) -> int:
    """Optimal string alignment distance (Levenshtein + transposition)."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1,
                          d[i][j - 1] + 1,
                          d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] \
                    and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def token_sim(a: str, b: str) -> float:
    """0..1 similarity between two tokens. 1.0 = identical."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la >= 3 and lb >= 3 and (a in b or b in a):
        return 0.85  # containment, e.g. boards/board
    return 1.0 - _osa(a, b) / max(la, lb)


class MemoryIndex:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS qa ("
            " id INTEGER PRIMARY KEY,"
            " question TEXT NOT NULL,"
            " answer TEXT NOT NULL,"
            " score REAL NOT NULL DEFAULT 0,"
            " created REAL NOT NULL)")
        self.db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS qa_q ON qa(question)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS aliases ("
            " phrase TEXT PRIMARY KEY,"
            " expansion TEXT NOT NULL)")
        self.db.commit()
        self._idf: dict[str, float] | None = None
        self._aliases: list[tuple[str, str]] | None = None

    # ---- writes -------------------------------------------------------
    def add(self, question: str, answer: str, score: float = 0.0) -> bool:
        """Insert a promoted pair. Returns False if already present."""
        try:
            self.db.execute(
                "INSERT INTO qa (question, answer, score, created)"
                " VALUES (?, ?, ?, ?)",
                (question.strip(), answer.strip(), score, time.time()))
            self.db.commit()
        except sqlite3.IntegrityError:
            return False
        self._idf = None
        return True

    def upsert(self, question: str, answer: str,
               score: float = 0.0) -> str:
        """Insert a new pair or replace an existing one.
        Returns 'added' or 'updated'. Used by dataset merging."""
        q, a = question.strip(), answer.strip()
        if not q or not a:
            raise ValueError("empty question or answer")
        cur = self.db.execute(
            "SELECT id FROM qa WHERE question=?", (q,)).fetchone()
        if cur is None:
            self.db.execute(
                "INSERT INTO qa (question, answer, score, created)"
                " VALUES (?, ?, ?, ?)", (q, a, score, time.time()))
        else:
            self.db.execute(
                "UPDATE qa SET answer=?, score=?, created=?"
                " WHERE id=?", (a, score, time.time(), cur[0]))
        self.db.commit()
        self._idf = None
        return "added" if cur is None else "updated"

    def add_alias(self, phrase: str, expansion: str) -> bool:
        """Teach one phrase -> expansion mapping (stored both ways by
        learn_aliases_from; this stores a single directed row)."""
        phrase = " ".join(phrase.lower().split())
        expansion = " ".join(expansion.lower().split())
        if not phrase or not expansion or phrase == expansion:
            return False
        try:
            self.db.execute(
                "INSERT INTO aliases (phrase, expansion) VALUES (?, ?)",
                (phrase, expansion))
            self.db.commit()
        except sqlite3.IntegrityError:
            return False
        self._aliases = None
        return True

    def learn_aliases_from(self, answer_text: str) -> int:
        """Extract acronym<->expansion pairs from a promoted answer.
        Returns the number of new aliases learned."""
        found = 0
        pairs: list[tuple[str, str]] = []
        for m in _ACRONYM_RX.finditer(answer_text):
            pairs.append((m.group(1), m.group(2)))
        for m in _ACRONYM_RX2.finditer(answer_text):
            pairs.append((m.group(2), m.group(1)))
        for m in _OR_RX.finditer(answer_text):
            pairs.append((m.group(1), m.group(2)))
        for acronym, expansion in pairs:
            # both directions, so either phrasing matches later
            if self.add_alias(acronym, expansion):
                found += 1
            if self.add_alias(expansion, acronym):
                found += 1
        return found

    def alias_count(self) -> int:
        row = self.db.execute("SELECT COUNT(*) FROM aliases").fetchone()
        return row[0] if row else 0

    def count(self) -> int:
        row = self.db.execute("SELECT COUNT(*) FROM qa").fetchone()
        return row[0] if row else 0

    # ---- reads --------------------------------------------------------
    def _get_aliases(self) -> list[tuple[str, str]]:
        if self._aliases is None:
            self._aliases = [
                (p, e) for p, e in
                self.db.execute("SELECT phrase, expansion FROM aliases")]
        return self._aliases

    def _phrase_coverage(self, phrase: str, tokens: list[str]) -> float:
        """Fraction of the phrase's tokens that (fuzzily) appear in tokens."""
        ptokens = tokenize(phrase)
        if not ptokens:
            return 0.0
        hits = 0
        for p in ptokens:
            if max((token_sim(p, t) for t in tokens), default=0.0) >= 0.6:
                hits += 1
        return hits / len(ptokens)

    def _expand_with_aliases(self, text: str) -> list[str]:
        """Tokens plus alias expansions. A phrase like "bulletin board
        system" fires on "bulletin boards" too (fuzzy coverage >= 2/3),
        so paraphrases resolve to the same terms on both sides."""
        tokens = tokenize(text)
        for phrase, expansion in self._get_aliases():
            if self._phrase_coverage(phrase, tokens) >= 0.66:
                tokens.extend(tokenize(expansion))
        return tokens

    def _sim(self, qtokens: list[str], dtokens: list[str]) -> float:
        """TF-IDF-weighted fuzzy cosine between two token lists."""
        assert self._idf is not None
        idf = self._idf
        qset, dset = set(qtokens), set(dtokens)
        if not qset or not dset:
            return 0.0
        qnorm = math.sqrt(sum(idf.get(t, 1.0) ** 2 for t in qset)) or 1.0
        dnorm = math.sqrt(sum(idf.get(t, 1.0) ** 2 for t in dset)) or 1.0
        dot = 0.0
        for t in qset:
            if t in dset:
                best = 1.0
            else:
                best = max((token_sim(t, d) for d in dset), default=0.0)
            if best >= _FUZZY_FLOOR:
                dot += idf.get(t, 1.0) ** 2 * best
        return dot / (qnorm * dnorm) if dot else 0.0

    def _build_idf(self) -> None:
        df: dict[str, int] = {}
        n = 0
        for (q,) in self.db.execute("SELECT question FROM qa"):
            n += 1
            for tok in set(tokenize(q)):
                df[tok] = df.get(tok, 0) + 1
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1.0
                     for t, c in df.items()}

    def search(self, query: str, top_k: int = 3
               ) -> list[tuple[str, str, float, float]]:
        """Return up to top_k (question, answer, stored_score, similarity).

        Similarity is a TF-IDF-weighted fuzzy cosine: each query token
        contributes its best-matching doc token (exact or typo-tolerant),
        so paraphrases and misspellings still score.
        """
        if self._idf is None:
            self._build_idf()
        q_raw = tokenize(query)
        if not q_raw:
            return []
        q_exp = self._expand_with_aliases(query)

        scored = []
        rows = list(self.db.execute(
            "SELECT question, answer, score FROM qa"))
        # expand doc tokens once per search, not per comparison
        docs = [(q, a, s, tokenize(q), self._expand_with_aliases(q))
                for q, a, s in rows]
        for q, a, s, d_raw, d_exp in docs:
            # best of plain and alias-expanded: exact matches keep
            # their full strength, paraphrases get the alias boost
            sim = max(self._sim(q_raw, d_raw), self._sim(q_exp, d_exp))
            if sim > 0.0:
                scored.append((q, a, s, sim))
        scored.sort(key=lambda r: r[3], reverse=True)
        return scored[:top_k]

    def close(self) -> None:
        self.db.close()
