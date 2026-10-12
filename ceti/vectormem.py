"""VectorMemory — embedding-backed Q&A store. Same interface as MemoryIndex.

Embeddings: google/embeddinggemma-2 (768d) via the local llama-server
(OpenAI-compatible /v1/embeddings), with the model card's asymmetric
task prefixes:
  documents: "title: qa | text: {question} {answer}"
  queries:   "task: search result | query: {query}"

Storage is plain sqlite3 (no sqlite-vec, no numpy): embeddings live as
float32 BLOBs and cosine similarity runs in pure Python — fast enough
for thousands of pairs on low-spec hardware. The database
(knowledge.sqlite3) also holds the alias table and the research log,
so one file is the whole knowledge base.

Interface (mirrors MemoryIndex):
    add(question, answer, score) -> bool
    search(query, top_k=3) -> [(question, answer, score, sim)]
    learn_aliases_from(answer_text) -> int   (recorded, not needed for search)
    alias_count() / count() / close()
    ping() -> bool                            (is the embed server up?)
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import struct
import time
import urllib.request

from .memory import _ACRONYM_RX, _ACRONYM_RX2, _OR_RX, tokenize

DIM = 768
DEFAULT_EMBED_URL = "http://127.0.0.1:11435"


def doc_text(question: str, answer: str) -> str:
    return "title: qa | text: %s %s" % (question.strip(), answer.strip())


def query_text(query: str) -> str:
    return "task: search result | query: %s" % query.strip()


def pack(vec: list[float]) -> bytes:
    return struct.pack("<%df" % len(vec), *vec)


def unpack(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack("<%df" % n, blob))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class EmbedError(Exception):
    pass


class VectorMemory:
    def __init__(self, path: str,
                 embed_url: str = DEFAULT_EMBED_URL,
                 embed_fn=None):
        """embed_fn(texts) -> [vec] overrides HTTP (used by tests)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.embed_url = embed_url.rstrip("/")
        self._embed_fn = embed_fn
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS qa ("
            " id INTEGER PRIMARY KEY,"
            " question TEXT NOT NULL,"
            " answer TEXT NOT NULL,"
            " score REAL NOT NULL DEFAULT 0,"
            " created REAL NOT NULL,"
            " embedding BLOB NOT NULL)")
        self.db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS qa_q ON qa(question)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS aliases ("
            " phrase TEXT PRIMARY KEY,"
            " expansion TEXT NOT NULL)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS research_log ("
            " topic TEXT PRIMARY KEY,"
            " started REAL NOT NULL,"
            " finished REAL,"
            " qa_added INTEGER NOT NULL DEFAULT 0,"
            " report_path TEXT)")
        self.db.commit()

    # ---- embedding ----------------------------------------------------
    def _embed_http(self, texts: list[str]) -> list[list[float]]:
        req = urllib.request.Request(
            self.embed_url + "/v1/embeddings",
            data=json.dumps({"input": texts}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
        except Exception as e:
            raise EmbedError("embedding server unreachable: %s" % e)
        items = sorted(data["data"], key=lambda d: d["index"])
        vecs = [list(map(float, d["embedding"])) for d in items]
        for v in vecs:
            if len(v) != DIM:
                raise EmbedError(
                    "expected %d dims, got %d" % (DIM, len(v)))
        return vecs

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embed_fn is not None:
            return self._embed_fn(texts)
        return self._embed_http(texts)

    def ping(self) -> bool:
        try:
            self._embed(["ping"])
            return True
        except EmbedError:
            return False

    # ---- writes -------------------------------------------------------
    def add(self, question: str, answer: str, score: float = 0.0) -> bool:
        q = " ".join(question.split())
        a = " ".join(answer.split())
        if not q or not a:
            return False
        vec = self._embed([doc_text(q, a)])[0]
        try:
            self.db.execute(
                "INSERT INTO qa (question, answer, score, created,"
                " embedding) VALUES (?, ?, ?, ?, ?)",
                (q, a, score, time.time(), pack(vec)))
            self.db.commit()
        except sqlite3.IntegrityError:
            return False
        return True

    def upsert(self, question: str, answer: str,
               score: float = 0.0) -> str:
        """Insert a new pair or replace an existing one (re-embeds).
        Returns 'added' or 'updated'. Used by dataset merging."""
        q = " ".join(question.split())
        a = " ".join(answer.split())
        if not q or not a:
            raise ValueError("empty question or answer")
        vec = self._embed([doc_text(q, a)])[0]
        cur = self.db.execute(
            "SELECT id FROM qa WHERE question=?", (q,)).fetchone()
        if cur is None:
            self.db.execute(
                "INSERT INTO qa (question, answer, score, created,"
                " embedding) VALUES (?, ?, ?, ?, ?)",
                (q, a, score, time.time(), pack(vec)))
            self.db.commit()
            return "added"
        self.db.execute(
            "UPDATE qa SET answer=?, score=?, created=?, embedding=?"
            " WHERE id=?", (a, score, time.time(), pack(vec), cur[0]))
        self.db.commit()
        return "updated"

    def add_alias(self, phrase: str, expansion: str) -> bool:
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
        return True

    def learn_aliases_from(self, answer_text: str) -> int:
        """Record acronym<->expansion pairs (embeddings handle the
        matching natively; this keeps the alias ledger complete)."""
        found = 0
        pairs: list[tuple[str, str]] = []
        for m in _ACRONYM_RX.finditer(answer_text):
            pairs.append((m.group(1), m.group(2)))
        for m in _ACRONYM_RX2.finditer(answer_text):
            pairs.append((m.group(2), m.group(1)))
        for m in _OR_RX.finditer(answer_text):
            pairs.append((m.group(1), m.group(2)))
        for acronym, expansion in pairs:
            if self.add_alias(acronym, expansion):
                found += 1
            if self.add_alias(expansion, acronym):
                found += 1
        return found

    def log_research_start(self, topic: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO research_log"
            " (topic, started, finished, qa_added, report_path)"
            " VALUES (?, ?, NULL, 0, NULL)",
            (topic, time.time()))
        self.db.commit()

    def log_research_done(self, topic: str, qa_added: int,
                          report_path: str | None) -> None:
        self.db.execute(
            "UPDATE research_log SET finished=?, qa_added=?,"
            " report_path=? WHERE topic=?",
            (time.time(), qa_added, report_path, topic))
        self.db.commit()

    def researched_topics(self) -> set[str]:
        return {r[0] for r in self.db.execute(
            "SELECT topic FROM research_log WHERE finished IS NOT NULL")}

    def alias_count(self) -> int:
        row = self.db.execute("SELECT COUNT(*) FROM aliases").fetchone()
        return row[0] if row else 0

    def count(self) -> int:
        row = self.db.execute("SELECT COUNT(*) FROM qa").fetchone()
        return row[0] if row else 0

    # ---- reads --------------------------------------------------------
    def search(self, query: str, top_k: int = 3
               ) -> list[tuple[str, str, float, float]]:
        if not query.strip():
            return []
        qvec = self._embed([query_text(query)])[0]
        scored = []
        for q, a, s, blob in self.db.execute(
                "SELECT question, answer, score, embedding FROM qa"):
            sim = cosine(qvec, unpack(blob))
            scored.append((q, a, s, sim))
        scored.sort(key=lambda r: r[3], reverse=True)
        return scored[:top_k]

    def close(self) -> None:
        self.db.close()
