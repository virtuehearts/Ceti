"""Library ingest: drop files in data/library/, they become vector memory.

Supported: .pdf (via pdftotext), .txt, .md. Files are chunked
(paragraph-aware, ~600 tokens with overlap), embedded with
embeddinggemma-2, and stored in data/library.sqlite3 — a separate
table from the Q&A knowledge base, same cosine machinery.

    library/
      philip-k-dick.pdf
      notes.md
      ...

Run Library.scan() (or /ingest in the REPL) to pick up new files.
Ingested files are tracked by sha256 — re-drops are skipped.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import subprocess
import time
import urllib.request

from .vectormem import DIM, EmbedError, cosine, pack, unpack

LIBRARY_DIRNAME = "library"
DB_FILENAME = "library.sqlite3"
EMBED_URL = "http://127.0.0.1:11435"

# chunking: ~600 tokens at ~4 chars/token, ~120 tokens overlap
CHUNK_CHARS = 2400
OVERLAP_CHARS = 480
EMBED_BATCH = 16


class IngestError(Exception):
    """A file could not be ingested."""


def library_dir(data_dir: str) -> str:
    d = os.path.join(os.path.abspath(data_dir), LIBRARY_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return d


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def extract_text(path: str) -> str:
    """Plain text out of pdf/txt/md. Raises IngestError."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            p = subprocess.run(
                ["pdftotext", "-layout", path, "-"],
                capture_output=True, timeout=300)
        except FileNotFoundError:
            raise IngestError("pdftotext not installed")
        except subprocess.TimeoutExpired:
            raise IngestError("pdftotext timed out on %s" % path)
        if p.returncode != 0:
            raise IngestError("pdftotext failed: %s"
                              % p.stderr.decode()[:200])
        text = p.stdout.decode("utf-8", "replace")
    elif ext in (".txt", ".md", ".markdown"):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError as e:
            raise IngestError("cannot read %s: %s" % (path, e))
    else:
        raise IngestError("unsupported type: %s (want .pdf/.txt/.md)"
                          % ext)
    # normalize: collapse 3+ newlines, strip trailing space per line
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # drop likely headers/footers: very short repeated lines are noise;
    # keep it simple — drop page-number-only lines
    text = re.sub(r"\n\d{1,4}\n", "\n", text)
    if len(text.strip()) < 200:
        raise IngestError("too little text extracted from %s" % path)
    return text.strip()


def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS,
               overlap_chars: int = OVERLAP_CHARS) -> list[str]:
    """Greedy paragraph packing with character overlap between chunks."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in paras:
        # a single giant paragraph gets hard-split
        while len(p) > chunk_chars:
            if cur_len:
                chunks.append("\n\n".join(cur))
                cur, cur_len = [], 0
            chunks.append(p[:chunk_chars])
            p = p[chunk_chars - overlap_chars:]
        if cur_len + len(p) > chunk_chars and cur:
            chunks.append("\n\n".join(cur))
            # overlap: carry the tail forward
            tail = "\n\n".join(cur)[-overlap_chars:]
            cur, cur_len = [tail], len(tail)
        cur.append(p)
        cur_len += len(p) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return [c for c in chunks if len(c.strip()) >= 100]


class Library:
    """Document vector memory over data/library.sqlite3."""

    def __init__(self, data_dir: str, embed_url: str = EMBED_URL):
        self.data_dir = os.path.abspath(data_dir)
        self.libdir = library_dir(data_dir)
        self.embed_url = embed_url.rstrip("/")
        self.db = sqlite3.connect(
            os.path.join(self.data_dir, DB_FILENAME))
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS docs ("
            " doc_id INTEGER PRIMARY KEY,"
            " filename TEXT, sha256 TEXT UNIQUE, chunks INTEGER,"
            " chars INTEGER, added REAL)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS doc_chunks ("
            " id INTEGER PRIMARY KEY,"
            " doc_id INTEGER, chunk_no INTEGER, text TEXT,"
            " embedding BLOB)")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc"
            " ON doc_chunks(doc_id)")
        self.db.commit()

    def close(self):
        try:
            self.db.close()
        except sqlite3.Error:
            pass

    # ---- embedding ----------------------------------------------------
    def _embed(self, texts: list[str]) -> list[list[float]]:
        req = urllib.request.Request(
            self.embed_url + "/v1/embeddings",
            data=json.dumps({"input": texts}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.load(r)
        except Exception as e:
            raise EmbedError("embedding server unreachable: %s" % e)
        items = sorted(data["data"], key=lambda d: d["index"])
        vecs = [list(map(float, d["embedding"])) for d in items]
        for v in vecs:
            if len(v) != DIM:
                raise EmbedError("expected %d dims, got %d"
                                 % (DIM, len(v)))
        return vecs

    def ping(self) -> bool:
        try:
            self._embed(["ping"])
            return True
        except EmbedError:
            return False

    # ---- ingest --------------------------------------------------------
    def ingest_file(self, path: str) -> dict:
        """Extract, chunk, embed, store. Returns stats."""
        path = os.path.abspath(path)
        digest = sha256_file(path)
        row = self.db.execute(
            "SELECT doc_id FROM docs WHERE sha256=?", (digest,)).fetchone()
        if row:
            return {"filename": os.path.basename(path),
                    "status": "already-ingested"}
        text = extract_text(path)
        chunks = chunk_text(text)
        if not chunks:
            raise IngestError("no chunks from %s" % path)
        cur = self.db.execute(
            "INSERT INTO docs (filename, sha256, chunks, chars, added)"
            " VALUES (?, ?, ?, ?, ?)",
            (os.path.basename(path), digest, len(chunks),
             len(text), time.time()))
        doc_id = cur.lastrowid
        embedded = 0
        for i in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[i:i + EMBED_BATCH]
            vecs = self._embed(batch)
            self.db.executemany(
                "INSERT INTO doc_chunks (doc_id, chunk_no, text,"
                " embedding) VALUES (?, ?, ?, ?)",
                [(doc_id, i + j, batch[j], pack(vecs[j]))
                 for j in range(len(batch))])
            embedded += len(batch)
        self.db.commit()
        return {"filename": os.path.basename(path), "status": "ingested",
                "chunks": len(chunks), "chars": len(text)}

    def scan(self) -> list[dict]:
        """Ingest every new supported file in data/library/."""
        results = []
        for name in sorted(os.listdir(self.libdir)):
            path = os.path.join(self.libdir, name)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(name)[1].lower() not in (
                    ".pdf", ".txt", ".md", ".markdown"):
                continue
            try:
                results.append(self.ingest_file(path))
            except (IngestError, EmbedError) as e:
                results.append({"filename": name, "status": "error",
                                "error": str(e)[:200]})
        return results

    def docs(self) -> list[dict]:
        return [{"filename": r[0], "chunks": r[1], "chars": r[2],
                 "added": r[3]}
                for r in self.db.execute(
                    "SELECT filename, chunks, chars, added FROM docs"
                    " ORDER BY added").fetchall()]

    def drop(self, filename: str) -> bool:
        """Remove a document and its chunks."""
        row = self.db.execute(
            "SELECT doc_id FROM docs WHERE filename=?",
            (filename,)).fetchone()
        if not row:
            return False
        self.db.execute("DELETE FROM doc_chunks WHERE doc_id=?",
                        (row[0],))
        self.db.execute("DELETE FROM docs WHERE doc_id=?", (row[0],))
        self.db.commit()
        return True

    # ---- search ---------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Cosine search over document chunks. Returns text + score."""
        query = query.strip()
        if not query:
            return []
        qv = self._embed([query])[0]
        scored = []
        for cid, doc_id, chunk_no, text, blob, filename in self.db.execute(
                "SELECT c.id, c.doc_id, c.chunk_no, c.text, c.embedding,"
                " d.filename FROM doc_chunks c"
                " JOIN docs d ON d.doc_id = c.doc_id"):
            try:
                sim = cosine(qv, unpack(blob))
            except (struct.error, ValueError):
                continue
            scored.append({"text": text, "score": sim,
                           "filename": filename, "chunk_no": chunk_no})
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]

    def count_chunks(self) -> int:
        return self.db.execute(
            "SELECT COUNT(*) FROM doc_chunks").fetchone()[0]
