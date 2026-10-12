"""tasks — persistent task queue, gaps log, and agent state.

The agent is not just ask/answer: it has tasks (research, benchmark,
curiosity, ingest, release), each with a schedule. tick() picks the
next due task and runs it. Gaps — questions the brain failed to
answer — are logged here; the agent turns repeated gaps into
curiosity tasks. That is the self-learning loop.

All sqlite, all stdlib. One file: tasks.db in the data dir.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

_KINDS = ("research", "benchmark", "curiosity", "ingest", "release")


class TaskStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS tasks ("
            " id INTEGER PRIMARY KEY, kind TEXT NOT NULL,"
            " title TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',"
            " payload TEXT NOT NULL DEFAULT '{}',"
            " run_after REAL NOT NULL DEFAULT 0,"
            " interval_s REAL NOT NULL DEFAULT 0,"
            " attempts INTEGER NOT NULL DEFAULT 0,"
            " result TEXT, created REAL NOT NULL, updated REAL NOT NULL)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS gaps ("
            " question TEXT PRIMARY KEY, count INTEGER NOT NULL,"
            " first_seen REAL NOT NULL, last_seen REAL NOT NULL,"
            " tasked INTEGER NOT NULL DEFAULT 0)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS bench_runs ("
            " id INTEGER PRIMARY KEY, started REAL NOT NULL,"
            " finished REAL, n INTEGER, accuracy REAL, summary TEXT)")
        self.db.commit()

    # ---- tasks ---------------------------------------------------------
    def add(self, kind: str, title: str, payload: dict | None = None,
            delay_s: float = 0, interval_s: float = 0) -> int:
        if kind not in _KINDS:
            raise ValueError("unknown task kind: %s" % kind)
        now = time.time()
        cur = self.db.execute(
            "INSERT INTO tasks (kind, title, status, payload, run_after,"
            " interval_s, created, updated) VALUES (?,?,'pending',?,?,?,?,?)",
            (kind, title, json.dumps(payload or {}), now + delay_s,
             interval_s, now, now))
        self.db.commit()
        return cur.lastrowid

    def due(self, limit: int = 1) -> list[dict]:
        now = time.time()
        rows = self.db.execute(
            "SELECT id, kind, title, payload, interval_s, attempts FROM tasks"
            " WHERE status='pending' AND run_after <= ?"
            " ORDER BY run_after LIMIT ?", (now, limit))
        return [{"id": r[0], "kind": r[1], "title": r[2],
                 "payload": json.loads(r[3]), "interval_s": r[4],
                 "attempts": r[5]} for r in rows]

    def pending(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT id, kind, title, status, run_after, interval_s"
            " FROM tasks WHERE status='pending' ORDER BY run_after")
        return [{"id": r[0], "kind": r[1], "title": r[2], "status": r[3],
                 "run_after": r[4], "interval_s": r[5]} for r in rows]

    def start(self, task_id: int) -> None:
        self.db.execute(
            "UPDATE tasks SET status='running', attempts=attempts+1,"
            " updated=? WHERE id=?", (time.time(), task_id))
        self.db.commit()

    def finish(self, task_id: int, result: dict,
               interval_s: float = 0) -> None:
        now = time.time()
        if interval_s > 0:
            # recurring: back to pending, next run scheduled
            self.db.execute(
                "UPDATE tasks SET status='pending', result=?,"
                " run_after=?, updated=? WHERE id=?",
                (json.dumps(result)[:4000], now + interval_s, now,
                 task_id))
        else:
            self.db.execute(
                "UPDATE tasks SET status='done', result=?, updated=?"
                " WHERE id=?", (json.dumps(result)[:4000], now, task_id))
        self.db.commit()

    def fail(self, task_id: int, error: str) -> None:
        self.db.execute(
            "UPDATE tasks SET status='failed', result=?, updated=?"
            " WHERE id=?", (error[:1000], time.time(), task_id))
        self.db.commit()

    # ---- gaps: questions the brain couldn't answer ----------------------
    def note_gap(self, question: str) -> None:
        q = " ".join(question.split())[:300]
        if not q:
            return
        now = time.time()
        cur = self.db.execute(
            "SELECT count FROM gaps WHERE question=?", (q,)).fetchone()
        if cur is None:
            self.db.execute(
                "INSERT INTO gaps VALUES (?,?,?, ?,0)",
                (q, 1, now, now))
        else:
            self.db.execute(
                "UPDATE gaps SET count=count+1, last_seen=? WHERE question=?",
                (now, q))
        self.db.commit()

    def top_gaps(self, limit: int = 5, min_count: int = 2) -> list[dict]:
        rows = self.db.execute(
            "SELECT question, count FROM gaps"
            " WHERE tasked=0 AND count>=? ORDER BY count DESC LIMIT ?",
            (min_count, limit))
        return [{"question": r[0], "count": r[1]} for r in rows]

    def mark_tasked(self, question: str) -> None:
        self.db.execute("UPDATE gaps SET tasked=1 WHERE question=?",
                        (question,))
        self.db.commit()

    # ---- agent meta ------------------------------------------------------
    def get_meta(self, k: str, default: str | None = None) -> str | None:
        r = self.db.execute(
            "SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return r[0] if r else default

    def set_meta(self, k: str, v: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, v))
        self.db.commit()

    def close(self) -> None:
        self.db.close()
