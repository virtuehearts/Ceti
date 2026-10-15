"""Per-session local context memory.

SessionNotes is the agent's own scratch memory: timestamped notes it
writes to itself during a task run and reads back later. One file per
session, plain JSON, no server, no model involved.

The agent loop exposes these as the `record_note` / `recall_notes`
tools; the REPL can read the file directly.
"""

from __future__ import annotations

import json
import os
import time


class SessionNotes:
    """Timestamped notes persisted to a single JSON file."""

    def __init__(self, path: str):
        self.path = path
        self._notes: list[dict] | None = None

    # ---- internals -----------------------------------------------------
    def _load(self) -> list[dict]:
        if self._notes is None:
            try:
                with open(self.path, encoding="utf-8") as f:
                    data = json.load(f)
                self._notes = data if isinstance(data, list) else []
            except (OSError, ValueError):
                self._notes = []
        return self._notes

    def _save(self) -> None:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._load(), f, indent=1)
        os.replace(tmp, self.path)

    # ---- API -----------------------------------------------------------
    def record(self, content: str) -> int:
        """Append a note. Returns its index."""
        if not isinstance(content, str) or not content.strip():
            raise ValueError("note content must be a non-empty string")
        notes = self._load()
        notes.append({"t": time.time(), "content": content.strip()})
        self._save()
        return len(notes) - 1

    def recall(self, limit: int | None = None) -> list[dict]:
        """All notes, oldest first. `limit` keeps the newest N."""
        notes = self._load()
        if limit is not None:
            notes = notes[max(0, len(notes) - limit):]
        return [dict(n) for n in notes]

    def search(self, keyword: str) -> list[dict]:
        """Substring search over note contents (case-insensitive)."""
        kw = (keyword or "").lower()
        return [n for n in self.recall() if kw in n["content"].lower()]

    def count(self) -> int:
        return len(self._load())

    def format(self, limit: int | None = None) -> str:
        """Human-readable rendering for prompts and the REPL."""
        notes = self.recall(limit)
        if not notes:
            return "(no notes recorded)"
        lines = []
        for i, n in enumerate(notes):
            ts = time.strftime("%H:%M:%S", time.localtime(n["t"]))
            lines.append("[%d %s] %s" % (i, ts, n["content"]))
        return "\n".join(lines)
