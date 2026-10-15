"""tui_sessions — saved chat sessions for the TUI.

One JSON file per session under data/tui-sessions/, named
<id>.json. A session is a transcript (the same lines the TUI
shows) plus a title derived from the first question, Hermes
style. Sessions with no user line are never written — an
empty chat leaves no file behind.

The store is deliberately separate from the agent loop's
data/sessions/ (tool-driving runs), which use a different
format and purpose.
"""
from __future__ import annotations

import json
import os
import secrets
import time

TITLE_LEN = 40


class Session:
    def __init__(self, session_id: str, created: float,
                 lines: list[str] | None = None) -> None:
        self.id = session_id
        self.created = created
        self.lines: list[str] = list(lines or [])

    @staticmethod
    def _is_question(ln: str) -> bool:
        """A real question, not a slash command: command-only
        chats are not conversations and never get saved."""
        return ln.startswith("you> ") and not ln.startswith("you> /")

    @property
    def title(self) -> str:
        for ln in self.lines:
            if self._is_question(ln):
                return ln[5:].strip()[:TITLE_LEN] or "new chat"
        return "new chat"

    @property
    def has_user_lines(self) -> bool:
        return any(self._is_question(ln) for ln in self.lines)

    def to_dict(self) -> dict:
        return {"id": self.id, "created": self.created,
                "title": self.title, "lines": self.lines}

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        return cls(d["id"], d["created"], d.get("lines", []))


def session_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "tui-sessions")


def new_session_id() -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return f"{stamp}-{secrets.token_hex(2)}"


def new_session() -> Session:
    return Session(new_session_id(), time.time())


def _path(data_dir: str, session_id: str) -> str:
    return os.path.join(session_dir(data_dir), session_id + ".json")


def save_session(data_dir: str, session: Session) -> bool:
    """Write the session. Skips sessions with no user line that
    have no file yet — an empty chat leaves nothing behind.
    Returns True when a file exists afterwards."""
    path = _path(data_dir, session.id)
    if not session.has_user_lines and not os.path.exists(path):
        return False
    os.makedirs(session_dir(data_dir), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(session.to_dict(), f, ensure_ascii=False)
    os.replace(tmp, path)
    return True


def load_session(data_dir: str, session_id: str) -> Session | None:
    try:
        with open(_path(data_dir, session_id),
                  encoding="utf-8") as f:
            return Session.from_dict(json.load(f))
    except (OSError, ValueError, KeyError):
        return None


def resolve_id(data_dir: str, query: str) -> str | None:
    """Exact id, or a unique prefix of one. None when the query
    matches nothing or is ambiguous."""
    if load_session(data_dir, query) is not None:
        return query
    matches = [s["id"] for s in list_sessions(data_dir)
               if s["id"].startswith(query)]
    if len(matches) == 1:
        return matches[0]
    return None


def list_sessions(data_dir: str) -> list[dict]:
    """Session summaries, newest first: id, created, title,
    n_lines. Corrupt files are skipped, never fatal."""
    out = []
    d = session_dir(data_dir)
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, name), encoding="utf-8") as f:
                rec = json.load(f)
            out.append({"id": rec["id"], "created": rec["created"],
                        "title": rec.get("title", "new chat"),
                        "n_lines": len(rec.get("lines", []))})
        except (OSError, ValueError, KeyError):
            continue
    out.sort(key=lambda r: r["created"], reverse=True)
    return out


def delete_session(data_dir: str, session_id: str) -> bool:
    try:
        os.remove(_path(data_dir, session_id))
        return True
    except OSError:
        return False
