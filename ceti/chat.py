"""chat — a room where nodes talk. Fully interactive, terminal only.

A room is a directory of signed messages: data/chat/<room>/.
Every message is signed with the sender's node key and verified
on read. Two terminals sharing a room directory — same box, or
a shared folder — chat live: the loop polls for new messages
and prints them as they arrive. No server, no account, just
signed files in a shared dir.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import select
import sys
from datetime import datetime, timezone

from . import signing
from .peers import derive_peer


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_room(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "lobby"


def canonical_chat_bytes(msg: dict) -> bytes:
    unsigned = {k: v for k, v in msg.items() if k != "signature"}
    return json.dumps(unsigned, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def chat_id(msg: dict) -> str:
    return hashlib.sha256(canonical_chat_bytes(msg)).hexdigest()


def verify_chat(msg: dict) -> tuple[bool, str]:
    sender = msg.get("from") or {}
    sig = msg.get("signature") or {}
    try:
        derived = derive_peer(sender.get("public_hex", ""))
    except ValueError as e:
        return False, str(e)
    if derived["fingerprint"] != sender.get("fingerprint") or \
            derived["node_id"] != sender.get("node_id"):
        return False, "sender identity does not match its public key"
    if not sig.get("signature"):
        return False, "no signature"
    if signing.verify_bytes(canonical_chat_bytes(msg),
                            sender["public_hex"],
                            sig["signature"]):
        return True, ""
    return False, "signature invalid"


class ChatRoom:
    """data/chat/<room>/ — one JSON file per signed message."""

    def __init__(self, data_dir: str, name: str):
        self.name = _safe_room(name)
        self.dir = os.path.join(os.path.abspath(data_dir), "chat",
                                self.name)
        os.makedirs(self.dir, exist_ok=True)

    def post(self, identity, body: str) -> dict:
        msg = {
            "from": {
                "node_id": identity.node_id,
                "fingerprint": identity.fingerprint,
                "public_hex": identity.public_hex(),
            },
            "room": self.name,
            "created": _utcnow(),
            "body": body.strip(),
        }
        msg["signature"] = {
            "signature": identity.sign(canonical_chat_bytes(msg)),
            "scheme": signing.SCHEME,
        }
        mid = chat_id(msg)
        with open(os.path.join(self.dir, mid + ".json"),
                  "w", encoding="utf-8") as f:
            json.dump(msg, f, indent=2, sort_keys=True)
            f.write("\n")
        return msg

    def _load_all(self) -> tuple[list[dict], int]:
        valid, rejected = [], 0
        for fn in sorted(os.listdir(self.dir)):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.dir, fn),
                          encoding="utf-8") as f:
                    msg = json.load(f)
            except (OSError, ValueError):
                rejected += 1
                continue
            if msg.get("room") != self.name:
                continue
            ok, _ = verify_chat(msg)
            if ok:
                valid.append(msg)
            else:
                rejected += 1
        valid.sort(key=lambda m: (m.get("created", ""), chat_id(m)))
        return valid, rejected

    def history(self, limit: int = 20) -> list[dict]:
        valid, _ = self._load_all()
        return valid[-limit:]

    def poll(self, known_ids: set) -> tuple[list[dict], set, int]:
        """Messages not in known_ids. Returns (new, all_ids, rejected)."""
        valid, rejected = self._load_all()
        all_ids = {chat_id(m) for m in valid}
        return [m for m in valid if chat_id(m) not in known_ids], \
            all_ids, rejected


def format_msg(msg: dict) -> str:
    return "[%s] %s: %s" % (msg["created"][11:19],
                            msg["from"]["fingerprint"], msg["body"])


def run_chat(brain, room_name: str) -> None:
    """Interactive chat loop: poll every 2s, post what you type.
    /quit (or EOF / Ctrl-C) leaves the room."""
    if brain.identity is None:
        print("no identity: ecdsa not installed")
        return
    room = ChatRoom(brain.cfg.data_dir, room_name)
    print(f"--- room #{room.name} (type /quit to leave) ---")
    seen: set = set()
    for m in room.history():
        print(format_msg(m))
        seen.add(chat_id(m))
    try:
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 2.0)
            if not ready:
                new, seen, _ = room.poll(seen)
                for m in new:
                    print(format_msg(m))
                continue
            line = sys.stdin.readline()
            if line == "" or line.strip() == "/quit":
                break
            line = line.strip()
            if not line:
                continue
            msg = room.post(brain.identity, line)
            seen.add(chat_id(msg))
    except KeyboardInterrupt:
        pass
    print(f"--- left #{room.name} ---")
