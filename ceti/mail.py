"""mail — signed direct messages between nodes.

A message is a signed envelope addressed to one fingerprint.
The sender signs with their node key; the recipient verifies
before reading. Messages travel as files, like everything else
in the Gopher layer: send writes to the outbox, the file is
handed over (shared drop, torrent, USB), the recipient runs
import and it lands in the inbox. No live transport, no server.

Sending to a fingerprint you have not filed as a peer is
refused — no mail to strangers you have not met.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

from . import signing
from .peers import derive_peer


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_mail_bytes(msg: dict) -> bytes:
    unsigned = {k: v for k, v in msg.items() if k != "signature"}
    return json.dumps(unsigned, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def mail_id(msg: dict) -> str:
    return hashlib.sha256(canonical_mail_bytes(msg)).hexdigest()


def verify_mail(msg: dict) -> tuple[bool, str]:
    sender = msg.get("from") or {}
    sig = msg.get("signature") or {}
    try:
        derived = derive_peer(sender.get("public_hex", ""))
    except ValueError as e:
        return False, str(e)
    if derived["fingerprint"] != sender.get("fingerprint") or \
            derived["node_id"] != sender.get("node_id"):
        return False, "sender identity does not match its public key"
    if not msg.get("to", {}).get("fingerprint"):
        return False, "no recipient"
    if not sig.get("signature"):
        return False, "no signature"
    if signing.verify_bytes(canonical_mail_bytes(msg),
                            sender["public_hex"],
                            sig["signature"]):
        return True, ""
    return False, "signature invalid"


class MailBox:
    """data/mail/outbox/ + data/mail/inbox/ — one file per message."""

    def __init__(self, data_dir: str):
        base = os.path.join(os.path.abspath(data_dir), "mail")
        self.outbox = os.path.join(base, "outbox")
        self.inbox_dir = os.path.join(base, "inbox")
        os.makedirs(self.outbox, exist_ok=True)
        os.makedirs(self.inbox_dir, exist_ok=True)

    def send(self, identity, to_fingerprint: str, subject: str,
             body: str, peer_book) -> dict:
        """Sign and file a message in the outbox. The recipient
        must be a filed peer — no mail to strangers."""
        if peer_book.get(to_fingerprint) is None:
            raise ValueError("unknown peer %s — /peer add first"
                             % to_fingerprint)
        msg = {
            "from": {
                "node_id": identity.node_id,
                "fingerprint": identity.fingerprint,
                "public_hex": identity.public_hex(),
            },
            "to": {"fingerprint": to_fingerprint},
            "created": _utcnow(),
            "subject": subject.strip(),
            "body": body.strip(),
        }
        msg["signature"] = {
            "signature": identity.sign(canonical_mail_bytes(msg)),
            "scheme": signing.SCHEME,
        }
        mid = mail_id(msg)
        with open(os.path.join(self.outbox, mid + ".json"),
                  "w", encoding="utf-8") as f:
            json.dump(msg, f, indent=2, sort_keys=True)
            f.write("\n")
        return msg

    def inbox(self, own_fingerprint: str) -> tuple[list[dict], list[dict]]:
        """(valid messages to me, rejected). Verified on load;
        messages addressed to someone else are skipped silently —
        they are not mine to read."""
        valid, rejected = [], []
        for fn in sorted(os.listdir(self.inbox_dir)):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.inbox_dir, fn),
                          encoding="utf-8") as f:
                    msg = json.load(f)
            except (OSError, ValueError):
                rejected.append({"file": fn, "reason": "unreadable"})
                continue
            if msg.get("to", {}).get("fingerprint") != own_fingerprint:
                continue
            ok, reason = verify_mail(msg)
            if ok:
                valid.append(msg)
            else:
                rejected.append({"file": fn, "reason": reason})
        valid.sort(key=lambda m: (m.get("created", ""), mail_id(m)))
        return valid, rejected

    def import_mail(self, path: str) -> dict:
        """Verify a message file and file it in the inbox.
        Refuses loudly on any verification failure."""
        with open(path, encoding="utf-8") as f:
            msg = json.load(f)
        ok, reason = verify_mail(msg)
        if not ok:
            raise ValueError("refusing mail %s: %s" % (path, reason))
        mid = mail_id(msg)
        dest = os.path.join(self.inbox_dir, mid + ".json")
        if not os.path.exists(dest):
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(msg, f, indent=2, sort_keys=True)
                f.write("\n")
        return msg
