"""board — signed posts in the BBS tradition.

One JSON file per post in data/board/posts/. Every post is signed
with the author's node identity; a post whose signature does not
verify is never displayed — it is counted as rejected and left
where it is, loudly. No passwords on the wire: the signature IS
the authorship proof.

Posts travel as files: hand a post file to another node, it runs
/import, the signature is checked, and the post is filed. That is
the whole federation primitive for v1 — no live transport, no
central server, just signed files moving between nodes.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

from . import signing


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_post_bytes(post: dict) -> bytes:
    """Deterministic bytes for signing: everything except the
    signature block, JSON-sorted, no whitespace. Same pattern as
    datasets.canonical_bytes."""
    unsigned = {k: v for k, v in post.items() if k != "signature"}
    return json.dumps(unsigned, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def post_id(post: dict) -> str:
    """Content address: sha256 of the canonical post bytes."""
    return hashlib.sha256(canonical_post_bytes(post)).hexdigest()


def verify_post(post: dict) -> tuple[bool, str]:
    """Check a post's signature against its author key. Returns
    (ok, reason). Never trusts claimed identifiers: the fingerprint
    must match the public key."""
    from .peers import derive_peer
    sig = post.get("signature") or {}
    author = post.get("author") or {}
    try:
        derived = derive_peer(author.get("public_hex", ""))
    except ValueError as e:
        return False, str(e)
    if derived["fingerprint"] != author.get("fingerprint") or \
            derived["node_id"] != author.get("node_id"):
        return False, "author identity does not match its public key"
    if not sig.get("signature"):
        return False, "no signature"
    if signing.verify_bytes(canonical_post_bytes(post),
                            author["public_hex"],
                            sig["signature"]):
        return True, ""
    return False, "signature invalid"


class Board:
    """data/board/posts/<id>.json — one file per signed post."""

    def __init__(self, data_dir: str):
        self.dir = os.path.join(os.path.abspath(data_dir), "board",
                                "posts")
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, pid: str) -> str:
        return os.path.join(self.dir, pid + ".json")

    def post(self, identity, subject: str, body: str) -> dict:
        """Sign and file a post as this node."""
        post = {
            "author": {
                "node_id": identity.node_id,
                "fingerprint": identity.fingerprint,
                "public_hex": identity.public_hex(),
            },
            "created": _utcnow(),
            "subject": subject.strip(),
            "body": body.strip(),
        }
        post["signature"] = {
            "signature": identity.sign(canonical_post_bytes(post)),
            "scheme": signing.SCHEME,
        }
        pid = post_id(post)
        with open(self._path(pid), "w", encoding="utf-8") as f:
            json.dump(post, f, indent=2, sort_keys=True)
            f.write("\n")
        return post

    def list(self) -> tuple[list[dict], list[dict]]:
        """(valid posts, rejected posts). Valid posts are sorted
        oldest first. Rejected ones are reported, never displayed."""
        valid, rejected = [], []
        for fn in sorted(os.listdir(self.dir)):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.dir, fn),
                          encoding="utf-8") as f:
                    post = json.load(f)
            except (OSError, ValueError):
                rejected.append({"file": fn, "reason": "unreadable"})
                continue
            ok, reason = verify_post(post)
            if ok:
                valid.append(post)
            else:
                rejected.append({"file": fn, "reason": reason})
        valid.sort(key=lambda p: (p.get("created", ""),
                                  post_id(p)))
        return valid, rejected

    def read(self, n: int) -> dict | None:
        """The nth valid post (1-based, oldest first)."""
        valid, _ = self.list()
        if 1 <= n <= len(valid):
            return valid[n - 1]
        return None

    def import_post(self, path: str) -> dict:
        """Verify a post file from another node and file it.
        Refuses loudly on any verification failure."""
        with open(path, encoding="utf-8") as f:
            post = json.load(f)
        ok, reason = verify_post(post)
        if not ok:
            raise ValueError("refusing post %s: %s" % (path, reason))
        pid = post_id(post)
        dest = self._path(pid)
        if not os.path.exists(dest):
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(post, f, indent=2, sort_keys=True)
                f.write("\n")
        return post
