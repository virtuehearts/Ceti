"""nodelist — signed lists of peers: discovery for the Gopher net.

A nodelist is how nodes find each other: a signed JSON file
listing known nodes (fingerprint, public key, address). It ships
with releases, lives at a well-known URL (e.g. darknet.ca/nodes),
and moves between peers as gossip — every exchange is the same
format, the same verification.

Trust rules: the whole list is refused if ANY entry fails —
a bad signature, a signer whose fingerprint does not match its
key, or a node entry whose fingerprint does not match its key.
Addresses are claimed, never verified; a key is self-verifying
by derivation, an address is just a hint.
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone

from . import signing
from .peers import derive_peer

VERSION = 1


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_nodelist_bytes(doc: dict) -> bytes:
    unsigned = {k: v for k, v in doc.items() if k != "signature"}
    return json.dumps(unsigned, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def build_nodelist(peer_records: list[dict], identity) -> dict:
    """Build a signed nodelist from peer records. Private fields
    (notes) are stripped — only public routing info ships."""
    nodes = []
    for p in peer_records:
        entry = {
            "node_id": p["node_id"],
            "fingerprint": p["fingerprint"],
            "public_hex": p["public_hex"],
        }
        if p.get("address"):
            entry["address"] = p["address"]
        nodes.append(entry)
    nodes.sort(key=lambda n: n["fingerprint"])
    doc = {
        "version": VERSION,
        "created": _utcnow(),
        "nodes": nodes,
        "signer": {
            "node_id": identity.node_id,
            "fingerprint": identity.fingerprint,
            "public_hex": identity.public_hex(),
        },
    }
    doc["signature"] = {
        "signature": identity.sign(canonical_nodelist_bytes(doc)),
        "scheme": signing.SCHEME,
    }
    return doc


def verify_nodelist(doc: dict) -> tuple[bool, str]:
    """Verify a nodelist: signature, signer identity, and every
    node entry's key derivation. Any failure refuses the list."""
    if doc.get("version") != VERSION:
        return False, "unsupported nodelist version: %r" % doc.get("version")
    signer = doc.get("signer") or {}
    sig = doc.get("signature") or {}
    try:
        derived = derive_peer(signer.get("public_hex", ""))
    except ValueError as e:
        return False, "bad signer key: %s" % e
    if derived["fingerprint"] != signer.get("fingerprint") or \
            derived["node_id"] != signer.get("node_id"):
        return False, "signer identity does not match its public key"
    if not sig.get("signature"):
        return False, "no signature"
    if not signing.verify_bytes(canonical_nodelist_bytes(doc),
                                signer["public_hex"],
                                sig["signature"]):
        return False, "signature invalid"
    for entry in doc.get("nodes", []):
        try:
            d = derive_peer(entry.get("public_hex", ""))
        except ValueError:
            return False, "bad node key in entry: %s" % entry.get("fingerprint")
        if d["fingerprint"] != entry.get("fingerprint") or \
                d["node_id"] != entry.get("node_id"):
            return False, ("node entry does not match its key: %s"
                           % entry.get("fingerprint"))
    return True, ""


def merge_nodelist(book, doc: dict) -> dict:
    """Verify a nodelist, then file every node in it. Returns
    {added, known, total}. Refuses loudly on any verification
    failure — nothing merges."""
    ok, reason = verify_nodelist(doc)
    if not ok:
        raise ValueError("refusing nodelist: %s" % reason)
    added = known = 0
    for entry in doc.get("nodes", []):
        before = book.get(entry["fingerprint"])
        book.add(entry["public_hex"],
                 address=entry.get("address", ""))
        if before is None:
            added += 1
        else:
            known += 1
    return {"added": added, "known": known,
            "total": len(doc.get("nodes", []))}


def fetch_nodelist(source: str) -> dict:
    """Fetch a nodelist from a URL (http/https) or a file path.
    Returns the parsed document — verification is the caller's job
    (merge_nodelist verifies before merging)."""
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    with open(source, encoding="utf-8") as f:
        return json.load(f)


def save_nodelist(doc: dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")
    return path
