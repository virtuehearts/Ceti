"""merge — fold a verified peer dataset into the local store.

Git-the-idea, not git-the-binary:

  - content-addressed records: id = sha256(normalized question)
  - history: the manifest chain (prev_hash) is already our commit log
  - authentication: verify_tarball() before anything is trusted
  - merge: semantic set reconciliation below (the part git can't do)
  - commit: the caller's next signed release, chained as usual

Why not the git binary:
  - sqlite doesn't merge — git can only flag binary conflicts.
  - git's line-based 3-way merge is the wrong granularity for Q&A:
    "A BBS is X" vs "A BBS is X, from the 1980s" are different lines,
    so git keeps both and the knowledge base fills with duplicates.
  - a git binary dependency breaks the stdlib-only, runs-anywhere
    story of this brain.

The merge has to be semantic anyway: same normalized question means
the same record. Conflict policy 'score': higher relevance score
wins, ties keep the local answer, and every conflict is logged.
(Peer scores are peer-asserted — a future web-of-trust can weight
them by signer; for now the signature is the trust anchor.)

Embeddings are always recomputed locally on ingest. Peer vectors are
never trusted: they're cheap to rebuild and blindly accepting them
would allow vector poisoning.

Usage:
    report = ingest_tarball(store, memory, aiml, "peer-v3.tar.gz")
    # then cut a new signed release so the seeded chain continues:
    store.release("ceti-knowledge", files, note="merged <peer>")
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from .aiml import normalize_input, normalize_pattern
from .datasets import DatasetStore, unpack_tarball

_POLICIES = ("score",)


def record_id(question: str) -> str:
    """Content address of a Q&A record: sha256 of the normalized question."""
    return hashlib.sha256(normalize_input(question).encode()).hexdigest()


def _peer_table(bundle: str, table: str, cols: str) -> list[tuple]:
    db_path = os.path.join(bundle, "knowledge.sqlite3")
    if not os.path.exists(db_path):
        return []
    con = sqlite3.connect(db_path)
    try:
        return list(con.execute(f"SELECT {cols} FROM {table}"))
    except sqlite3.Error:
        return []
    finally:
        con.close()


def _peer_categories(bundle: str) -> list[tuple[str, str, str]]:
    """(normalized_pattern, raw_pattern, template) from peer AIML files."""
    cats: list[tuple[str, str, str]] = []
    for root, _dirs, files in os.walk(bundle):
        for f in sorted(files):
            if not f.endswith(".aiml"):
                continue
            try:
                r = ET.parse(os.path.join(root, f)).getroot()
            except ET.ParseError:
                continue  # corrupt peer file: skip, never crash
            for cat in r.findall("category"):
                pat_el = cat.find("pattern")
                tpl_el = cat.find("template")
                if pat_el is None or tpl_el is None:
                    continue
                raw = (pat_el.text or "").strip()
                norm = normalize_pattern(raw)
                tpl = re.sub(r"\s+", " ",
                             "".join(tpl_el.itertext())).strip()
                if norm and tpl:
                    cats.append((norm, raw, tpl))
    return cats


def _local_qa(memory) -> dict[str, tuple[str, str, float]]:
    """record_id -> (question, answer, score) for the local store."""
    out: dict[str, tuple[str, str, float]] = {}
    for q, a, s in memory.db.execute(
            "SELECT question, answer, score FROM qa"):
        out.setdefault(record_id(q), (q, a, s))
    return out


def merge_bundle(memory, aiml, bundle: str,
                 policy: str = "score") -> dict:
    """Reconcile a peer's extracted bundle into local memory + AIML.
    The bundle MUST already be verified. Returns a report dict."""
    if policy not in _POLICIES:
        raise ValueError("unknown merge policy: %s" % policy)
    report: dict = {"qa_added": 0, "qa_updated": 0, "qa_skipped": 0,
                    "conflicts": [],
                    "aliases_added": 0, "aliases_skipped": 0,
                    "aiml_added": 0, "aiml_skipped": 0,
                    "aiml_conflicts": []}

    # ---- Q&A records ----
    local = _local_qa(memory)
    for pq, pa, ps in _peer_table(bundle, "qa",
                                  "question, answer, score"):
        rid = record_id(pq)
        if rid not in local:
            memory.upsert(pq, pa, ps)
            local[rid] = (pq, pa, ps)
            report["qa_added"] += 1
        elif local[rid][1] == pa:
            report["qa_skipped"] += 1
        elif ps > local[rid][2]:
            memory.upsert(pq, pa, ps)
            report["qa_updated"] += 1
            report["conflicts"].append(
                {"question": pq, "winner": "peer",
                 "peer_score": ps, "local_score": local[rid][2]})
            local[rid] = (pq, pa, ps)
        else:
            report["qa_skipped"] += 1
            report["conflicts"].append(
                {"question": pq, "winner": "local",
                 "peer_score": ps, "local_score": local[rid][2]})

    # ---- aliases (phrase identity; keep local on conflict) ----
    local_alias = {r[0] for r in
                   memory.db.execute("SELECT phrase FROM aliases")}
    for phrase, expansion in _peer_table(bundle, "aliases",
                                         "phrase, expansion"):
        if phrase in local_alias:
            report["aliases_skipped"] += 1
            continue
        if memory.add_alias(phrase, expansion):
            report["aliases_added"] += 1
        else:
            report["aliases_skipped"] += 1
        local_alias.add(phrase)

    # ---- AIML categories -> learned file (it outranks cassettes) ----
    for norm, raw, tpl in _peer_categories(bundle):
        existing = aiml.learned_template(norm)
        if existing is None:
            if aiml.learn(raw, tpl):
                report["aiml_added"] += 1
            else:
                report["aiml_skipped"] += 1
        elif existing == tpl:
            report["aiml_skipped"] += 1
        else:
            report["aiml_skipped"] += 1
            report["aiml_conflicts"].append(
                {"pattern": norm, "winner": "local"})

    return report


def ingest_tarball(store: DatasetStore, memory, aiml,
                   tarball_path: str, policy: str = "score") -> dict:
    """Verify a peer release tarball, then merge it. Refuses loudly
    if verification fails. Appends a merge record to the ledger.
    Returns the merge report."""
    ok, reasons = store.verify_tarball(tarball_path)
    if not ok:
        raise ValueError("refusing to merge %s: %s"
                         % (tarball_path, "; ".join(reasons)))

    tmp = tempfile.mkdtemp(prefix="dataset-ingest-")
    try:
        bundle = unpack_tarball(tarball_path, tmp)
        with open(os.path.join(bundle, "manifest.json")) as f:
            manifest = json.load(f)
        sig = manifest.get("signature", {})
        report = merge_bundle(memory, aiml, bundle, policy)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    entry = {"type": "merge",
             "peer_node_id": sig.get("node_id"),
             "peer_fingerprint": sig.get("fingerprint"),
             "peer_dataset": manifest.get("dataset"),
             "peer_version": manifest.get("version"),
             "peer_merkle_root": manifest.get("merkle_root"),
             "policy": policy,
             "qa_added": report["qa_added"],
             "qa_updated": report["qa_updated"],
             "qa_skipped": report["qa_skipped"],
             "conflicts": len(report["conflicts"]),
             "aliases_added": report["aliases_added"],
             "aiml_added": report["aiml_added"],
             "created_utc": datetime.now(timezone.utc).isoformat()}
    with open(store.ledger, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    report["peer"] = sig.get("fingerprint")
    return report
