"""benchmark — the brain exams itself.

For every Q&A pair in its knowledge it asks itself the question three
ways (exact, lowercased, typo'd) and checks the answer actually comes
back with the key terms. Metrics: accuracy, tier distribution,
latency. Runs are recorded so the agent can watch itself improve —
or notice itself rotting.

A miss (tier 'failed', or an answer missing the key terms) is a gap:
the agent turns repeated gaps into curiosity tasks.
"""
from __future__ import annotations

import json
import re
import time

_WORD = re.compile(r"[a-z]{5,}")  # content-ish words only


def key_terms(answer: str, k: int = 6) -> set[str]:
    words = _WORD.findall(answer.lower())
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= k:
            break
    return set(out)


def typo_variant(q: str) -> str:
    """One adjacent transposition, like a real typo."""
    chars = list(q)
    for i in range(len(chars) - 1):
        if chars[i].isalpha() and chars[i + 1].isalpha():
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            return "".join(chars)
    return q


def variants(question: str) -> list[str]:
    return [question, question.lower(), typo_variant(question.lower())]


def run(brain, max_questions: int = 25, store=None) -> dict:
    """Quiz the brain on its own knowledge. Returns the summary; if a
    TaskStore is given the run is recorded and misses become gaps."""
    rows = list(brain.memory.db.execute(
        "SELECT question, answer FROM qa"))
    if max_questions:
        rows = rows[:max_questions]
    t0 = time.time()
    results = []
    for q, a in rows:
        terms = key_terms(a)
        for v in variants(q):
            r = brain.ask(v, learn=False)
            hit = (r.tier != "failed" and
                   (not terms or len(terms & set(r.text.lower().split()))
                    >= max(1, len(terms) // 2)))
            results.append({"question": q, "variant": v, "tier": r.tier,
                            "ms": r.latency_ms, "hit": hit})
            if not hit and store is not None:
                store.note_gap(v)
    n = len(results)
    hits = sum(1 for r in results if r["hit"])
    tiers: dict[str, int] = {}
    for r in results:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    lat = sorted(r["ms"] for r in results)
    summary = {
        "questions": len(rows), "probes": n,
        "accuracy": round(hits / n, 3) if n else 0.0,
        "tier_dist": tiers,
        "latency_ms_p50": lat[n // 2] if n else 0,
        "latency_ms_max": lat[-1] if n else 0,
        "misses": [{"question": r["question"], "variant": r["variant"],
                    "tier": r["tier"]} for r in results if not r["hit"]],
        "elapsed_s": round(time.time() - t0, 1),
    }
    if store is not None:
        store.db.execute(
            "INSERT INTO bench_runs (started, finished, n, accuracy,"
            " summary) VALUES (?,?,?,?,?)",
            (t0, time.time(), n, summary["accuracy"],
             json.dumps({"tier_dist": tiers,
                         "p50": summary["latency_ms_p50"]})[:2000]))
        store.db.commit()
    return summary
