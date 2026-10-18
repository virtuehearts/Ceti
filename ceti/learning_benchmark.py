"""Held-out learning benchmark — SWE-bench-Lite-STYLE methodology
applied to knowledge learning (NOT the literal SWE-bench code suite).

The point is leakage control. A benchmark that the brain could already
answer — because the fact sits in a seed cassette, or because the
"learning" set just repeats the benchmark question — proves nothing.
So this harness enforces three separations:

  1. HELD-OUT vs SEED: no benchmark question or answer may appear in
     any *.aiml file under the seed dirs (the shipped cassettes plus
     the brain's seed file). Any hit is leakage: the baseline could
     already know it.
  2. HELD-OUT vs LEARNING: no benchmark QUESTION may appear in the
     learning set, verbatim or as a >80% content-word near-duplicate.
     The learning set teaches the same facts with DIFFERENT phrasing,
     so the after-run measures generalization, not memorization of
     the exact question.
  3. TEACHING is not leakage: a learning ANSWER necessarily contains
     the taught answer (that is what teaching means). What is checked
     instead is that no benchmark answer appears in a learning
     QUESTION, and no benchmark question is smuggled into a learning
     answer.

The before/after protocol: a fresh Brain in a temp dir with a fake
local backend (returns "n/a") and no cloud scores ~0 on the benchmark
(it has never seen the facts). The learning set is then fed through
Brain._learn, and the benchmark runs again. A real delta means the
brain generalized from paraphrased teaching to the held-out wording.

Stdlib only. No network, no cloud/oracle keys, no real Ollama. The
test brain runs with handler_mode=False: with a fake backend, handler
mode would rewrite retrieved knowledge with fake output, and this
benchmark measures learning/retrieval, not handler presentation.
(The fact tier bypasses the handler anyway; the memory tier returns
the raw learned answer with handler_mode=False.)
"""
from __future__ import annotations

import json
import os
import re
import tempfile

from .backends import ChatBackend
from .brain import Brain
from .config import BrainConfig

__all__ = [
    "run_benchmark",
    "leakage_check",
    "run_before_after",
    "judge",
    "content_words",
]

# Question words, auxiliaries, articles and glue: not content. Note
# "have", "there", "many", "first", "most" are deliberately NOT here —
# they discriminate benchmark items (first President, most abundant
# gas, how many legs) and the fixture's overlap budget counts them.
_STOPWORDS = {
    "what", "who", "when", "where", "which", "whose", "whom", "how",
    "why", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "can", "could", "should", "would", "will",
    "has", "had", "the", "a", "an", "of", "to", "in", "on", "for",
    "with", "by", "at", "from", "as", "and", "or", "but", "it", "its",
    "this", "that", "these", "those", "he", "she", "they", "we", "you",
    "his", "her", "their", "our", "your", "my", "than", "then", "not",
    "all", "any", "some", "one", "two", "also", "into", "over", "under",
}

_TOKEN = re.compile(r"[a-z0-9]+")
_WS = re.compile(r"\s+")

# Above this content-word overlap, a learning question is a
# near-duplicate of the benchmark question (memorization, not
# generalization).
OVERLAP_LIMIT = 0.8

_MINIMAL_SEED = """<?xml version="1.0"?><aiml version="1.0">\
<category><pattern>HELLO</pattern>\
<template>Hi.</template></category></aiml>
"""


def normalize(text: str) -> str:
    """Lowercase, punctuation -> space, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return _WS.sub(" ", text).strip()


def content_words(text: str) -> set[str]:
    """Content words: tokens len>=4 that are not stopwords."""
    return {t for t in _TOKEN.findall(text.lower())
            if len(t) >= 4 and t not in _STOPWORDS}


def _overlap(a: set[str], b: set[str]) -> float:
    """Shared content words / max size. 0.0 when either side is empty
    (a content-less side, e.g. the answer "H2O", cannot measurably
    overlap — equality is caught by the substring checks)."""
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), len(b))


def judge(answer: str, response: str) -> bool:
    """Grounded match: EVERY content word (len>=4) of the ground-truth
    answer must appear as a whole word in the response. Answers with
    no content words (e.g. "H2O", "366") fall back to a whole-word
    phrase match of the full normalized answer."""
    need = content_words(answer)
    if need:
        have = set(_TOKEN.findall(response.lower()))
        return need <= have
    norm_ans = normalize(answer)
    if not norm_ans:
        return False
    return re.search(r"\b" + re.escape(norm_ans) + r"\b",
                     normalize(response)) is not None


def run_benchmark(brain, questions) -> dict:
    """Score a brain on held-out questions. Each question is asked
    with learn=False (the benchmark must not teach itself). Returns
    accuracy + per-question rows."""
    rows = []
    correct = 0
    for item in questions:
        q = item["question"]
        ans = brain.ask(q, learn=False)
        ok = judge(item["answer"], ans.text)
        correct += 1 if ok else 0
        rows.append({
            "question": q,
            "answer": item["answer"],
            "domain": item.get("domain", ""),
            "response": ans.text,
            "tier": ans.tier,
            "correct": ok,
        })
    total = len(questions)
    return {"accuracy": (correct / total) if total else 0.0,
            "correct": correct, "total": total, "rows": rows}


def leakage_check(benchmark, learning, seed_dirs) -> dict:
    """Return {"leaks": [...]}. Empty list = clean.

    Checks, per benchmark item:
      * benchmark QUESTION vs every learning QUESTION: verbatim /
        substring / >80% content-word overlap = leak.
      * benchmark QUESTION vs every learning ANSWER: >80% overlap =
        leak (the question smuggled into the teaching text).
      * benchmark ANSWER vs every learning QUESTION: substring /
        >80% overlap = leak (the answer sitting in a question).
      * benchmark QUESTION / ANSWER vs every *.aiml file under
        seed_dirs (recursive): question as substring, answer as a
        whole-word phrase = leak (the baseline could already know it).

    Deliberately NOT checked: benchmark answer inside a learning
    ANSWER. That is teaching, not leakage — the learning set has to
    contain the fact it teaches.
    """
    leaks: list[str] = []
    for b in benchmark:
        bq, ba = b["question"], b["answer"]
        nbq, nba = normalize(bq), normalize(ba)
        cw_bq, cw_ba = content_words(bq), content_words(ba)
        for pair in learning:
            lq, la = pair["question"], pair["answer"]
            nlq, nla = normalize(lq), normalize(la)
            if nbq and (nbq == nlq or nbq in nlq or nlq in nbq):
                leaks.append(
                    f"benchmark question {bq!r} appears verbatim in "
                    f"learning question {lq!r}")
            ov = _overlap(cw_bq, content_words(lq))
            if ov > OVERLAP_LIMIT:
                leaks.append(
                    f"benchmark question {bq!r} shares {ov:.0%} content "
                    f"words with learning question {lq!r}")
            ov = _overlap(cw_bq, content_words(la))
            if ov > OVERLAP_LIMIT:
                leaks.append(
                    f"benchmark question {bq!r} shares {ov:.0%} content "
                    f"words with learning answer {la!r}")
            if nba and len(nba) >= 3 and re.search(
                    r"\b" + re.escape(nba) + r"\b", nlq):
                leaks.append(
                    f"benchmark answer {ba!r} appears in learning "
                    f"question {lq!r}")
            ov = _overlap(cw_ba, content_words(lq))
            if ov > OVERLAP_LIMIT:
                leaks.append(
                    f"benchmark answer {ba!r} shares {ov:.0%} content "
                    f"words with learning question {lq!r}")
        # Seed scan: every *.aiml file under every seed dir.
        for seed_dir in seed_dirs or []:
            for root, _dirs, files in os.walk(seed_dir):
                for name in files:
                    if not name.endswith(".aiml"):
                        continue
                    path = os.path.join(root, name)
                    try:
                        with open(path, encoding="utf-8",
                                  errors="ignore") as f:
                            norm_file = normalize(f.read())
                    except OSError:
                        continue
                    if nbq and nbq in norm_file:
                        leaks.append(
                            f"benchmark question {bq!r} found in seed "
                            f"file {path}")
                    if nba and re.search(
                            r"\b" + re.escape(nba) + r"\b", norm_file):
                        leaks.append(
                            f"benchmark answer {ba!r} found in seed "
                            f"file {path}")
    return {"leaks": leaks}


class _FakeBackend(ChatBackend):
    """A backend that knows nothing: every model-tier call returns
    "n/a". The baseline therefore scores 0 unless the answer comes
    from retrieval — which, on a fresh temp dir, it cannot."""
    name = "fake-none"

    def complete(self, system, user, max_tokens=512, temperature=0.3):
        return "n/a"


def run_before_after(benchmark, learning, seed_dirs=None) -> dict:
    """Full protocol: fresh Brain in a temp dir -> benchmark (before)
    -> learn the learning set -> benchmark again (after).

    The leakage check runs FIRST, before the Brain is created, while
    the temp dir holds only the minimal seed — learned.aiml does not
    exist yet, so the scan cannot mistake taught facts for seed
    knowledge. Returns {"before", "after", "delta", "rows", "leaks"}.
    """
    tmp = tempfile.mkdtemp(prefix="ceti-learn-bench-")
    with open(os.path.join(tmp, "seed.aiml"), "w",
              encoding="utf-8") as f:
        f.write(_MINIMAL_SEED)
    with open(os.path.join(tmp, "topics.txt"), "w",
              encoding="utf-8") as f:
        f.write("")

    leaks = leakage_check(
        benchmark, learning, list(seed_dirs or []) + [tmp])["leaks"]

    cfg = BrainConfig(data_dir=tmp, adjudicate_with_llm=False,
                      memory_backend="sqlite", handler_mode=False,
                      enable_cloud=False)
    brain = Brain(config=cfg, local_backend=_FakeBackend(),
                  cloud_backends=[])
    try:
        before = run_benchmark(brain, benchmark)
        for pair in learning:
            brain._learn(pair["question"], pair["answer"],
                         score=0.9, source="benchmark")
        after = run_benchmark(brain, benchmark)
    finally:
        brain.close()

    rows = []
    for brow, arow in zip(before["rows"], after["rows"]):
        rows.append({
            "question": arow["question"],
            "answer": arow["answer"],
            "domain": arow["domain"],
            "correct_before": brow["correct"],
            "response_before": brow["response"],
            "tier_before": brow["tier"],
            "correct_after": arow["correct"],
            "response_after": arow["response"],
            "tier_after": arow["tier"],
        })
    return {"before": before, "after": after,
            "delta": after["accuracy"] - before["accuracy"],
            "rows": rows, "leaks": leaks}


def load_fixture(path: str) -> tuple[list, list]:
    """Load {"benchmark": [...], "learning": [...]} from JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["benchmark"], data["learning"]
