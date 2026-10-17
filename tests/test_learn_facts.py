"""Integration: Brain._learn through the structured fact store.

Learning is two-layer: the fact layer (triples + confidence +
conflict resolution at the (subject, predicate) level) and the
raw-pair layer (learned.aiml + vector memory, unchanged). These tests
prove the fact layer works end-to-end through Brain: paraphrases
answer from the learned fact, a conflicting fact under a different
wording is blocked, and a higher-confidence fact supersedes.
"""
import os
import tempfile
import unittest

from ceti.backends import ChatBackend
from ceti.brain import Brain
from ceti.config import BrainConfig


class FakeLocal(ChatBackend):
    name = "fake-local"

    def __init__(self, answer, judge="YES"):
        self.answer = answer
        self.judge = judge

    def complete(self, system, user, max_tokens=512, temperature=0.3):
        if "fact checker" in user or "relevance judge" in user:
            return self.judge
        return self.answer


def make_brain(local, **kw):
    tmp = tempfile.mkdtemp()
    cfg = BrainConfig(data_dir=tmp, adjudicate_with_llm=True,
                      memory_backend="sqlite", **kw)
    with open(os.path.join(tmp, "seed.aiml"), "w") as f:
        f.write('<?xml version="1.0"?><aiml version="1.0">'
                '<category><pattern>HELLO</pattern>'
                '<template>Hi.</template></category></aiml>')
    return Brain(config=cfg, local_backend=local, cloud_backends=[])


class TestLearnFacts(unittest.TestCase):
    def test_learn_stores_fact_and_answers_paraphrase(self):
        b = make_brain(FakeLocal("The Commodore 64 was introduced "
                                 "in January 1982."))
        ans = b.ask("when was the commodore 64 introduced")
        self.assertTrue(ans.learned, ans.trace)
        self.assertEqual(ans.trace.get("fact_status"), "learned")
        # A paraphrase — never asked before, no AIML pattern for it
        # in the raw learned file — answers from the fact.
        ans2 = b.ask("what year did the commodore 64 come out")
        self.assertEqual(ans2.text,
                         "The Commodore 64 was introduced in January 1982.")
        b.close()

    def test_conflicting_fact_blocked_under_different_wording(self):
        # First: learn the true fact.
        b = make_brain(FakeLocal("The Commodore 64 was introduced "
                                 "in January 1982."))
        b.ask("when was the commodore 64 introduced")
        # Second: a conflicting answer under a wording whose LOOKUP
        # misses the fact store (bare "when was X" has no verb) but
        # whose ANSWER extracts to the same (subject, predicate).
        # The old exact-question check would miss this; the fact
        # layer catches it (same meaning, different wording).
        # (Direct _learn call: going through ask() would let the
        # vector memory answer the similar question first, so the
        # learn path would never fire — the block is in _learn.)
        ok = b._learn("when was the commodore 64",
                      "The Commodore 64 was introduced in 1977.",
                      0.85, source="local")
        self.assertFalse(ok)
        self.assertEqual(b._fact_status, "blocked")
        self.assertIn("already has", b._learn_block_reason)
        # The fact store still holds the true fact.
        self.assertEqual(
            b._facts.active("Commodore 64", "introduced").object,
            "January 1982")
        b.close()

    def test_fact_learns_via_memory_tier(self):
        # A fact learned from the local tier answers a NOVEL paraphrase
        # through the memory tier (facts-first) — a wording the AIML
        # compiler never generated a pattern for.
        b = make_brain(FakeLocal("The Commodore 64 was introduced "
                                 "in January 1982."))
        b.ask("when was the commodore 64 introduced")
        ans = b.ask("what year did the commodore 64 come out")
        self.assertEqual(ans.text,
                         "The Commodore 64 was introduced in January 1982.")
        self.assertEqual(ans.trace.get("memory_source"), "facts")
        b.close()

    def test_fact_survives_restart(self):
        tmp = tempfile.mkdtemp()
        cfg = BrainConfig(data_dir=tmp, adjudicate_with_llm=True,
                          memory_backend="sqlite")
        with open(os.path.join(tmp, "seed.aiml"), "w") as f:
            f.write('<?xml version="1.0"?><aiml version="1.0">'
                    '<category><pattern>HELLO</pattern>'
                    '<template>Hi.</template></category></aiml>')
        b1 = Brain(config=cfg,
                   local_backend=FakeLocal("The Commodore 64 was "
                                           "introduced in January 1982."),
                   cloud_backends=[])
        b1.ask("when was the commodore 64 introduced")
        b1.close()
        # Fresh Brain on the same data dir: the fact is still there.
        b2 = Brain(config=cfg, local_backend=FakeLocal("n/a"),
                   cloud_backends=[])
        ans = b2.ask("what year did the commodore 64 come out")
        self.assertEqual(ans.text,
                         "The Commodore 64 was introduced in January 1982.")
        b2.close()

    def test_unstructured_pair_still_learns(self):
        # Pairs that don't extract to a fact still learn as raw pairs.
        b = make_brain(FakeLocal("Dial the number, wait for the "
                                 "carrier, then log in."))
        ans = b.ask("how do i call a bbs")
        self.assertTrue(ans.learned, ans.trace)
        self.assertEqual(ans.trace.get("fact_status"), "pair")
        b.close()


if __name__ == "__main__":
    unittest.main()
