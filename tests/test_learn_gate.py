"""Learning gates: the truth judge and the contradiction check.

Regression coverage for the 1977 glitch — a confident model
answer ("the C64 was introduced in 1977") scored 0.85 on
relevance alone and was promoted into permanent memory.
Now: factual promotes need a truth-judge YES, and _learn
refuses anything that contradicts existing knowledge."""
import os
import tempfile
import unittest

from ceti.backends import ChatBackend
from ceti.brain import Brain
from ceti.classify import RelevanceClassifier
from ceti.config import BrainConfig


class FakeJudgeLocal(ChatBackend):
    """Answers questions with `answer`; answers judge prompts
    with the configured verdict."""
    name = "fake-judge-local"

    def __init__(self, answer, judge="YES"):
        self.answer = answer
        self.judge = judge
        self.judge_calls = 0

    def complete(self, system, user, max_tokens=512, temperature=0.3):
        if "fact checker" in user:
            self.judge_calls += 1
            return self.judge
        if "relevance judge" in user:
            return "YES"
        return self.answer


def make_brain(local, gap_log=None, **kw):
    tmp = tempfile.mkdtemp()
    cfg = BrainConfig(data_dir=tmp, adjudicate_with_llm=True,
                      memory_backend="sqlite", **kw)
    with open(os.path.join(tmp, "seed.aiml"), "w") as f:
        f.write('<?xml version="1.0"?><aiml version="1.0">'
                '<category><pattern>HELLO</pattern>'
                '<template>Hi.</template></category></aiml>')
    return Brain(config=cfg, local_backend=local, cloud_backends=[],
                 gap_log=gap_log)


class FakeGap:
    def __init__(self):
        self.gaps = []

    def note_gap(self, q):
        self.gaps.append(q)


class TestTruthJudge(unittest.TestCase):
    def test_verify_truth(self):
        c = RelevanceClassifier()
        self.assertTrue(c.verify_truth("q", "a",
                                       lambda p: "YES, definitely"))
        self.assertFalse(c.verify_truth("q", "a", lambda p: "NO"))
        self.assertFalse(c.verify_truth("q", "a", lambda p: 1 / 0))

    def test_wrong_answer_not_learned(self):
        brain = make_brain(FakeJudgeLocal(
            "The C64 was introduced in 1977.", judge="NO"))
        ans = brain.ask("when was the c64 introduced")
        self.assertFalse(ans.learned)
        self.assertIs(ans.trace.get("truth_check"), False)
        # and it never entered the stores
        self.assertEqual(brain.memory.search(
            "when was the c64 introduced"), [])

    def test_right_answer_still_learns(self):
        brain = make_brain(FakeJudgeLocal(
            "The C64 was introduced in January 1982.", judge="YES"))
        ans = brain.ask("when was the c64 introduced")
        self.assertTrue(ans.learned)
        self.assertIs(ans.trace.get("truth_check"), True)
        hits = brain.memory.search("when was the c64 introduced")
        self.assertTrue(any("1982" in h[1] for h in hits))


class TestContradictionCheck(unittest.TestCase):
    def test_memory_conflict_blocks_learn(self):
        gap = FakeGap()
        brain = make_brain(FakeJudgeLocal("x"), gap_log=gap)
        brain.memory.add("when was the c64 introduced",
                         "The C64 was introduced in January 1982.", 1.0)
        ok = brain._learn("when was the c64 introduced",
                          "The C64 was introduced in 1977.", 0.9)
        self.assertFalse(ok)
        self.assertIn("memory", brain._learn_block_reason)
        self.assertIn("when was the c64 introduced", gap.gaps)

    def test_aiml_conflict_blocks_learn(self):
        brain = make_brain(FakeJudgeLocal("x"))
        brain.aiml.learn("when was the c64 introduced",
                         "The C64 was introduced in January 1982.")
        ok = brain._learn("when was the c64 introduced",
                          "The C64 was introduced in 1977.", 0.9)
        self.assertFalse(ok)
        self.assertIn("AIML", brain._learn_block_reason)

    def test_no_conflict_learns(self):
        brain = make_brain(FakeJudgeLocal("x"))
        ok = brain._learn("what is a bbs",
                          "A BBS is a bulletin board system.", 0.9)
        self.assertTrue(ok)

    def test_elaboration_is_not_a_conflict(self):
        brain = make_brain(FakeJudgeLocal("x"))
        brain.memory.add("when was the c64 introduced",
                         "January 1982", 1.0)
        ok = brain._learn(
            "when was the c64 introduced",
            "The C64 was introduced in January 1982.", 0.9)
        self.assertTrue(ok)

    def test_answers_conflict(self):
        c = Brain._answers_conflict
        self.assertFalse(c("January 1982", "january 1982"))
        self.assertFalse(c("January 1982",
                           "The C64 was introduced in January 1982."))
        self.assertTrue(c("The C64 was introduced in 1977.",
                          "The C64 was introduced in January 1982."))
        self.assertTrue(c("Paris", "London"))


if __name__ == "__main__":
    unittest.main()
