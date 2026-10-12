"""Researcher tests: stubbed research + LLM (no network)."""
import json
import os
import tempfile
import unittest

from ceti.backends import ChatBackend
from ceti.brain import Brain
from ceti.config import BrainConfig
from ceti.researcher import Researcher


class FakeDistill(ChatBackend):
    name = "fake-distill"

    def complete(self, system, user, max_tokens=512, temperature=0.3):
        return json.dumps([{
            "q": "What is a dial-up modem?",
            "a": ("A dial-up modem is a device that modulates digital "
                  "data into audio signals for telephone lines, because "
                  "phone networks were built for voice."),
        }])


def make_test_brain(tmp):
    cfg = BrainConfig(data_dir=tmp, adjudicate_with_llm=False,
                      memory_backend="sqlite")
    with open(os.path.join(tmp, "seed.aiml"), "w") as f:
        f.write('<?xml version="1.0"?><aiml version="1.0"></aiml>')
    return Brain(config=cfg, local_backend=FakeDistill(),
                 cloud_backends=[])


class TestResearcher(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.brain = make_test_brain(self.tmp)
        topics = os.path.join(self.tmp, "topics.txt")
        with open(topics, "w") as f:
            f.write("# comment\n\ndial-up modems\nansi art\n")
        self.r = Researcher(self.brain, topics_file=topics)
        self.r.idle_ok = lambda: (True, "test idle")

    def tearDown(self):
        self.brain.close()

    def test_topics_parsing(self):
        self.assertEqual(self.r.topics(), ["dial-up modems", "ansi art"])

    def test_next_topic_skips_done(self):
        self.assertEqual(self.r.next_topic(), "dial-up modems")
        self.r._log("dial-up modems", start=True)
        self.r._log("dial-up modems", qa_added=1, report="/tmp/x.md")
        self.assertEqual(self.r.next_topic(), "ansi art")

    def test_idle_skip(self):
        self.r.idle_ok = lambda: (False, "busy")
        out = self.r.idle_cycle()
        self.assertIn("skipped", out)

    def test_full_cycle_records_cassette(self):
        report = os.path.join(self.tmp, "rep.md")
        with open(report, "w") as f:
            f.write("# Research: dial-up modems\n\nSome findings.")
        self.r._run_research = lambda topic: report
        out = self.r.idle_cycle()
        self.assertEqual(out["topic"], "dial-up modems")
        self.assertEqual(out["qa_added"], 1)
        self.assertIn("dial-up-modems", self.r.rack.list())
        # cassette is hot-swapped: the live brain answers instantly
        ans = self.brain.ask("what is a dial-up modem")
        self.assertEqual(ans.tier, "aiml")
        self.assertIn("modem", ans.text.lower())

    def test_distill_bad_json(self):
        report = os.path.join(self.tmp, "rep.md")
        with open(report, "w") as f:
            f.write("hello")
        self.r._llm = lambda prompt: "not json at all"
        self.assertEqual(self.r._distill(report, "t"), [])

    def test_status(self):
        s = self.r.status()
        self.assertIn("cassettes", s)
        self.assertIn("completed", s)


if __name__ == "__main__":
    unittest.main()
