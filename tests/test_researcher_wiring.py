"""Researcher wiring tests: allowlist flag, interests, nightly budget."""
import os
import tempfile
import time
import unittest

from ceti.researcher import Researcher
from tests.test_researcher import make_test_brain


class TestResearcherWiring(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.brain = make_test_brain(self.tmp)
        topics = os.path.join(self.tmp, "topics.txt")
        with open(topics, "w") as f:
            f.write("dial-up modems\nansi art\n")
        self.r = Researcher(self.brain, topics_file=topics)
        self.r.idle_ok = lambda: (True, "test idle")

    def tearDown(self):
        self.brain.close()

    def test_cmd_default_is_wiki_only(self):
        cmd = self.r._research_cmd("dial-up modems")
        self.assertIn("--wiki-only", cmd)
        self.assertNotIn("--allow-sites", cmd)

    def test_cmd_custom_sites_passes_allowlist(self):
        self.r.config.add_site("example.com")
        cmd = self.r._research_cmd("dial-up modems")
        self.assertIn("--allow-sites", cmd)
        self.assertIn("en.wikipedia.org,example.com",
                      cmd[cmd.index("--allow-sites") + 1])
        self.assertNotIn("--wiki-only", cmd)

    def test_cmd_env_override_forces_wiki(self):
        self.r.config.add_site("example.com")
        os.environ["CETI_WIKI_ONLY"] = "1"
        try:
            cmd = self.r._research_cmd("dial-up modems")
        finally:
            del os.environ["CETI_WIKI_ONLY"]
        self.assertIn("--wiki-only", cmd)

    def test_next_topic_falls_back_to_interests(self):
        for t in ("dial-up modems", "ansi art"):
            self.r._log(t, start=True)
            self.r._log(t, qa_added=1, report="/tmp/x.md")
        self.assertIsNone(self.r.next_topic())
        self.r.config.add_interest("retro computing")
        self.assertEqual(self.r.next_topic(), "retro computing")

    def test_next_topic_exclude_skips_interest(self):
        self.r.config.add_interest("retro computing")
        self.r.config.add_interest("sovereign ai")
        got = self.r.next_topic(exclude={"dial-up modems", "ansi art",
                                         "retro computing"})
        self.assertEqual(got, "sovereign ai")

    def test_add_remove_topic(self):
        self.r.add_topic("packet radio")
        self.assertIn("packet radio", self.r.topics())
        self.r.add_topic("packet radio")
        self.assertEqual(self.r.topics().count("packet radio"), 1)
        self.r.remove_topic("packet radio")
        self.assertNotIn("packet radio", self.r.topics())
        with self.assertRaises(ValueError):
            self.r.remove_topic("packet radio")
        with self.assertRaises(ValueError):
            self.r.add_topic("   ")

    def test_run_nightly_respects_max_topics(self):
        calls = []
        self.r.research_topic = lambda t: (calls.append(t),
                                           {"qa_added": 1})[1]
        out = self.r.run_nightly()
        self.assertEqual(len(calls), 2)  # queue has 2 topics
        self.assertEqual(out["nightly"]["topics_done"], 2)
        self.assertIn("queue dry", out["nightly"]["stopped"])

    def test_run_nightly_respects_max_qa(self):
        self.r.config.set_budget(max_topics=10, max_qa=6)
        calls = []
        self.r.research_topic = lambda t: (calls.append(t),
                                           {"qa_added": 5})[1]
        out = self.r.run_nightly()
        self.assertEqual(len(calls), 2)  # 5 + 5 >= 6 -> stop
        self.assertEqual(out["nightly"]["qa_added"], 10)
        self.assertIn("max_qa", out["nightly"]["stopped"])

    def test_run_nightly_deadline(self):
        calls = []
        self.r.research_topic = lambda t: (calls.append(t),
                                           {"qa_added": 1})[1]
        out = self.r.run_nightly(deadline=time.time() - 1)
        self.assertEqual(calls, [])
        self.assertIn("deadline", out["nightly"]["stopped"])

    def test_run_nightly_single_interest_once(self):
        for t in ("dial-up modems", "ansi art"):
            self.r._log(t, start=True)
            self.r._log(t, qa_added=1, report="/tmp/x.md")
        self.r.config.add_interest("retro computing")
        self.r.config.set_budget(max_topics=5)
        calls = []
        self.r.research_topic = lambda t: (calls.append(t),
                                           {"qa_added": 1})[1]
        out = self.r.run_nightly()
        self.assertEqual(calls, ["retro computing"])
        self.assertIn("interests covered", out["nightly"]["stopped"])

    def test_status_includes_config(self):
        st = self.r.status()
        self.assertEqual(st["sites"], ["en.wikipedia.org"])
        self.assertEqual(st["interests"], [])
        self.assertEqual(st["nightly_budget"]["max_topics"], 5)


if __name__ == "__main__":
    unittest.main()
