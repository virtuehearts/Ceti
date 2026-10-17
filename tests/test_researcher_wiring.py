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

    def test_extract_pairs_pure_json(self):
        from ceti.researcher import _extract_pairs
        raw = ('[{"q": "What is the SID?", "a": "A sound chip."},'
               ' {"q": "Who made it?", "a": "MOS Technology."}]')
        pairs = _extract_pairs(raw)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["q"], "What is the SID?")

    def test_extract_pairs_prose_wrapped(self):
        from ceti.researcher import _extract_pairs
        raw = ('Here is the list:\n'
               '[{"q": "What is the SID?", "a": "A sound chip."}]\n'
               'Hope that helps.')
        self.assertEqual(len(_extract_pairs(raw)), 1)

    def test_extract_pairs_trailing_bracket_note(self):
        # the old greedy \\[.*\\] spanned to the LAST ], so a trailing
        # "[note: ...]" broke the parse and the topic distilled to 0
        from ceti.researcher import _extract_pairs
        raw = ('[{"q": "What is the SID?", "a": "A sound chip."}]\n'
               '[note: check the 8580 date]')
        self.assertEqual(len(_extract_pairs(raw)), 1)

    def test_extract_pairs_invalid(self):
        from ceti.researcher import _extract_pairs
        self.assertEqual(_extract_pairs("no json here"), [])
        self.assertEqual(_extract_pairs('{"q": "not a list"}'), [])

    def test_distill_retries_after_garbage(self):
        good = ('[{"q": "What is the SID?", "a": "A sound chip."}]')
        calls = {"n": 0}

        def fake_llm(prompt):
            calls["n"] += 1
            return "not json at all" if calls["n"] == 1 else good

        self.r._llm = fake_llm
        self.r._llm_cloud = lambda prompt: None
        report = os.path.join(self.tmp, "rep.md")
        with open(report, "w") as f:
            f.write("# Report\nSome findings.\n")
        pairs = self.r._distill(report, "SID")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(calls["n"], 2)

    def test_distill_falls_back_to_cloud(self):
        good = ('[{"q": "What is the SID?", "a": "A sound chip."}]')
        self.r._llm = lambda prompt: "garbage, no brackets"
        self.r._llm_cloud = lambda prompt: good
        report = os.path.join(self.tmp, "rep.md")
        with open(report, "w") as f:
            f.write("# Report\nSome findings.\n")
        pairs = self.r._distill(report, "SID")
        self.assertEqual(len(pairs), 1)


if __name__ == "__main__":
    unittest.main()


class TestResearcherFactGate(unittest.TestCase):
    """The distiller-bypass fix: _record routes distilled pairs
    through the fact layer. A pair that conflicts with an existing
    fact is dropped — never written to the cassette or the memory
    store."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.brain = make_test_brain(self.tmp)
        topics = os.path.join(self.tmp, "topics.txt")
        with open(topics, "w") as f:
            f.write("commodore 64\n")
        self.r = Researcher(self.brain, topics_file=topics)

    def tearDown(self):
        self.brain.close()

    def test_conflicting_distilled_pair_blocked(self):
        # Seed the fact store with the true fact.
        self.brain._facts.learn(
            "when was the commodore 64 introduced",
            "The Commodore 64 was introduced in January 1982.",
            confidence=0.9, source="research")
        # A distilled pair that conflicts (different object).
        added = self.r._record("commodore 64", [
            {"q": "when was the commodore 64 introduced",
             "a": "The Commodore 64 was introduced in 1977."}])
        self.assertEqual(added, 0)
        # Not in the cassette.
        cassette = self.r.rack.path("commodore-64")
        with open(cassette) as f:
            self.assertNotIn("1977", f.read())
        # The fact store still holds the true fact.
        self.assertEqual(
            self.brain._facts.active("Commodore 64", "introduced").object,
            "January 1982")

    def test_non_conflicting_distilled_pair_recorded(self):
        added = self.r._record("commodore 64", [
            {"q": "when was the commodore 64 introduced",
             "a": "The Commodore 64 was introduced in January 1982."}])
        self.assertEqual(added, 1)
        cassette = self.r.rack.path("commodore-64")
        with open(cassette) as f:
            self.assertIn("January 1982", f.read())

    def test_unstructured_pair_still_recorded(self):
        # Pairs that don't extract to a fact pass through (the fact
        # layer can't judge them — the human audit still applies).
        added = self.r._record("commodore 64", [
            {"q": "how do i call a bbs",
             "a": "Dial the number, wait for the carrier, log in."}])
        self.assertEqual(added, 1)
