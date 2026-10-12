"""End-to-end brain tests with fake LLM backends (no network)."""
import os
import tempfile
import unittest

from ceti import decide
from ceti.backends import BackendError, ChatBackend
from ceti.brain import Brain
from ceti.config import BrainConfig


class FakeLocal(ChatBackend):
    name = "fake-local"

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def complete(self, system, user, max_tokens=512, temperature=0.3):
        self.calls.append(user)
        if self.fail:
            raise BackendError("local down")
        return ("A BBS is a bulletin board system: users dialed in over "
                "phone lines to read and post messages, because that was "
                "the network before the internet.")


class FakeCloud(ChatBackend):
    name = "fake-cloud"

    def __init__(self):
        self.calls = []

    def complete(self, system, user, max_tokens=512, temperature=0.3):
        self.calls.append(user)
        return ("Cloud fallback answer: a BBS is a bulletin board system "
                "from the dial-up era, used because modems were the way "
                "computers talked to each other.")


def make_brain(**kw):
    tmp = tempfile.mkdtemp()
    local = kw.pop("local", FakeLocal())
    clouds = kw.pop("clouds", [FakeCloud()])
    cfg = BrainConfig(
        data_dir=tmp,
        adjudicate_with_llm=False,
        memory_backend="sqlite",  # hermetic: no embedding server in tests
        **kw,
    )
    # minimal seed so AIML tier exists
    with open(os.path.join(tmp, "seed.aiml"), "w") as f:
        f.write('<?xml version="1.0"?><aiml version="1.0">'
                '<category><pattern>HELLO</pattern>'
                '<template>Hi.</template></category></aiml>')
    return Brain(config=cfg, local_backend=local,
                 cloud_backends=clouds)


class TestDecide(unittest.TestCase):
    def test_intents(self):
        self.assertEqual(decide.classify_intent("hello"), "smalltalk")
        self.assertEqual(decide.classify_intent("How are you?"), "smalltalk")
        self.assertEqual(decide.classify_intent("run the tests"), "command")
        self.assertEqual(decide.classify_intent("What is a BBS?"), "factual")
        self.assertEqual(decide.classify_intent("tell me about modems"),
                         "chat")

    def test_smalltalk_skips_cloud(self):
        cfg = BrainConfig()
        self.assertNotIn("cloud", decide.plan("smalltalk", cfg))
        self.assertIn("cloud", decide.plan("factual", cfg))


class TestBrain(unittest.TestCase):
    def test_aiml_instant(self):
        b = make_brain()
        ans = b.ask("hello")
        self.assertEqual(ans.tier, "aiml")
        self.assertEqual(ans.text, "Hi.")
        self.assertFalse(ans.learned)
        b.close()

    def test_local_then_learned(self):
        b = make_brain()
        q = "What is a BBS?"
        ans1 = b.ask(q)
        self.assertEqual(ans1.tier, "local")
        self.assertTrue(ans1.learned,
                        f"expected promotion, got {ans1.trace}")
        # second ask: AIML answers instantly, no LLM call
        local_calls = b.local.calls
        ans2 = b.ask("what is a bbs?")
        self.assertEqual(ans2.tier, "aiml")
        self.assertEqual(b.local.calls, local_calls)
        b.close()

    def test_memory_tier(self):
        b = make_brain(enable_aiml=True)
        b.memory.add("What is a BBS?", "A dial-up message board.", 0.9)
        ans = b.ask("tell me what a BBS is")
        # fuzzy memory should catch the paraphrase before local LLM
        self.assertEqual(ans.tier, "memory")
        b.close()

    def test_handler_mode_presents(self):
        b = make_brain()  # handler_mode=True by default
        b.memory.add("What is a BBS?", "A dial-up message board.", 0.9)
        ans = b.ask("tell me what a BBS is")
        self.assertEqual(ans.tier, "memory")
        self.assertTrue(ans.trace.get("handled"))

    def test_handler_mode_off_returns_raw(self):
        b = make_brain(handler_mode=False)
        b.memory.add("What is a BBS?", "A dial-up message board.", 0.9)
        ans = b.ask("tell me what a BBS is")
        self.assertEqual(ans.tier, "memory")
        self.assertFalse(ans.trace.get("handled"))
        self.assertEqual(ans.text, "A dial-up message board.")

    def test_numbers_grounded(self):
        from ceti.brain import _numbers_grounded
        know = "CBBS was launched on February 16, 1978, by Ward Christensen."
        self.assertTrue(
            _numbers_grounded("It launched in 1978.", know, "when?"))
        self.assertFalse(
            _numbers_grounded("It launched in 1971.", know, "when?"))
        self.assertTrue(
            _numbers_grounded("no numbers here at all", know, "when?"))
        self.assertTrue(
            _numbers_grounded("In 1994, things changed.", know,
                              "what happened in 1994?"))
        self.assertTrue(
            _numbers_grounded("1,000 users by 1990.", "1000 users by 1990.",
                              "how many?"))

    def test_handler_ungrounded_falls_back_to_raw(self):
        class LyingLocal(ChatBackend):
            name = "lying-local"

            def complete(self, system, user, max_tokens=512,
                         temperature=0.3):
                return "The first BBS was launched in 1971."

        b = make_brain(handler_mode=True, local=LyingLocal())
        b.memory.add("When was CBBS launched?",
                     "CBBS was launched on February 16, 1978.", 0.9)
        ans = b.ask("when was CBBS launched")
        self.assertEqual(ans.tier, "memory")
        # confabulated date refused: raw knowledge served instead
        self.assertIn("1978", ans.text)
        self.assertNotIn("1971", ans.text)
        self.assertTrue(ans.trace.get("handler_ungrounded"))
        self.assertFalse(ans.trace.get("handled"))
        b.close()

    def test_persona_missing_is_empty(self):
        b = make_brain()  # tmp data dir has no persona.md
        self.assertEqual(b.persona(), "")
        b.close()

    def test_cloud_fallback(self):
        b = make_brain(local=FakeLocal(fail=True))
        ans = b.ask("What is a BBS?")
        self.assertEqual(ans.tier, "cloud")
        b.close()

    def test_cloud_gets_stripped_query(self):
        cloud = FakeCloud()
        b = make_brain(local=FakeLocal(fail=True), clouds=[cloud])
        b.ask("What is a BBS? mail jero@darknet.ca")
        self.assertNotIn("jero@darknet.ca", cloud.calls[0])
        self.assertIn("[REDACTED_EMAIL]", cloud.calls[0])
        b.close()

    def test_personal_not_learned(self):
        b = make_brain()
        ans = b.ask("What is my phone number?")
        self.assertFalse(ans.learned)
        b.close()

    def test_force_tier(self):
        b = make_brain()
        ans = b.ask("hello", force_tier="local")
        self.assertEqual(ans.tier, "local")
        b.close()


class ParaphraseLocal(ChatBackend):
    name = "paraphrase-local"

    def complete(self, system, user, max_tokens=512, temperature=0.3):
        return ("A BBS (Bulletin Board System) is a dial-up message board "
                "system, because phone lines were the network before the "
                "internet.")


class TestParaphrase(unittest.TestCase):
    """Same intent, different words: the brain should still answer fast."""

    def test_paraphrase_hits_memory(self):
        b = make_brain(local=ParaphraseLocal())
        ans1 = b.ask("What is a BBS?")
        self.assertTrue(ans1.learned, f"expected promote: {ans1.trace}")
        # Jero's case: same intent, spelled different
        ans2 = b.ask("what's a bulletin board system")
        self.assertEqual(ans2.tier, "memory")
        self.assertIn("Bulletin Board System", ans2.text)
        b.close()

    def test_typo_hits_memory(self):
        b = make_brain(local=ParaphraseLocal())
        b.ask("What is a BBS?")
        ans = b.ask("whta is a BBS")
        self.assertEqual(ans.tier, "memory")
        b.close()

    def test_article_variant_hits_aiml(self):
        b = make_brain(local=ParaphraseLocal())
        b.ask("What is a BBS?")
        ans = b.ask("what is bbs")
        self.assertEqual(ans.tier, "aiml")
        b.close()


if __name__ == "__main__":
    unittest.main()
