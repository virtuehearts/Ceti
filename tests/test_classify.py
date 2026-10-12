import unittest

from ceti.classify import RelevanceClassifier


class TestClassify(unittest.TestCase):
    def setUp(self):
        self.c = RelevanceClassifier()

    def test_general_knowledge_promotes(self):
        v = self.c.decide(
            "What is a BBS?",
            "A bulletin board system: a dial-up server users called to "
            "read and post messages, because phone lines were the network "
            "before the internet.")
        self.assertEqual(v.action, "promote")
        self.assertGreaterEqual(v.score, 0.65)

    def test_personal_discards(self):
        v = self.c.decide(
            "What is my phone number?",
            "Your phone number is 416-555-0199.")
        self.assertEqual(v.action, "discard")

    def test_time_sensitive_discards(self):
        v = self.c.decide(
            "What is today's weather?",
            "I don't have real-time data access right now.")
        self.assertEqual(v.action, "discard")

    def test_command_discards(self):
        v = self.c.decide(
            "Run the backup script",
            "Running the backup script now, because you asked.")
        self.assertEqual(v.action, "discard")

    def test_hedge_penalized(self):
        v = self.c.decide(
            "What is the stack?",
            "As an AI language model, I don't have a physical stack. "
            "However, common Python stacks include Flask and Django.")
        self.assertLess(v.score, 0.65)

    def test_adjudicate_band(self):
        # crisp one-word fact -> borderline -> cheap LLM judge decides
        v = self.c.decide("What is the capital of France?", "Paris.")
        self.assertEqual(v.action, "adjudicate")
        # short but complete explanation -> promotable
        v2 = self.c.decide("Why is the sky blue?",
                           "Because of Rayleigh scattering.")
        self.assertIn(v2.action, ("promote", "adjudicate"))

    def test_adjudicate_yes(self):
        ok = self.c.adjudicate("q", "a", lambda prompt: "YES, save it.")
        self.assertTrue(ok)

    def test_adjudicate_no(self):
        ok = self.c.adjudicate("q", "a", lambda prompt: "NO.")
        self.assertFalse(ok)

    def test_adjudicate_failure_defaults_no(self):
        def boom(prompt):
            raise RuntimeError("llm down")
        self.assertFalse(self.c.adjudicate("q", "a", boom))


if __name__ == "__main__":
    unittest.main()
