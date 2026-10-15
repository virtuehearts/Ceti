import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ceti.calc import try_calculate


class TestCalc(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(try_calculate("2+2"), "4")
        self.assertEqual(try_calculate("3*4"), "12")
        self.assertEqual(try_calculate("10-3"), "7")
        self.assertEqual(try_calculate("7/2"), "3.5")
        self.assertEqual(try_calculate("7//2"), "3")

    def test_prefixes(self):
        self.assertEqual(try_calculate("what is 2+2?"), "4")
        self.assertEqual(try_calculate("calculate 10/4"), "2.5")
        self.assertEqual(try_calculate("= 5*5"), "25")
        self.assertEqual(try_calculate("compute (3+4)*2"), "14")

    def test_precedence(self):
        self.assertEqual(try_calculate("2+3*4"), "14")
        self.assertEqual(try_calculate("(2+3)*4"), "20")
        self.assertEqual(try_calculate("-5+10"), "5")

    def test_power(self):
        self.assertEqual(try_calculate("2^10"), "1024")
        self.assertEqual(try_calculate("2**10"), "1024")

    def test_funcs_consts(self):
        self.assertEqual(try_calculate("sqrt(144)"), "12")
        self.assertEqual(try_calculate("abs(-5)"), "5")
        self.assertEqual(try_calculate("round(3.14159, 2)"), "3.14")
        self.assertEqual(try_calculate("pi"), "3.1415926536")
        self.assertEqual(try_calculate("2*pi"), "6.2831853072")

    def test_percent(self):
        self.assertEqual(try_calculate("15% of 200"), "30")
        self.assertEqual(try_calculate("what is 20% of 150?"), "30")

    def test_float_cleanup(self):
        self.assertEqual(try_calculate("0.1+0.2"), "0.3")
        self.assertEqual(try_calculate("1,000 + 1,000"), "2000")

    def test_div_zero_deterministic(self):
        # never falls through to a model that would freelance it
        self.assertEqual(try_calculate("1/0"),
                         "undefined (division by zero)")

    def test_not_math(self):
        for q in ("hello", "what is your name", "what is the time",
                  "tell me about BBS", ""):
            self.assertIsNone(try_calculate(q), q)

    def test_injection_rejected(self):
        for q in ("__import__('os')", "open('/etc/passwd')",
                  "[1, 2, 3]", "().__class__", "lambda: 1",
                  "exec('1')", "sqrt.__class__"):
            self.assertIsNone(try_calculate(q), q)

    def test_huge_rejected(self):
        self.assertIsNone(try_calculate("10**10**10"))
        self.assertIsNone(try_calculate("9**9**9"))
        self.assertIsNone(try_calculate("x" * 300))


class TestCalcTier(unittest.TestCase):
    def test_calc_first_in_ladder(self):
        from ceti import decide
        from ceti.config import BrainConfig
        for intent in ("smalltalk", "command", "factual", "chat",
                       "unknown"):
            self.assertEqual(
                decide.plan(intent, BrainConfig())[0], "calc", intent)

    def test_brain_answers_math_deterministically(self):
        import tempfile
        from ceti.brain import Brain
        from ceti.config import BrainConfig
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "seed.aiml"), "w") as f:
            f.write('<?xml version="1.0"?><aiml version="1.0">'
                    '<category><pattern>HELLO</pattern>'
                    '<template>Hi.</template></category></aiml>')
        cfg = BrainConfig(data_dir=tmp, adjudicate_with_llm=False,
                          memory_backend="sqlite",
                          enable_local=False, enable_cloud=False)
        brain = Brain(config=cfg)
        ans = brain.ask("what is 2+2?")
        self.assertEqual(ans.tier, "calc")
        self.assertEqual(ans.text, "4")
        self.assertFalse(ans.learned)  # never learned, always computed
        # non-math still falls through normally
        ans2 = brain.ask("hello")
        self.assertEqual(ans2.tier, "aiml")


if __name__ == "__main__":
    unittest.main()
