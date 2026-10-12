import os
import tempfile
import unittest

from ceti.aiml import AimlBrain, normalize_input


class TestAiml(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.seed = os.path.join(self.tmp, "seed.aiml")
        with open(self.seed, "w") as f:
            f.write("""<?xml version="1.0"?>
<aiml version="1.0">
<category><pattern>HELLO</pattern><template>Hi there.</template></category>
<category><pattern>HI</pattern><template><srai>HELLO</srai></template></category>
<category><pattern>MY NAME IS *</pattern><template>Nice to meet you, <star/>.</template></category>
<category><pattern>WHAT IS *</pattern><template>I don't know what <star/> is.</template></category>
<category><pattern>PICK * OR *</pattern><template><random><li><star index="2"/></li><li><star/></li></random></template></category>
</aiml>""")
        self.learned = os.path.join(self.tmp, "learned.aiml")
        self.brain = AimlBrain(seed_path=self.seed,
                               learned_path=self.learned)

    def test_normalize(self):
        self.assertEqual(normalize_input("  Hello,  World! "), "HELLO WORLD")

    def test_article_insensitive(self):
        self.assertEqual(normalize_input("what is a bbs"), "WHAT IS BBS")
        self.assertEqual(normalize_input("what is the bbs"), "WHAT IS BBS")

    def test_exact(self):
        self.assertEqual(self.brain.respond("hello"), "Hi there.")

    def test_srai(self):
        self.assertEqual(self.brain.respond("hi"), "Hi there.")

    def test_wildcard_star(self):
        out = self.brain.respond("my name is Jero")
        self.assertEqual(out, "Nice to meet you, JERO.")

    def test_wildcard_specificity(self):
        # WHAT IS * beats nothing else here; check capture works
        # (articles are dropped in normalization: "a bbs" -> "BBS")
        out = self.brain.respond("what is a bbs")
        self.assertEqual(out, "I don't know what BBS is.")

    def test_random_star_index(self):
        out = self.brain.respond("pick cats or dogs")
        self.assertIn(out, ("CATS", "DOGS"))

    def test_miss(self):
        self.assertIsNone(self.brain.respond("xyzzy plugh nonsense"))

    def test_learn_roundtrip(self):
        self.assertTrue(
            self.brain.learn("What is a BBS?",
                             "A dial-up bulletin board system."))
        # now instant:
        self.assertEqual(self.brain.respond("what is a bbs?"),
                         "A dial-up bulletin board system.")
        # and persisted (articles dropped from the stored pattern):
        with open(self.learned) as f:
            xml = f.read()
        self.assertIn("WHAT IS BBS", xml)
        # reload from disk:
        b2 = AimlBrain(seed_path=self.seed, learned_path=self.learned)
        self.assertEqual(b2.respond("what is a bbs"),
                         "A dial-up bulletin board system.")

    def test_learn_dedupe(self):
        self.assertTrue(self.brain.learn("What is a BBS?", "answer one"))
        self.assertFalse(self.brain.learn("What is a BBS?", "answer two"))
        self.assertEqual(self.brain.respond("what is a bbs"), "answer one")


if __name__ == "__main__":
    unittest.main()
