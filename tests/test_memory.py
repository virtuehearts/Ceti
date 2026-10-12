import os
import tempfile
import unittest

from ceti.memory import MemoryIndex, token_sim, tokenize


class TestTokenSim(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(token_sim("bulletin", "bulletin"), 1.0)

    def test_transposition(self):
        # "dofferent" vs "different": single transposition-ish typo
        self.assertGreaterEqual(token_sim("dofferent", "different"), 0.6)

    def test_substitution(self):
        self.assertGreaterEqual(token_sim("bulleton", "bulletin"), 0.6)

    def test_containment(self):
        self.assertGreaterEqual(token_sim("boards", "board"), 0.6)

    def test_unrelated(self):
        self.assertLess(token_sim("is", "as"), 0.6)
        self.assertLess(token_sim("bbs", "modem"), 0.6)

    def test_contractions(self):
        self.assertIn("is", tokenize("what's a bbs"))
        self.assertNotIn("s", tokenize("what's a bbs"))


class TestMemoryFuzzy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.idx = MemoryIndex(os.path.join(self.tmp, "m.sqlite3"))
        self.answer = ("A BBS (Bulletin Board System) is a dial-up message "
                       "board system, because phone lines were the network "
                       "before the internet.")
        self.idx.add("What is a BBS?", self.answer, 0.9)
        self.n_alias = self.idx.learn_aliases_from(self.answer)

    def tearDown(self):
        self.idx.close()

    def test_aliases_learned(self):
        self.assertGreaterEqual(self.n_alias, 2)  # both directions
        self.assertGreaterEqual(self.idx.alias_count(), 2)

    def test_exact(self):
        hits = self.idx.search("What is a BBS?")
        self.assertTrue(hits)
        self.assertGreaterEqual(hits[0][3], 0.9)

    def test_paraphrase_with_expansion(self):
        # Jero's case: same intent, different words
        hits = self.idx.search("what's a bulletin board system")
        self.assertTrue(hits, "paraphrase should match")
        self.assertGreaterEqual(hits[0][3], 0.35)

    def test_paraphrase_reverse(self):
        hits = self.idx.search("tell me about bulletin boards")
        self.assertTrue(hits, "boards/board fuzzy should match")

    def test_typo(self):
        hits = self.idx.search("whta is a BBS")
        self.assertTrue(hits, "transposition typo should match")

    def test_unrelated_misses(self):
        hits = self.idx.search("how do I bake sourdough")
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
