"""VectorMemory tests with a fake embedder (no network, no server)."""
import math
import os
import tempfile
import unittest

from ceti.vectormem import VectorMemory, cosine, pack, unpack


def fake_embed(texts):
    # deterministic 64-dim vectors from word hashes (crc32, NOT hash():
    # hash() is salted per process). Texts sharing words get similar
    # vectors — good enough to test the plumbing.
    import zlib
    vecs = []
    for t in texts:
        words = t.lower().split()
        v = [0.0] * 64
        for w in words:
            v[zlib.crc32(w.encode()) % 64] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / n for x in v])
    return vecs


class TestVecMath(unittest.TestCase):
    def test_pack_roundtrip(self):
        v = [0.1, -0.5, 0.0, 1.0]
        r = unpack(pack(v))
        for a, b in zip(v, r):
            self.assertAlmostEqual(a, b, places=5)

    def test_cosine(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cosine([1, 1], [1, 1]), 1.0)


class TestVectorMemory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vm = VectorMemory(os.path.join(self.tmp, "k.sqlite3"),
                               embed_fn=fake_embed)

    def tearDown(self):
        self.vm.close()

    def test_add_and_search(self):
        self.assertTrue(self.vm.add(
            "What is a BBS?",
            "A bulletin board system from the dial-up era.", 0.9))
        self.assertEqual(self.vm.count(), 1)
        hits = self.vm.search("what is a BBS")
        self.assertTrue(hits)
        self.assertEqual(hits[0][0], "What is a BBS?")
        related = hits[0][3]
        other = self.vm.search("sourdough bread recipe")
        self.assertGreater(related, other[0][3],
                           "related query must outscore unrelated")

    def test_dedupe(self):
        self.vm.add("What is a BBS?", "answer one", 0.9)
        self.assertFalse(self.vm.add("What is a BBS?", "answer two", 0.9))

    def test_unrelated_scores_lower(self):
        self.vm.add("What is a BBS?",
                    "A bulletin board system from the dial-up era.", 0.9)
        hits = self.vm.search("sourdough bread recipe")
        # unrelated query should score well below the real match
        self.assertLess(hits[0][3], 0.5)

    def test_aliases_recorded(self):
        n = self.vm.learn_aliases_from(
            "A BBS (Bulletin Board System) is old.")
        self.assertGreaterEqual(n, 2)
        self.assertGreaterEqual(self.vm.alias_count(), 2)

    def test_research_log(self):
        self.vm.log_research_start("dial-up modems")
        self.assertNotIn("dial-up modems", self.vm.researched_topics())
        self.vm.log_research_done("dial-up modems", 5, "/tmp/r.md")
        self.assertIn("dial-up modems", self.vm.researched_topics())

    def test_ping_uses_fake(self):
        self.assertTrue(self.vm.ping())


if __name__ == "__main__":
    unittest.main()
