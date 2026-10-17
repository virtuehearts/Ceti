"""Memory tier reads the /ingest library, not just the Q&A store.

Regression: the Library was a separate vector store the ladder never
read, so a curated document (mike.txt) could not answer "who did
John call the first BBS with" — the local model confabulated
"The New World" instead, and the truth judge approved it.
"""
import os
import tempfile
import unittest

from ceti.brain import Brain
from ceti.config import BrainConfig


class FakeLibrary:
    def __init__(self, hits=None, exc=None):
        self._hits = hits if hits is not None else []
        self._exc = exc

    def search(self, query, top_k=1):
        if self._exc is not None:
            raise self._exc
        return self._hits[:top_k]

    def close(self):
        pass


def make_brain(tmp):
    cfg = BrainConfig(data_dir=tmp, memory_backend="sqlite")
    cfg.handler_mode = False  # raw knowledge, no handler rewrite
    with open(os.path.join(tmp, "seed.aiml"), "w") as f:
        f.write('<?xml version="1.0"?><aiml version="1.0"></aiml>')
    return Brain(config=cfg, local_backend=None, cloud_backends=[])


class TestLibraryInMemoryTier(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.brain = make_brain(self.tmp)

    def tearDown(self):
        self.brain.close()

    def test_library_answers_before_qa_store(self):
        doc = "I called the first BBSs with Mike, one of my sister's friends."
        self.brain._library = lambda: FakeLibrary(
            [{"text": doc, "score": 0.9, "filename": "mike.txt",
              "chunk_no": 0}])
        out = self.brain._tier_memory("who did john call the first bbs with")
        self.assertEqual(out, doc)
        self.assertEqual(self.brain._memory_source, "library")

    def test_library_below_floor_falls_back_to_qa(self):
        self.brain.memory.add("what is a bbs",
                              "A bulletin board system.", 0.9)
        self.brain._library = lambda: FakeLibrary(
            [{"text": "irrelevant", "score": 0.1, "filename": "x.txt",
              "chunk_no": 0}])
        out = self.brain._tier_memory("what is a bbs")
        self.assertEqual(out, "A bulletin board system.")
        self.assertEqual(self.brain._memory_source, "knowledge")

    def test_library_failure_falls_through(self):
        self.brain.memory.add("what is a bbs",
                              "A bulletin board system.", 0.9)
        self.brain._library = lambda: FakeLibrary(exc=RuntimeError("boom"))
        out = self.brain._tier_memory("what is a bbs")
        self.assertEqual(out, "A bulletin board system.")

    def test_no_library_no_qa_is_miss(self):
        self.brain._library = lambda: FakeLibrary([])
        self.assertIsNone(
            self.brain._tier_memory("who did john call the first bbs with"))
        self.assertIsNone(self.brain._memory_source)


if __name__ == "__main__":
    unittest.main()
