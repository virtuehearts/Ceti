import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ceti import ingest
from ceti.ingest import (Library, chunk_text, extract_text, sha256_file,
                         IngestError)
from ceti.vectormem import DIM


def fake_embed(texts):
    """Keyword-presence vectors: dim0=cat, dim1=dog, dim2=other."""
    vecs = []
    for t in texts:
        t = t.lower()
        v = [0.0] * DIM
        v[0] = float(t.count("cat"))
        v[1] = float(t.count("dog"))
        v[2] = float(len(t) % 7)
        vecs.append(v)
    return vecs


class FakeLibrary(Library):
    def _embed(self, texts):
        return fake_embed(texts)

    def ping(self):
        return True


class TestChunking(unittest.TestCase):
    def test_paragraph_packing(self):
        text = "\n\n".join("para %d. " % i + "x" * 200 for i in range(20))
        chunks = chunk_text(text, chunk_chars=1000, overlap_chars=100)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 1100)

    def test_short_text_single_chunk(self):
        self.assertEqual(len(chunk_text("hello world " * 10)), 1)

    def test_giant_paragraph_split(self):
        chunks = chunk_text("z" * 5000, chunk_chars=1000,
                            overlap_chars=100)
        self.assertGreater(len(chunks), 3)

    def test_tiny_chunks_dropped(self):
        self.assertEqual(chunk_text("hi"), [])


class TestExtract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ceti-ingest-")

    def _f(self, name, content):
        p = os.path.join(self.tmp, name)
        with open(p, "w") as f:
            f.write(content)
        return p

    def test_txt(self):
        p = self._f("a.txt", "hello " * 100)
        self.assertIn("hello", extract_text(p))

    def test_unsupported(self):
        p = self._f("a.exe", "x" * 500)
        with self.assertRaises(IngestError):
            extract_text(p)

    def test_too_short(self):
        p = self._f("b.txt", "tiny")
        with self.assertRaises(IngestError):
            extract_text(p)

    def test_sha256(self):
        p = self._f("c.txt", "data " * 100)
        self.assertEqual(sha256_file(p), sha256_file(p))
        self.assertEqual(len(sha256_file(p)), 64)


class TestLibrary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ceti-ingest-")
        self.lib = FakeLibrary(self.tmp)

    def tearDown(self):
        self.lib.close()

    def _doc(self, name, text):
        p = os.path.join(self.lib.libdir, name)
        with open(p, "w") as f:
            f.write(text)
        return p

    def test_ingest_and_search(self):
        self._doc("cats.txt", "cat " * 400 + "\n\n" + "cat feline " * 400)
        self._doc("dogs.txt", "dog " * 400 + "\n\n" + "dog canine " * 400)
        r1 = self.lib.ingest_file(os.path.join(self.lib.libdir,
                                               "cats.txt"))
        r2 = self.lib.ingest_file(os.path.join(self.lib.libdir,
                                               "dogs.txt"))
        self.assertEqual(r1["status"], "ingested")
        self.assertGreater(r1["chunks"], 0)
        hits = self.lib.search("cat feline whiskers", top_k=2)
        self.assertEqual(hits[0]["filename"], "cats.txt")
        self.assertGreater(hits[0]["score"], hits[1]["score"])

    def test_dedup(self):
        p = self._doc("d.txt", "dup " * 300)
        self.assertEqual(self.lib.ingest_file(p)["status"], "ingested")
        self.assertEqual(self.lib.ingest_file(p)["status"],
                         "already-ingested")
        self.assertEqual(self.lib.count_chunks(),
                         self.lib.count_chunks())  # unchanged

    def test_scan(self):
        self._doc("s1.txt", "scan me " * 300)
        self._doc("s2.txt", "scan me too " * 300)
        results = self.lib.scan()
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["status"] == "ingested" for r in results))
        # second scan: nothing new
        results = self.lib.scan()
        self.assertTrue(all(r["status"] == "already-ingested"
                            for r in results))

    def test_drop(self):
        p = self._doc("drop.txt", "bye " * 300)
        self.lib.ingest_file(p)
        before = self.lib.count_chunks()
        self.assertTrue(self.lib.drop("drop.txt"))
        self.assertLess(self.lib.count_chunks(), before)
        self.assertFalse(self.lib.drop("drop.txt"))

    def test_docs_list(self):
        self._doc("l.txt", "listed " * 300)
        self.lib.scan()
        docs = self.lib.docs()
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["filename"], "l.txt")


if __name__ == "__main__":
    unittest.main()
