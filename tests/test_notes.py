import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ceti.notes import SessionNotes


class TestSessionNotes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ceti-notes-")
        self.path = os.path.join(self.tmp, "notes.json")
        self.notes = SessionNotes(self.path)

    def test_record_and_recall(self):
        i = self.notes.record("first finding")
        self.assertEqual(i, 0)
        self.notes.record("second finding")
        all_notes = self.notes.recall()
        self.assertEqual(len(all_notes), 2)
        self.assertEqual(all_notes[0]["content"], "first finding")
        self.assertIn("t", all_notes[0])

    def test_persistence(self):
        self.notes.record("survives")
        again = SessionNotes(self.path)
        self.assertEqual(again.count(), 1)
        self.assertEqual(again.recall()[0]["content"], "survives")

    def test_recall_limit(self):
        for i in range(5):
            self.notes.record("n%d" % i)
        self.assertEqual(len(self.notes.recall(limit=2)), 2)
        self.assertEqual(self.notes.recall(limit=2)[-1]["content"], "n4")

    def test_search(self):
        self.notes.record("the sandbox blocks network")
        self.notes.record("unrelated fact")
        hits = self.notes.search("sandbox")
        self.assertEqual(len(hits), 1)

    def test_empty_note_rejected(self):
        with self.assertRaises(ValueError):
            self.notes.record("   ")

    def test_format(self):
        self.assertIn("no notes", self.notes.format())
        self.notes.record("hello")
        self.assertIn("hello", self.notes.format())

    def test_corrupt_file_starts_empty(self):
        with open(self.path, "w") as f:
            f.write("not json{{{")
        self.assertEqual(SessionNotes(self.path).count(), 0)


if __name__ == "__main__":
    unittest.main()
