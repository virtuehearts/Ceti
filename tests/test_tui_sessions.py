import os
import tempfile
import unittest

from ceti import tui_sessions as ts


class TestSessionStore(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_round_trip(self):
        s = ts.new_session()
        s.lines = ["you> hello", "[local] hi"]
        self.assertTrue(ts.save_session(self.d, s))
        loaded = ts.load_session(self.d, s.id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.lines, s.lines)
        self.assertEqual(loaded.title, "hello")

    def test_empty_session_leaves_no_file(self):
        s = ts.new_session()
        s.lines = ["CETI // welcome line"]
        self.assertFalse(ts.save_session(self.d, s))
        self.assertEqual(ts.list_sessions(self.d), [])

    def test_title_from_first_question_truncated(self):
        s = ts.new_session()
        s.lines = ["you> " + "x" * 100]
        self.assertEqual(len(s.title), ts.TITLE_LEN)

    def test_list_newest_first(self):
        a = ts.new_session()
        a.lines = ["you> first"]
        ts.save_session(self.d, a)
        b = ts.new_session()
        b.lines = ["you> second"]
        ts.save_session(self.d, b)
        ids = [r["id"] for r in ts.list_sessions(self.d)]
        self.assertEqual(ids, [b.id, a.id])

    def test_resolve_prefix(self):
        s = ts.new_session()
        s.lines = ["you> hello"]
        ts.save_session(self.d, s)
        self.assertEqual(ts.resolve_id(self.d, s.id[:8]), s.id)
        self.assertEqual(ts.resolve_id(self.d, s.id), s.id)
        self.assertIsNone(ts.resolve_id(self.d, "zzz-nope"))
        # ambiguous prefix: two sessions share a date prefix
        s2 = ts.new_session()
        s2.lines = ["you> hi"]
        ts.save_session(self.d, s2)
        if s.id[:13] == s2.id[:13]:
            self.assertIsNone(ts.resolve_id(self.d, s.id[:13]))

    def test_delete(self):
        s = ts.new_session()
        s.lines = ["you> hello"]
        ts.save_session(self.d, s)
        self.assertTrue(ts.delete_session(self.d, s.id))
        self.assertIsNone(ts.load_session(self.d, s.id))
        self.assertFalse(ts.delete_session(self.d, s.id))

    def test_corrupt_file_skipped(self):
        bad = os.path.join(ts.session_dir(self.d), "junk.json")
        os.makedirs(ts.session_dir(self.d), exist_ok=True)
        with open(bad, "w") as f:
            f.write("{not json")
        self.assertEqual(ts.list_sessions(self.d), [])
        self.assertIsNone(ts.load_session(self.d, "junk"))


if __name__ == "__main__":
    unittest.main()
