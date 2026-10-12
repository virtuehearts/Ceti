import os
import tempfile
import time
import unittest

from ceti.tasks import TaskStore


class TestTasks(unittest.TestCase):
    def setUp(self):
        self.store = TaskStore(
            os.path.join(tempfile.mkdtemp(), "tasks.db"))

    def test_add_and_due(self):
        tid = self.store.add("research", "study BBS",
                             {"topic": "bbs"}, delay_s=100)
        self.assertEqual(self.store.due(), [])  # not due yet
        tid2 = self.store.add("benchmark", "self-exam")
        due = self.store.due()
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0]["id"], tid2)
        self.assertEqual(due[0]["payload"], {})

    def test_lifecycle(self):
        tid = self.store.add("research", "study BBS", {"topic": "bbs"})
        self.store.start(tid)
        self.assertEqual(self.store.due(), [])
        self.store.finish(tid, {"ok": True})
        self.assertEqual(self.store.pending(), [])

    def test_recurring_reschedules(self):
        tid = self.store.add("benchmark", "weekly", {}, interval_s=1000)
        self.store.start(tid)
        self.store.finish(tid, {"accuracy": 1.0}, interval_s=1000)
        pending = self.store.pending()
        self.assertEqual(len(pending), 1)
        self.assertGreater(pending[0]["run_after"], time.time() + 900)

    def test_fail(self):
        tid = self.store.add("research", "x")
        self.store.fail(tid, "boom")
        self.assertEqual(self.store.pending(), [])
        self.assertEqual(self.store.due(), [])

    def test_bad_kind_rejected(self):
        with self.assertRaises(ValueError):
            self.store.add("nap", "sleep")

    def test_gaps(self):
        self.store.note_gap("what is a modem")
        self.store.note_gap("what is a modem")
        self.store.note_gap("what is x")
        top = self.store.top_gaps()
        self.assertEqual(len(top), 1)  # min_count=2
        self.assertEqual(top[0]["question"], "what is a modem")
        self.assertEqual(top[0]["count"], 2)
        self.store.mark_tasked("what is a modem")
        self.assertEqual(self.store.top_gaps(), [])

    def test_meta(self):
        self.assertIsNone(self.store.get_meta("name"))
        self.store.set_meta("name", "GENESIS")
        self.assertEqual(self.store.get_meta("name"), "GENESIS")


if __name__ == "__main__":
    unittest.main()
