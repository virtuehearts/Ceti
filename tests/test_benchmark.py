import os
import tempfile
import unittest

from ceti import benchmark
from ceti.brain import Brain
from ceti.config import BrainConfig
from ceti.tasks import TaskStore


def make_brain():
    d = tempfile.mkdtemp()
    cfg = BrainConfig(data_dir=d, memory_backend="sqlite",
                      enable_aiml=False, enable_local=False,
                      enable_cloud=False)
    b = Brain(config=cfg)
    b.memory.add("What is a BBS?",
                 "A BBS is a dial-up message board system.", 0.9)
    return b, d


class TestBenchmark(unittest.TestCase):
    def test_helpers(self):
        self.assertTrue(len(benchmark.key_terms(
            "A BBS is a dial-up message board system.")) > 0)
        t = benchmark.typo_variant("hello world")
        self.assertNotEqual(t, "hello world")
        self.assertEqual(len(benchmark.variants("Q?")), 3)

    def test_run_records(self):
        brain, d = make_brain()
        store = TaskStore(os.path.join(d, "tasks.db"))
        try:
            summary = benchmark.run(brain, store=store)
            self.assertEqual(summary["questions"], 1)
            self.assertEqual(summary["probes"], 3)
            self.assertIn("accuracy", summary)
            self.assertIn("tier_dist", summary)
            rows = store.db.execute(
                "SELECT COUNT(*) FROM bench_runs").fetchone()[0]
            self.assertEqual(rows, 1)
        finally:
            brain.close()
            store.close()

    def test_run_without_store(self):
        brain, d = make_brain()
        try:
            summary = benchmark.run(brain)
            self.assertEqual(summary["probes"], 3)
        finally:
            brain.close()


if __name__ == "__main__":
    unittest.main()
