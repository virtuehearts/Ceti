import os
import tempfile
import unittest

from ceti.agent import Agent
from ceti.brain import Brain
from ceti.config import BrainConfig


def make_brain(d):
    cfg = BrainConfig(data_dir=d, memory_backend="sqlite",
                      enable_aiml=False, enable_local=False,
                      enable_cloud=False)
    return Brain(config=cfg)


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.brain = make_brain(self.d)
        self.agent = Agent(self.brain, name="TESTAGENT")

    def tearDown(self):
        self.agent.close()
        self.brain.close()

    def test_name(self):
        self.assertEqual(self.agent.name, "TESTAGENT")
        self.agent.rename("OTHER")
        self.assertEqual(self.agent.name, "OTHER")

    def test_tick_idle(self):
        out = self.agent.tick()
        self.assertIn("idle", out)

    def test_benchmark_task_executes(self):
        self.brain.memory.add("What is a BBS?",
                              "A BBS is a dial-up message board.", 0.9)
        self.brain.gap_log = self.agent.tasks
        self.agent.tasks.add("benchmark", "self-exam")
        out = self.agent.tick()
        self.assertEqual(out["kind"], "benchmark")
        self.assertIn("accuracy", out["result"])
        runs = self.agent.tasks.db.execute(
            "SELECT COUNT(*) FROM bench_runs").fetchone()[0]
        self.assertEqual(runs, 1)

    def test_gaps_become_curiosity(self):
        self.brain.gap_log = self.agent.tasks
        for _ in range(2):
            r = self.brain.ask("qzx unknown thing")
            self.assertEqual(r.tier, "failed")
        # promote only: don't execute (that would run real research)
        self.agent._promote_gaps()
        pending = self.agent.tasks.pending()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["kind"], "curiosity")
        self.assertIn("qzx", pending[0]["title"])

    def test_status(self):
        st = self.agent.status()
        self.assertEqual(st["name"], "TESTAGENT")
        self.assertIn("pending_tasks", st)
        self.assertIn("gaps", st)


if __name__ == "__main__":
    unittest.main()
