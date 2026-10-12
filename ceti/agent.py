"""agent — the brain, with a name, tasks, and a loop.

The criticism was fair: ask/answer is a shell, not an agent. So now
the brain has:

  - a NAME (stored, changeable — it introduces itself by it)
  - TASKS with schedules: research, benchmark, curiosity, ingest,
    release — persisted in tasks.db, picked up by tick()
  - a LOOP: tick() runs the next due task; the REPL heartbeat calls
    it when you're quiet
  - CURIOSITY: questions it repeatedly fails to answer become gaps;
    gaps become research tasks. It finds what it thinks it needs —
    via the research stack (lynx-backed page fetching) — and the
    results land in cassettes and datasets.

This is the self-learning sentence: it exams itself (benchmark),
notices what it doesn't know (gaps), goes and learns it (curiosity),
and ships it (release).
"""
from __future__ import annotations

import os
import time

from . import benchmark
from .datasets import DatasetStore
from .merge import ingest_tarball
from .researcher import Researcher
from .tasks import TaskStore

DEFAULT_NAME = "CETI"


class Agent:
    def __init__(self, brain, name: str | None = None,
                 data_dir: str | None = None):
        self.brain = brain
        data_dir = os.path.abspath(
            data_dir or brain.cfg.data_dir)
        self.tasks = TaskStore(os.path.join(data_dir, "tasks.db"))
        if name:
            self.tasks.set_meta("name", name)
        elif not self.tasks.get_meta("name"):
            self.tasks.set_meta("name", DEFAULT_NAME)
            self.tasks.set_meta("born",
                                time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.store = DatasetStore(os.path.join(data_dir, "datasets"))

    @property
    def name(self) -> str:
        return self.tasks.get_meta("name") or DEFAULT_NAME

    def rename(self, name: str) -> None:
        self.tasks.set_meta("name", name)

    def status(self) -> dict:
        pending = self.tasks.pending()
        gaps = self.tasks.top_gaps(limit=5)
        return {"name": self.name,
                "pending_tasks": len(pending),
                "tasks": [(t["id"], t["kind"], t["title"])
                          for t in pending[:10]],
                "gaps": gaps,
                "cycles": int(self.tasks.get_meta("cycles") or 0)}

    # ---- the loop -------------------------------------------------------
    def tick(self) -> dict:
        """One agent step: turn top gaps into curiosity, then run the
        next due task. Returns what happened."""
        self._promote_gaps()
        due = self.tasks.due(limit=1)
        if not due:
            return {"idle": "no due tasks",
                    "gaps": len(self.tasks.top_gaps())}
        task = due[0]
        self.tasks.start(task["id"])
        try:
            result = self._execute(task)
        except Exception as e:
            self.tasks.fail(task["id"], "%s: %s"
                            % (type(e).__name__, e))
            return {"task": task["title"], "failed": str(e)[:200]}
        self.tasks.finish(task["id"], result,
                          interval_s=task["interval_s"])
        self.tasks.set_meta(
            "cycles", str(int(self.tasks.get_meta("cycles") or 0) + 1))
        return {"task": task["title"], "kind": task["kind"],
                "result": _short(result)}

    def _promote_gaps(self) -> None:
        for g in self.tasks.top_gaps(limit=3):
            self.tasks.add(
                "curiosity", "investigate: %s" % g["question"][:60],
                {"question": g["question"]})
            self.tasks.mark_tasked(g["question"])

    def _execute(self, task: dict) -> dict:
        kind, p = task["kind"], task["payload"]
        if kind == "research":
            return Researcher(self.brain).research_topic(p["topic"])
        if kind == "benchmark":
            return benchmark.run(self.brain, store=self.tasks)
        if kind == "curiosity":
            # a gap, investigated like any research topic: lynx-backed
            # fetching, distill, cassette, hot-reload
            return Researcher(self.brain).research_topic(p["question"])
        if kind == "ingest":
            return ingest_tarball(self.store, self.brain.memory,
                                  self.brain.aiml, p["tarball"])
        if kind == "release":
            return self._release(p.get("note", "agent release"))
        raise ValueError("unknown task kind: %s" % kind)

    def _release(self, note: str) -> dict:
        data_dir = os.path.abspath(self.brain.cfg.data_dir)
        files = [os.path.join(data_dir, "knowledge.sqlite3")]
        cas = os.path.join(data_dir, "cassettes")
        if os.path.isdir(cas):
            files.append(cas)
        files = [f for f in files if os.path.exists(f)]
        if not files:
            return {"skipped": "nothing to release"}
        manifest, _bundle, tarball = self.store.release(
            "ceti-knowledge", files, note=note)
        return {"version": manifest["version"],
                "tarball": os.path.basename(tarball),
                "merkle": manifest["merkle_root"][:16]}

    def close(self) -> None:
        self.tasks.close()


def _short(result: dict) -> dict:
    return {k: (str(v)[:160] if not isinstance(v, (int, float))
                else v)
            for k, v in list(result.items())[:8]}
