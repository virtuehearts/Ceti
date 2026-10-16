"""researcher — the idle-cycle pre-trainer.

Wikipedia-style, but local: picks the next uncovered topic, researches
it with the sovereign research stack (local search engines, local
LFM2.5 synthesis), distills the report into Q&A pairs with the local
LLM, classifies each pair, and records the promoted ones onto a
cassette (data/cassettes/<slug>.aiml) plus the vector knowledge base.

Hybrid by design: research and distillation run on the box; the cloud
is only a fallback for distillation if the local LLM is down. Nothing
personal is ever involved — and pairs are PII-stripped before they are
recorded, same rule as live learning.

Idle gating: a cycle only starts when the box is actually idle
(load average and available memory checked first). Wire it to a cron
or the sovereign watchdog for true idle-cycle learning.

Usage:
    python -m brain.researcher --once [--topic "dial-up modems"]
    python -m brain.researcher --topics
    python -m brain.researcher --status
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import time

from . import pii
from .aiml import AimlBrain
from .cassettes import CassetteRack, slugify
from .research_config import ResearchConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESEARCH_PY = os.path.expanduser(
    "~/workspace/sovereign/research/research.py")
DEFAULT_RESEARCH_PYTHON = os.path.expanduser(
    "~/workspace/venvs/memvec/bin/python")

DISTILL_PROMPT = """Extract up to 8 self-contained question/answer pairs \
from the report below.
Rules:
- each answer 1-3 sentences, factual and timeless (no "today",
  no "recently", no dates that will rot)
- questions a curious person would actually ask
- no markdown, no personal data, plain text only
Reply with ONLY a JSON list, no other text:
[{"q": "...", "a": "..."}, ...]

REPORT:
%s
"""


def mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    # non-Linux (no /proc): don't gate on a reading we can't take
    return 999.0


class Researcher:
    def __init__(self, brain,
                 research_py: str | None = None,
                 research_python: str | None = None,
                 topics_file: str | None = None,
                 pages: int = 5,
                 min_mem_gb: float = 1.0,
                 max_load: float = 2.0,
                 timeout_s: int = 1800):
        self.brain = brain
        self.research_py = (
            research_py
            or os.environ.get("CETI_RESEARCH_PY")
            or DEFAULT_RESEARCH_PY)
        self.research_python = (
            research_python
            or os.environ.get("CETI_RESEARCH_PYTHON")
            or DEFAULT_RESEARCH_PYTHON)
        self.pages = pages
        self.min_mem_gb = min_mem_gb
        self.max_load = max_load
        self.timeout_s = timeout_s
        data_dir = os.path.abspath(brain.cfg.data_dir)
        self.topics_file = topics_file or os.path.join(data_dir,
                                                       "topics.txt")
        self.config = ResearchConfig(data_dir)
        self.rack = CassetteRack(os.path.join(data_dir, "cassettes"))
        self.log_db = os.path.join(data_dir, "researcher_log.sqlite3")
        self._log_init()

    # ---- bookkeeping --------------------------------------------------
    def _log_init(self) -> None:
        db = sqlite3.connect(self.log_db)
        db.execute("CREATE TABLE IF NOT EXISTS log ("
                   " topic TEXT PRIMARY KEY, started REAL,"
                   " finished REAL, qa_added INTEGER, report TEXT)")
        db.commit()
        db.close()

    def _log(self, topic: str, qa_added: int = 0,
             report: str | None = None, start: bool = False) -> None:
        db = sqlite3.connect(self.log_db)
        if start:
            db.execute("INSERT OR REPLACE INTO log (topic, started,"
                       " finished, qa_added, report)"
                       " VALUES (?, ?, NULL, 0, NULL)",
                       (topic, time.time()))
        else:
            db.execute("UPDATE log SET finished=?, qa_added=?, report=?"
                       " WHERE topic=?",
                       (time.time(), qa_added, report, topic))
        db.commit()
        db.close()

    def status(self) -> dict:
        db = sqlite3.connect(self.log_db)
        rows = list(db.execute(
            "SELECT topic, finished, qa_added FROM log"))
        db.close()
        return {"topics_file": self.topics_file,
                "cassettes": self.rack.list(),
                "completed": [t for t, f, _ in rows if f],
                "qa_total": sum(q for _, _, q in rows),
                "sites": list(self.config.sites),
                "interests": list(self.config.interests),
                "nightly_budget": dict(self.config.nightly)}

    def add_topic(self, text: str) -> str:
        """Append a topic to topics.txt (dedupe, keep comments)."""
        topic = " ".join((text or "").split())
        if not topic:
            raise ValueError("empty topic")
        if topic in self.topics():
            return topic
        with open(self.topics_file, "a", encoding="utf-8") as f:
            f.write(topic + "\n")
        return topic

    def remove_topic(self, text: str) -> str:
        topic = " ".join((text or "").split())
        if topic not in self.topics():
            raise ValueError("not in topics: %s" % topic)
        kept = []
        if os.path.exists(self.topics_file):
            with open(self.topics_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip() != topic:
                        kept.append(line)
        with open(self.topics_file, "w", encoding="utf-8") as f:
            f.writelines(kept)
        return topic

    def topics(self) -> list[str]:
        out = []
        if not os.path.exists(self.topics_file):
            return out
        with open(self.topics_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    out.append(line)
        return out

    def next_topic(self, exclude: set | None = None) -> str | None:
        done = {r[0] for r in sqlite3.connect(self.log_db).execute(
            "SELECT topic FROM log WHERE finished IS NOT NULL")}
        skip = exclude or set()
        for t in self.topics():
            if t not in done and t not in skip:
                return t
        # queue dry: cycle the evergreen interests, least-recent first.
        # Interests never "complete" — they are areas, not assignments.
        # exclude lets one run (e.g. a night) skip interests it already
        # covered, so a single interest isn't researched 5x in a row.
        cands = [i for i in self.config.interests
                 if i not in (exclude or set())]
        if cands:
            rows = dict(sqlite3.connect(self.log_db).execute(
                "SELECT topic, finished FROM log"))
            return min(cands, key=lambda i: rows.get(i) or 0)
        return None

    # ---- idle gating ---------------------------------------------------
    def idle_ok(self) -> tuple[bool, str]:
        try:
            load1 = os.getloadavg()[0]
        except OSError:
            load1 = 0.0
        mem = mem_available_gb()
        if load1 > self.max_load:
            return False, "load %.1f > %.1f" % (load1, self.max_load)
        if mem < self.min_mem_gb:
            return False, "mem %.1fGB < %.1fGB" % (mem, self.min_mem_gb)
        return True, "idle (load %.1f, mem %.1fGB)" % (load1, mem)

    # ---- the cycle ------------------------------------------------------
    def idle_cycle(self) -> dict:
        ok, reason = self.idle_ok()
        if not ok:
            return {"skipped": reason}
        topic = self.next_topic()
        if not topic:
            return {"done": "all topics covered",
                    "cassettes": self.rack.list()}
        return self.research_topic(topic)

    def run_nightly(self, deadline: float | None = None) -> dict:
        """Research until the nightly budget is spent, the deadline
        passes, or the queue (topics + interests) runs dry. The budget
        is the config's nightly cap: max topics, max Q&A pairs, max
        minutes. Interests keep the night going after the queue drains
        — they are evergreen, so a dry queue alone never ends a night
        early unless the budget says so."""
        budget = self.config.nightly
        t0 = time.time()
        spent_minutes = lambda: (time.time() - t0) / 60.0
        done: list[dict] = []
        qa_total = 0
        seen: set[str] = set()
        while True:
            if len(done) >= budget["max_topics"]:
                reason = "max_topics %d reached" % budget["max_topics"]
                break
            if qa_total >= budget["max_qa"]:
                reason = "max_qa %d reached" % budget["max_qa"]
                break
            if spent_minutes() >= budget["max_minutes"]:
                reason = "max_minutes %d reached" % budget["max_minutes"]
                break
            if deadline is not None and time.time() >= deadline:
                reason = "deadline reached"
                break
            topic = self.next_topic(exclude=seen)
            if not topic:
                reason = ("interests covered tonight"
                          if self.config.interests
                          else "queue dry (no topics, no interests)")
                break
            seen.add(topic)
            res = self.research_topic(topic)
            if isinstance(res, dict) and res.get("failed"):
                done.append({"topic": topic, "failed": res["failed"]})
                continue
            qa_total += res.get("qa_added", 0)
            done.append({"topic": topic,
                         "qa_added": res.get("qa_added", 0)})
        return {"nightly": {"topics_done": len(done),
                            "qa_added": qa_total,
                            "elapsed_minutes": round(spent_minutes(), 1),
                            "stopped": reason},
                "topics": done}

    def research_topic(self, topic: str) -> dict:
        t0 = time.time()
        self._log(topic, start=True)
        report_path = self._run_research(topic)
        if not report_path:
            # research failed: clear the start marker so the topic is
            # retried next cycle instead of being skipped forever
            db = sqlite3.connect(self.log_db)
            db.execute("DELETE FROM log WHERE topic=?", (topic,))
            db.commit()
            db.close()
            return {"topic": topic, "failed": "research produced no report",
                    "elapsed_s": int(time.time() - t0)}
        pairs = self._distill(report_path, topic)
        qa_added = self._record(topic, pairs)
        self._log(topic, qa_added=qa_added, report=report_path)
        if qa_added:
            # hot-swap the new cassette into the live brain
            try:
                self.brain.reload_cassettes()
            except Exception:
                pass
        return {"topic": topic, "report": report_path,
                "pairs_distilled": len(pairs), "qa_added": qa_added,
                "cassette": self.rack.path(slugify(topic)),
                "elapsed_s": int(time.time() - t0)}

    def _research_cmd(self, topic: str) -> list[str]:
        """The stack invocation. The allowlist rides along on every
        run: Wikipedia-only for the default config (cheap fast path),
        --allow-sites otherwise. CETI_WIKI_ONLY=1 still forces the
        wiki fast path for anyone who set it by hand."""
        cmd = [self.research_python, self.research_py, topic,
               "--pages", str(self.pages),
               "--min-mem", str(self.min_mem_gb)]
        if os.environ.get("CETI_WIKI_ONLY") == "1" \
                or self.config.wiki_only():
            cmd.append("--wiki-only")
        else:
            cmd.append("--allow-sites")
            cmd.append(",".join(self.config.sites))
        return cmd

    def _run_research(self, topic: str) -> str | None:
        """Run the research stack; return the report path."""
        if not os.path.exists(self.research_py):
            print("research skipped: no research stack at %s "
                  "(set CETI_RESEARCH_PY)" % self.research_py, flush=True)
            return None
        cmd = self._research_cmd(topic)
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=self.timeout_s)
        except (subprocess.TimeoutExpired, OSError) as e:
            print("research failed: %s" % e, flush=True)
            return None
        if p.returncode != 0:
            print("research failed: %s" % p.stderr[-500:], flush=True)
            return None
        m = re.search(r"^report: (\S+)", p.stdout, re.MULTILINE)
        return m.group(1) if m else None

    def _llm(self, prompt: str) -> str:
        """Local LLM first (the brain's Qwen), cloud fallback."""
        if self.brain.local:
            return self.brain.local.complete(
                "Reply with ONLY the requested JSON list.",
                prompt, max_tokens=1500, temperature=0.2)
        for backend in self.brain.clouds:
            return backend.complete(
                "Reply with ONLY the requested JSON list.",
                prompt, max_tokens=1500, temperature=0.2)
        raise RuntimeError("no LLM backend available for distillation")

    def _distill(self, report_path: str, topic: str) -> list[dict]:
        try:
            with open(report_path, encoding="utf-8",
                      errors="replace") as f:
                report = f.read()
        except OSError:
            return []
        # keep the prompt small: first ~6000 chars carry the brief
        raw = self._llm(DISTILL_PROMPT % report[:6000])
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            return []
        try:
            pairs = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        out = []
        for p in pairs if isinstance(pairs, list) else []:
            q = str(p.get("q", "")).strip()
            a = str(p.get("a", "")).strip()
            if q and a and len(q) <= 300:
                out.append({"q": q, "a": a})
        return out[:10]

    def _record(self, topic: str, pairs: list[dict]) -> int:
        """Classify each pair; record the promoted ones on the cassette
        and in the vector knowledge base. Returns count recorded."""
        cassette_path = self.rack.create(topic)
        writer = AimlBrain(learned_path=cassette_path)
        added = 0
        for p in pairs:
            q, _ = pii.strip(p["q"])
            a, _ = pii.strip(p["a"])
            verdict = self.brain.classifier.decide(q, a)
            if verdict.action == "promote":
                do = True
            elif verdict.action == "adjudicate" \
                    and self.brain.cfg.adjudicate_with_llm \
                    and self.brain.local:
                do = self.brain.classifier.adjudicate(
                    q, a, lambda pr: self.brain.local.complete(
                        "Reply with exactly YES or NO.", pr,
                        max_tokens=8, temperature=0.0))
            else:
                do = False
            if do and writer.learn(q, a):
                if self.brain.memory:
                    self.brain.memory.add(q, a, verdict.score)
                    self.brain.memory.learn_aliases_from(a)
                added += 1
        return added


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="run one idle cycle")
    ap.add_argument("--topic",
                    help="research one topic now (ignores idle gate)")
    ap.add_argument("--topics", action="store_true",
                    help="list topics")
    ap.add_argument("--status", action="store_true",
                    help="show researcher status")
    args = ap.parse_args()

    # late import so --help stays fast
    from .brain import Brain
    brain = Brain()
    r = Researcher(brain)
    try:
        if args.topics:
            for t in r.topics():
                print(t)
        elif args.status:
            print(json.dumps(r.status(), indent=1))
        elif args.topic:
            print(json.dumps(r.research_topic(args.topic), indent=1))
        elif args.once:
            print(json.dumps(r.idle_cycle(), indent=1))
        else:
            ap.print_help()
    finally:
        brain.close()


if __name__ == "__main__":
    main()
