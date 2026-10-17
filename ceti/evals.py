"""Eval loop — coding / tool / knowledge benchmarks.

The brain already exams itself on its OWN knowledge
(ceti/benchmark.py: quiz every Q&A pair, accuracy + tier dist +
latency, misses -> gaps). This file is different: a FIXED, curated
suite that measures CETI the system — not the model — on the three
things it must be good at:

  knowledge  Brain.ask, expected facts present, tier != failed.
             Uses seed AIML answers (deterministic, no LLM) so the
             suite is stable run to run.
  coding     Brain.ask for code -> extract the code block -> run it
             in the sandbox (via the tool registry) against a test
             harness -> check the output. Code is verifiable: it
             runs or it doesn't. No vibes, no judge.
  tool       Drive the tool registry directly: write->read round
             trip, run_python, jail refusal. The tools either work
             or they don't. No model in the loop for these cases —
             they test the hands, not the mouth.

A miss in any suite is a gap: the runner files it to the gap log
(curiosity feed) so research can close it. Scores are the
"smarter" metric: if coding accuracy goes up after a change, CETI
got smarter. If it goes down, the change regressed something.

Design rules:
  - No network, no cloud, no new dependencies. Stdlib only.
  - Evals never learn (Brain.ask(learn=False)) — measuring must
    not pollute memory.
  - Verifiers are deterministic: substring checks + sandbox exit
    codes, never a model judging a model.
  - The runner takes injectable callables (brain, code_runner,
    tool_executor) so tests run with fakes and the REPL runs live.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n?```",
                         re.DOTALL | re.IGNORECASE)


def extract_code(answer: str) -> str | None:
    """First fenced Python block, else the whole answer if it
    defines a function, else None (no code to run)."""
    m = _CODE_FENCE.search(answer or "")
    if m:
        return m.group(1).strip() or None
    if answer and re.search(r"^\s*def\s+\w+\s*\(", answer, re.MULTILINE):
        return answer.strip()
    return None


@dataclass
class EvalCase:
    id: str
    suite: str                     # knowledge | coding | tool
    prompt: str                    # question / task (gap-logged on miss)
    kind: str                      # contains | code_runs | tool_seq
    expected: list[str] = field(default_factory=list)
    harness: str = ""              # coding: test code appended to answer
    tool_seq: str = ""             # tool: which scripted sequence to run


@dataclass
class EvalResult:
    id: str
    suite: str
    passed: bool
    latency_ms: int
    detail: str = ""


# ---------------------------------------------------------------------------
# Built-in suite — 3 per domain, 9 total. Small on purpose: /evals
# should finish in ~a minute live (knowledge is AIML-instant, coding
# is 3 local-model calls, tool is registry-only).
# ---------------------------------------------------------------------------

def default_cases() -> list[EvalCase]:
    return [
        # knowledge — seed AIML, deterministic, no LLM needed
        EvalCase("know_usenet", "knowledge", "What is Usenet?",
                 "contains", ["distributed", "discussion"]),
        EvalCase("know_modem", "knowledge", "What is a modem?",
                 "contains", ["modem", "digital"]),
        EvalCase("know_identity", "knowledge", "Who are you?",
                 "contains", ["jero"]),
        # coding — Brain.ask -> extract -> sandbox -> harness output
        EvalCase("code_add", "coding",
                 "Write a Python function add(a, b) that returns a + b.",
                 "code_runs", ["5"], harness="print(add(2, 3))"),
        EvalCase("code_reverse", "coding",
                 "Write a Python function reverse_string(s) that "
                 "returns s reversed.",
                 "code_runs", ["cba"],
                 harness='print(reverse_string("abc"))'),
        EvalCase("code_even", "coding",
                 "Write a Python function is_even(n) that returns True "
                 "if n is even, else False.",
                 "code_runs", ["True False"],
                 harness="print(is_even(4), is_even(3))"),
        # tool — registry directly, no model in the loop
        EvalCase("tool_write_read", "tool",
                 "write_file then read_file round trip",
                 "tool_seq", ["hello evals"], tool_seq="write_read"),
        EvalCase("tool_run_python", "tool",
                 "run_python computes 40 + 2",
                 "tool_seq", ["42", "[exit 0]"], tool_seq="run_python"),
        EvalCase("tool_jail", "tool",
                 "read_file must refuse a path escape",
                 "tool_seq", ["refused"], tool_seq="jail"),
    ]


class EvalRunner:
    """Run the suite against a Brain + tool executors."""

    def __init__(self, brain, code_runner=None, tool_executor=None,
                 gap_log=None, cases: list[EvalCase] | None = None):
        self.brain = brain
        # code_runner(code: str) -> str (sandbox output incl. exit line).
        # Default: the real tool registry's run_python.
        self.code_runner = code_runner or self._default_code_runner
        # tool_executor(name, **kwargs) -> ToolResult-like with
        # .success / .content / .error. Default: tools.execute.
        self.tool_executor = tool_executor or self._default_tool_executor
        self.gap_log = gap_log
        self.cases = list(cases) if cases is not None else default_cases()

    @staticmethod
    def _default_code_runner(code: str) -> str:
        from . import tools as _tools
        res = _tools.execute("run_python", code=code)
        return res.content if res.success else (res.error or "tool failed")

    @staticmethod
    def _default_tool_executor(name: str, **kwargs):
        from . import tools as _tools
        return _tools.execute(name, **kwargs)

    # ---- per-kind verifiers ----------------------------------------------
    def _run_contains(self, case: EvalCase) -> EvalResult:
        t0 = time.monotonic()
        try:
            ans = self.brain.ask(case.prompt, learn=False)
        except Exception as e:
            return EvalResult(case.id, case.suite, False,
                              int((time.monotonic() - t0) * 1000),
                              "ask raised: %s" % e)
        ms = int((time.monotonic() - t0) * 1000)
        if ans.tier == "failed":
            return EvalResult(case.id, case.suite, False, ms,
                              "tier=failed")
        hay = ans.text.lower()
        missing = [e for e in case.expected if e.lower() not in hay]
        if missing:
            return EvalResult(case.id, case.suite, False, ms,
                              "missing: %s (tier=%s)"
                              % (", ".join(missing), ans.tier))
        return EvalResult(case.id, case.suite, True, ms,
                          "tier=%s" % ans.tier)

    def _run_code(self, case: EvalCase) -> EvalResult:
        t0 = time.monotonic()
        try:
            ans = self.brain.ask(case.prompt, learn=False)
        except Exception as e:
            return EvalResult(case.id, case.suite, False,
                              int((time.monotonic() - t0) * 1000),
                              "ask raised: %s" % e)
        code = extract_code(ans.text)
        if not code:
            return EvalResult(case.id, case.suite, False,
                              int((time.monotonic() - t0) * 1000),
                              "no code block in answer (tier=%s)"
                              % ans.tier)
        try:
            out = self.code_runner(code + "\n" + case.harness)
        except Exception as e:
            return EvalResult(case.id, case.suite, False,
                              int((time.monotonic() - t0) * 1000),
                              "code runner raised: %s" % e)
        ms = int((time.monotonic() - t0) * 1000)
        if "[exit 0]" not in out:
            return EvalResult(case.id, case.suite, False, ms,
                              "non-zero exit: %s" % out[-300:])
        missing = [e for e in case.expected if e not in out]
        if missing:
            return EvalResult(case.id, case.suite, False, ms,
                              "output missing %s; got: %s"
                              % (", ".join(missing), out[-300:]))
        return EvalResult(case.id, case.suite, True, ms,
                          "code ran, harness output matched")

    def _run_tool(self, case: EvalCase) -> EvalResult:
        t0 = time.monotonic()
        try:
            if case.tool_seq == "write_read":
                w = self.tool_executor(
                    "write_file", path="evals-probe.txt",
                    content="hello evals")
                if not w.success:
                    return EvalResult(case.id, case.suite, False,
                                      int((time.monotonic() - t0) * 1000),
                                      "write failed: %s" % w.error)
                r = self.tool_executor("read_file",
                                       path="evals-probe.txt")
                ok = r.success and "hello evals" in (r.content or "")
                return EvalResult(case.id, case.suite, ok,
                                  int((time.monotonic() - t0) * 1000),
                                  "round trip ok" if ok
                                  else "read mismatch: %s" % (r.error or r.content or ""))
            if case.tool_seq == "run_python":
                r = self.tool_executor("run_python", code="print(40 + 2)")
                out = (r.content or "") if r.success else (r.error or "")
                ok = r.success and all(e in out for e in case.expected)
                return EvalResult(case.id, case.suite, ok,
                                  int((time.monotonic() - t0) * 1000),
                                  out[-200:] if ok else "got: %s" % out[-200:])
            if case.tool_seq == "jail":
                r = self.tool_executor("read_file", path="../escape")
                ok = (not r.success) and ("refused" in (r.error or "").lower()
                                           or "escape" in (r.error or "").lower())
                return EvalResult(case.id, case.suite, ok,
                                  int((time.monotonic() - t0) * 1000),
                                  "jail refused" if ok
                                  else "jail did NOT refuse: %s" % (r.content or r.error or ""))
            return EvalResult(case.id, case.suite, False,
                              int((time.monotonic() - t0) * 1000),
                              "unknown tool_seq: %r" % case.tool_seq)
        except Exception as e:
            return EvalResult(case.id, case.suite, False,
                              int((time.monotonic() - t0) * 1000),
                              "tool raised: %s" % e)

    # ---- driver ------------------------------------------------------------
    def run_one(self, case: EvalCase) -> EvalResult:
        if case.kind == "contains":
            res = self._run_contains(case)
        elif case.kind == "code_runs":
            res = self._run_code(case)
        elif case.kind == "tool_seq":
            res = self._run_tool(case)
        else:
            res = EvalResult(case.id, case.suite, False, 0,
                             "unknown kind: %r" % case.kind)
        if not res.passed and self.gap_log is not None:
            try:
                self.gap_log.note_gap(case.prompt)
            except Exception:
                pass
        return res

    def run(self, suite: str | None = None) -> dict:
        """Run the suite (or one suite). Returns the report dict."""
        t0 = time.monotonic()
        cases = [c for c in self.cases
                 if suite in (None, "all", c.suite)]
        results = [self.run_one(c) for c in cases]
        by_suite: dict[str, dict] = {}
        for r in results:
            d = by_suite.setdefault(r.suite, {"total": 0, "passed": 0})
            d["total"] += 1
            d["passed"] += 1 if r.passed else 0
        for d in by_suite.values():
            d["accuracy"] = round(d["passed"] / d["total"], 3) \
                if d["total"] else 0.0
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        return {
            "suites": by_suite,
            "total": total, "passed": passed,
            "accuracy": round(passed / total, 3) if total else 0.0,
            "results": [{"id": r.id, "suite": r.suite,
                         "passed": r.passed, "ms": r.latency_ms,
                         "detail": r.detail} for r in results],
            "misses": [{"id": r.id, "suite": r.suite,
                        "detail": r.detail} for r in results
                       if not r.passed],
            "elapsed_s": round(time.monotonic() - t0, 1),
        }
