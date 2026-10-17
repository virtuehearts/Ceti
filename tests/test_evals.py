"""Tests for the eval loop (ceti/evals.py). No network, no LLM —
fake Brain, fake code runner, fake tool executor."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ceti.evals import (EvalCase, EvalRunner, default_cases, extract_code)


class FakeAnswer:
    def __init__(self, text, tier="aiml"):
        self.text = text
        self.tier = tier


class FakeBrain:
    def __init__(self, answers=None):
        # answers: {prompt: (text, tier)}
        self.answers = answers or {}
        self.asked = []

    def ask(self, prompt, learn=True):
        self.asked.append((prompt, learn))
        text, tier = self.answers.get(prompt, ("", "failed"))
        return FakeAnswer(text, tier)


class FakeToolResult:
    def __init__(self, success, content="", error=None):
        self.success = success
        self.content = content
        self.error = error


def fake_code_runner(code):
    # Minimal harness runner for tests: exec the code, capture prints.
    import io
    import contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {})
        return buf.getvalue() + "[exit 0]"
    except Exception as e:
        return "error: %s\n[exit 1]" % e


class FakeTools:
    def __init__(self):
        self.files = {}

    def __call__(self, name, **kwargs):
        if name == "write_file":
            self.files[kwargs["path"]] = kwargs["content"]
            return FakeToolResult(True, "wrote")
        if name == "read_file":
            p = kwargs["path"]
            if p.startswith(".."):
                return FakeToolResult(False, "", "refused: escapes the work root")
            if p not in self.files:
                return FakeToolResult(False, "", "no such file")
            return FakeToolResult(True, self.files[p])
        if name == "run_python":
            return FakeToolResult(True, fake_code_runner(kwargs["code"]))
        return FakeToolResult(False, "", "no such tool")


class FakeGapLog:
    def __init__(self):
        self.gaps = []

    def note_gap(self, q):
        self.gaps.append(q)


class TestExtractCode(unittest.TestCase):
    def test_fenced(self):
        self.assertIn("def add", extract_code(
            "Here:\n```python\ndef add(a, b):\n    return a + b\n```"))
        self.assertEqual(extract_code("no code here"), None)
        self.assertIn("def f", extract_code("def f():\n    pass"))


class TestEvalRunner(unittest.TestCase):
    def _brain(self):
        return FakeBrain({
            "What is Usenet?": ("USENET is a distributed discussion system.", "aiml"),
            "What is a modem?": ("A modem converts digital signals.", "aiml"),
            "Who are you?": ("I am Jero's intellect.", "aiml"),
            "Write a Python function add(a, b) that returns a + b.": (
                "```python\ndef add(a, b):\n    return a + b\n```", "local"),
            "Write a Python function reverse_string(s) that returns s reversed.": (
                "```python\ndef reverse_string(s):\n    return s[::-1]\n```", "local"),
            "Write a Python function is_even(n) that returns True if n is even, else False.": (
                "```python\ndef is_even(n):\n    return n % 2 == 0\n```", "local"),
        })

    def test_full_suite_green(self):
        tools = FakeTools()
        runner = EvalRunner(self._brain(), code_runner=fake_code_runner,
                            tool_executor=tools)
        rep = runner.run()
        self.assertEqual(rep["total"], 9)
        self.assertEqual(rep["passed"], 9)
        self.assertEqual(rep["accuracy"], 1.0)
        self.assertEqual(rep["suites"]["knowledge"]["accuracy"], 1.0)
        self.assertEqual(rep["suites"]["coding"]["accuracy"], 1.0)
        self.assertEqual(rep["suites"]["tool"]["accuracy"], 1.0)

    def test_knowledge_miss(self):
        brain = FakeBrain({"What is Usenet?": ("Something else.", "aiml")})
        runner = EvalRunner(brain, code_runner=fake_code_runner,
                            tool_executor=FakeTools())
        case = EvalCase("k1", "knowledge", "What is Usenet?",
                        "contains", ["distributed"])
        res = runner.run_one(case)
        self.assertFalse(res.passed)
        self.assertIn("missing", res.detail)

    def test_coding_wrong_output_fails(self):
        brain = FakeBrain({
            "Write a Python function add(a, b) that returns a + b.": (
                "```python\ndef add(a, b):\n    return a - b\n```", "local")})
        runner = EvalRunner(brain, code_runner=fake_code_runner,
                            tool_executor=FakeTools())
        case = EvalCase("c1", "coding", "Write a Python function add(a, b) that returns a + b.",
                        "code_runs", ["5"], harness="print(add(2, 3))")
        res = runner.run_one(case)
        self.assertFalse(res.passed)
        self.assertIn("output missing", res.detail)

    def test_coding_no_code_fails(self):
        brain = FakeBrain({
            "Write a Python function add(a, b) that returns a + b.": (
                "I don't know how to code.", "local")})
        runner = EvalRunner(brain, code_runner=fake_code_runner,
                            tool_executor=FakeTools())
        case = EvalCase("c1", "coding", "Write a Python function add(a, b) that returns a + b.",
                        "code_runs", ["5"], harness="print(add(2, 3))")
        res = runner.run_one(case)
        self.assertFalse(res.passed)
        self.assertIn("no code block", res.detail)

    def test_coding_crash_fails(self):
        brain = FakeBrain({
            "Write a Python function add(a, b) that returns a + b.": (
                "```python\ndef add(a, b):\n    raise RuntimeError('boom')\n```", "local")})
        runner = EvalRunner(brain, code_runner=fake_code_runner,
                            tool_executor=FakeTools())
        case = EvalCase("c1", "coding", "Write a Python function add(a, b) that returns a + b.",
                        "code_runs", ["5"], harness="print(add(2, 3))")
        res = runner.run_one(case)
        self.assertFalse(res.passed)
        self.assertIn("non-zero exit", res.detail)

    def test_tool_jail_must_refuse(self):
        class PermissiveTools(FakeTools):
            def __call__(self, name, **kwargs):
                if name == "read_file" and kwargs["path"].startswith(".."):
                    return FakeToolResult(True, "SECRET")
                return super().__call__(name, **kwargs)

        runner = EvalRunner(FakeBrain(), code_runner=fake_code_runner,
                            tool_executor=PermissiveTools())
        case = EvalCase("t1", "tool", "jail", "tool_seq", ["refused"],
                        tool_seq="jail")
        self.assertFalse(runner.run_one(case).passed)

    def test_suite_filter(self):
        runner = EvalRunner(self._brain(), code_runner=fake_code_runner,
                            tool_executor=FakeTools())
        rep = runner.run("tool")
        self.assertEqual(rep["total"], 3)
        self.assertEqual(set(rep["suites"]), {"tool"})

    def test_misses_become_gaps(self):
        gaps = FakeGapLog()
        brain = FakeBrain({})
        runner = EvalRunner(brain, code_runner=fake_code_runner,
                            tool_executor=FakeTools(), gap_log=gaps)
        case = EvalCase("k1", "knowledge", "What is Usenet?",
                        "contains", ["distributed"])
        runner.run_one(case)
        self.assertIn("What is Usenet?", gaps.gaps)

    def test_evals_never_learn(self):
        brain = self._brain()
        runner = EvalRunner(brain, code_runner=fake_code_runner,
                            tool_executor=FakeTools())
        runner.run()
        # every ask passed learn=False — measuring must not pollute memory
        self.assertTrue(brain.asked)
        self.assertTrue(all(learn is False for _, learn in brain.asked))

    def test_default_cases_shape(self):
        cases = default_cases()
        self.assertEqual(len(cases), 9)
        self.assertEqual({c.suite for c in cases},
                         {"knowledge", "coding", "tool"})


if __name__ == "__main__":
    unittest.main()
