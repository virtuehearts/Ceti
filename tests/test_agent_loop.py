"""Agent loop tests. The oracle is faked; tools run for real."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ceti import tools
from ceti.agent_loop import (AgentError, AgentSession, estimate_tokens,
                             run_task)
from ceti.config import BrainConfig


class FakeBackend:
    """Scripted oracle: pops one canned response per call."""
    name = "fake-oracle"

    def __init__(self, script, api_key="test-key"):
        self.script = list(script)
        self.api_key = api_key
        self.calls = 0

    def complete_with_tools(self, messages, tools_, max_tokens=0,
                            temperature=0.0):
        self.calls += 1
        if not self.script:
            raise AssertionError("backend called more times than scripted")
        return self.script.pop(0)

    def complete(self, system, user, max_tokens=512, temperature=0.3):
        return "SUMMARY: agent did things with tools."


def _tool_call(name, args, call_id="c1"):
    return {"content": "", "tool_calls": [
        {"id": call_id, "name": name, "arguments": args}]}


def _final(text):
    return {"content": text, "tool_calls": []}


class TestAgentLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ceti-agent-")
        self._old_root = os.environ.get("CETI_WORK_ROOT")
        os.environ["CETI_WORK_ROOT"] = os.path.join(self.tmp, "work")
        self.cfg = BrainConfig(data_dir=os.path.join(self.tmp, "data"))

    def tearDown(self):
        if self._old_root is None:
            os.environ.pop("CETI_WORK_ROOT", None)
        else:
            os.environ["CETI_WORK_ROOT"] = self._old_root

    def test_write_file_then_done(self):
        backend = FakeBackend([
            _tool_call("write_file",
                       {"path": "hello.txt", "content": "hi from agent"}),
            _final("wrote the file"),
        ])
        sess = AgentSession(self.cfg, backend)
        try:
            result = sess.run("create hello.txt")
        finally:
            sess.close()
        self.assertEqual(result, "wrote the file")
        with open(os.path.join(os.environ["CETI_WORK_ROOT"],
                               "hello.txt")) as f:
            self.assertEqual(f.read(), "hi from agent")
        # session persisted
        log = os.path.join(sess.dir, "messages.jsonl")
        self.assertTrue(os.path.isfile(log))
        with open(log) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        roles = [l["role"] for l in lines]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant"])

    def test_unknown_tool_is_an_error_not_a_crash(self):
        backend = FakeBackend([
            _tool_call("nope", {}),
            _final("gave up gracefully"),
        ])
        sess = AgentSession(self.cfg, backend)
        try:
            result = sess.run("do the impossible")
        finally:
            sess.close()
        self.assertEqual(result, "gave up gracefully")
        tool_msgs = [m for m in sess.messages if m["role"] == "tool"]
        self.assertIn("no such tool", tool_msgs[0]["content"])

    def test_session_notes_tools(self):
        backend = FakeBackend([
            _tool_call("record_note", {"content": "found the answer: 42"}),
            _tool_call("recall_notes", {}),
            _final("noted"),
        ])
        sess = AgentSession(self.cfg, backend)
        try:
            sess.run("remember something")
        finally:
            sess.close()
        self.assertEqual(sess.notes.count(), 1)
        self.assertIn("42", sess.notes.format())

    def test_summarization_keeps_user_messages(self):
        script = []
        for i in range(4):
            script.append(_tool_call(
                "run_python", {"code": "print(%d)" % i}, call_id="c%d" % i))
        script.append(_final("all done"))
        backend = FakeBackend(script)
        sess = AgentSession(self.cfg, backend, token_limit=200)
        try:
            result = sess.run("count to three with python, one tool call "
                              "per number, then report")
        finally:
            sess.close()
        self.assertEqual(result, "all done")
        self.assertGreaterEqual(sess.summaries, 1)
        users = [m for m in sess.messages if m.get("role") == "user"]
        # the original task survives summarization
        self.assertTrue(any("count to three" in m["content"]
                            for m in users))

    def test_summarization_trims_block_before_oracle(self):
        """The summarizer gets the trimmed transcript, not the raw block:
        fewer oracle tokens for the same signal."""

        class RecordingBackend(FakeBackend):
            def __init__(self, script):
                super().__init__(script)
                self.seen_transcripts = []

            def complete(self, system, user, max_tokens=512,
                         temperature=0.3):
                self.seen_transcripts.append(user)
                return "SUMMARY"

        backend = RecordingBackend([])
        sess = AgentSession(self.cfg, backend)
        sess.task = "why did the payment fail? find the ledger error"
        try:
            lines = []
            for i in range(300):
                lines.append(
                    "2026-10-11T03:%02d:%02dZ INFO checkout request %d "
                    "served in %dms"
                    % (i // 60 % 60, i % 60, 1000 + i, 10 + i % 17))
            lines.append("2026-10-11T03:04:31Z ERROR payment client: "
                         "ledger-db timeout after 30s (order 1271)")
            block = [
                {"role": "assistant", "content": "checking the logs",
                 "tool_calls": [
                     {"name": "read_file",
                      "arguments": {"path": "server.log"}}]},
                {"role": "tool", "name": "read_file",
                 "content": "\n".join(lines)},
            ]
            summary = sess._summarize_block(block)
            self.assertEqual(summary, "SUMMARY")
            self.assertEqual(len(backend.seen_transcripts), 1)
            seen = backend.seen_transcripts[0]
            self.assertLess(len(seen), 2000)
            self.assertIn("ledger-db timeout", seen)
        finally:
            sess.close()

    def test_max_steps(self):
        backend = FakeBackend(
            [_tool_call("list_dir", {"path": "."}, call_id="c%d" % i)
             for i in range(10)])
        sess = AgentSession(self.cfg, backend, max_steps=3)
        try:
            result = sess.run("loop forever")
        finally:
            sess.close()
        self.assertIn("stopped after 3 steps", result)

    def test_no_key_refuses(self):
        backend = FakeBackend([], api_key=None)
        with self.assertRaises(AgentError):
            AgentSession(self.cfg, backend)

    def test_run_task_helper(self):
        backend = FakeBackend([_final("quick")])
        result, sid = run_task(self.cfg, backend, "hi")
        self.assertEqual(result, "quick")
        self.assertTrue(sid)

    def test_empty_stall_nudges_then_completes(self):
        backend = FakeBackend([
            {"content": "", "tool_calls": []},
            {"content": "", "tool_calls": []},
            _final("finally done"),
        ])
        sess = AgentSession(self.cfg, backend)
        try:
            result = sess.run("do it")
        finally:
            sess.close()
        self.assertEqual(result, "finally done")
        nudges = [m for m in sess.messages
                  if m.get("role") == "user"
                  and "[nudge]" in m.get("content", "")]
        self.assertEqual(len(nudges), 2)

    def test_empty_stall_gives_up_after_two_nudges(self):
        backend = FakeBackend([{"content": "", "tool_calls": []}] * 5)
        sess = AgentSession(self.cfg, backend)
        try:
            result = sess.run("do it")
        finally:
            sess.close()
        self.assertEqual(result, "(no response)")

    def test_estimate_tokens(self):
        msgs = [{"role": "user", "content": "x" * 400}]
        self.assertGreaterEqual(estimate_tokens(msgs), 100)

    def test_tool_schemas_exported(self):
        backend = FakeBackend([_final("x")])
        sess = AgentSession(self.cfg, backend)
        try:
            names = {s["function"]["name"] for s in sess.schemas}
        finally:
            sess.close()
        self.assertTrue({"run_python", "write_file", "record_note",
                         "job_poll"} <= names)


if __name__ == "__main__":
    unittest.main()
