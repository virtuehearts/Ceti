"""Oracle-driven agent loop: a cloud model calls tools to do tasks.

An AgentSession is one task run:
- message history (system + user + assistant + tool messages)
- a tool table: the global registry plus session-scoped notes tools
- per-session local memory: notes.json + messages.jsonl under
  data/sessions/<session_id>/ — plain files, no server, no model
- context summarization: when estimated tokens exceed the limit,
  assistant/tool rounds between user messages are summarized away
  (user messages are always kept). Before the oracle summarizer reads
  a block, trim (ceti/trim.py) cuts it to the lines the task needs —
  fewer oracle tokens, same signal. Fails closed: if trim returns the
  block unchanged, the summarizer reads the whole block.
- retry with backoff on transient cloud failures

The loop needs a cloud backend with an API key and tool-calling
support (OpenAI-compatible). Without one it refuses loudly.
"""

from __future__ import annotations

import json
import os
import time

from . import tools as _tools
from .backends import BackendError
from .notes import SessionNotes
from .retry import retry
from . import skills as _skills
from . import trim as _trim


class AgentError(Exception):
    """The agent loop cannot run (no keyed backend, bad config, ...)."""


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate for a message list (chars/4)."""
    total = 0
    for m in messages:
        total += len(str(m.get("content") or "")) // 4
        for tc in m.get("tool_calls") or []:
            total += len(json.dumps(tc)) // 4
        total += 4  # per-message overhead
    return total


_AGENT_SYSTEM = """You are the task agent of CETI, a sovereign local-first AI \
that runs on its owner's hardware. You complete tasks by calling tools.

Rules:
- Work inside the work root shown below; file tools are jailed there.
- Python code runs ONLY through run_python / run_python_bg, inside an \
isolated sandbox: no network, no host filesystem, capped CPU and memory.
- For code that takes a while, launch it with run_python_bg and poll it \
with job_poll; stop runaways with job_kill.
- Prefer doing over asking: use the tools, observe the results, iterate.
- Record important findings, decisions, and facts with record_note so \
they survive the session.
- When the task says to build from a draft / source file, preserve \
that file's content VERBATIM — do not drop, summarize, paraphrase, \
or "improve" any part of it. Read the whole source, then reproduce \
every paragraph, list item, and line in the output. Dropping a \
paragraph is a failure, even if the rest looks good.
- When the task is complete, reply in plain text with NO further tool \
calls. Be concise: what was done, what the result was.
- If a task is impossible with the tools available, say so plainly \
instead of guessing.

Work root: {root}"""

_SUMMARIZER_SYSTEM = """You summarize an agent's tool-use session. \
Be concise and concrete: what was tried, what worked, key results, \
file paths touched. Under 400 words. Plain text, no preamble."""


class AgentSession:
    """One oracle-driven task run with its own local memory."""

    def __init__(self, cfg, backend, max_steps: int = 25,
                 token_limit: int = 24000, session_id: str | None = None,
                 skill_loader: "_skills.SkillLoader | None" = None):
        if backend is None or not getattr(backend, "api_key", None):
            raise AgentError(
                "agent loop needs a cloud backend with an API key "
                "(OPENROUTER_API_KEY or NVIDIA_API_KEY)")
        if not hasattr(backend, "complete_with_tools"):
            raise AgentError(
                "backend %r has no tool-calling support"
                % getattr(backend, "name", "?"))
        self.cfg = cfg
        self.backend = backend
        self.max_steps = max_steps
        self.token_limit = token_limit
        self.session_id = session_id or time.strftime("%Y%m%d-%H%M%S")
        self.dir = os.path.join(os.path.abspath(cfg.data_dir),
                                "sessions", self.session_id)
        os.makedirs(self.dir, exist_ok=True)

        # per-session local memory
        self.notes = SessionNotes(os.path.join(self.dir, "notes.json"))

        # skills: list_skills/read_skill tools + prompt listing
        self.skill_loader = (skill_loader
                             or _skills.ensure_tools_registered())

        # tool table: global registry + session notes tools
        self.tool_table: dict[str, _tools.Tool] = _tools.all_tools()
        for t in _tools.make_note_tools(self._record_note,
                                        self._recall_notes):
            self.tool_table[t.name] = t
        self.schemas = [t.to_openai_schema()
                        for t in self.tool_table.values()]

        system = _AGENT_SYSTEM.format(root=_tools.work_root())
        skills_section = self.skill_loader.prompt_section()
        if skills_section:
            system += "\n\n" + skills_section
        self.messages: list[dict] = [{"role": "system",
                                      "content": system}]
        self._logf = open(os.path.join(self.dir, "messages.jsonl"),
                          "a", encoding="utf-8")
        self.summaries = 0
        self.task = ""  # set by run(); the trim query for summarization

    # ---- session notes (bound as tools) --------------------------------
    def _record_note(self, content: str) -> str:
        idx = self.notes.record(content)
        return "recorded note #%d" % idx

    def _recall_notes(self, limit: int | None = None) -> str:
        return self.notes.format(limit)

    # ---- message log ----------------------------------------------------
    def _append(self, msg: dict) -> None:
        self.messages.append(msg)
        self._logf.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._logf.flush()

    def close(self) -> None:
        try:
            self._logf.close()
        except OSError:
            pass

    # ---- tool execution --------------------------------------------------
    def _exec_tool(self, tc: dict) -> str:
        name = tc.get("name") or "?"
        tool = self.tool_table.get(name)
        if tool is None:
            return "error: no such tool %r" % name
        args = tc.get("arguments")
        if not isinstance(args, dict):
            args = {}
        result = tool.execute(**args)
        if result.success:
            return result.content
        return "error: %s" % (result.error or "tool failed")

    # ---- summarization ---------------------------------------------------
    def _summarize_block(self, block: list[dict]) -> str:
        lines = []
        for m in block:
            role = m.get("role")
            if role == "assistant":
                if m.get("content"):
                    lines.append("Assistant: %s" % m["content"])
                for tc in m.get("tool_calls") or []:
                    lines.append("  -> %s(%s)"
                                 % (tc.get("name"),
                                    json.dumps(tc.get("arguments"))[:300]))
            elif role == "tool":
                # A generous per-result cap: the old 500-char cut hid the
                # tail, where errors live. Trim (below) does the real
                # cutting, against the task; the 8000-char fallback after
                # it is only a safety valve for when trim fails closed.
                lines.append("  <- %s: %s"
                             % (m.get("name") or "tool",
                                str(m.get("content") or "")[:20000]))
        transcript = "\n".join(lines)
        # Trim to the lines the task needs before the oracle summarizer
        # reads it: fewer oracle tokens, same signal. Fails closed — a
        # trim failure leaves the full transcript for the summarizer.
        try:
            trimmed = _trim.compress(
                transcript, self.task or "summarize the agent's work")
            if not trimmed.unchanged:
                transcript = trimmed.output
        except Exception:  # noqa: BLE001 - summarization must not break
            pass
        if len(transcript) > 8000:
            transcript = transcript[:8000] + "\n[...truncated]"
        try:
            summary = retry(
                lambda: self.backend.complete(
                    _SUMMARIZER_SYSTEM, transcript,
                    max_tokens=600, temperature=0.2),
                attempts=2, exceptions=(BackendError,))
            return summary.strip()
        except BackendError:
            return (transcript[:2000]
                    + "\n[...truncated: summarizer unavailable]")

    def _maybe_summarize(self) -> None:
        """Compress assistant/tool rounds once past the token limit.

        User messages are always kept; the agent's work between them
        is replaced by a summary. The active round (after the last
        user message) is summarized too, so the loop keeps going.
        """
        if estimate_tokens(self.messages) <= self.token_limit:
            return
        system = self.messages[0]
        new = [system]
        pending: list[dict] = []

        def flush() -> None:
            if pending:
                summary = self._summarize_block(pending)
                new.append({"role": "user",
                            "content": "[Summary of earlier agent work]\n"
                                       + summary})
                pending.clear()

        for m in self.messages[1:]:
            if m.get("role") == "user":
                flush()
                new.append(m)
            else:
                pending.append(m)
        flush()
        self.messages = new
        self.summaries += 1
        with open(os.path.join(self.dir, "summaries.log"), "a",
                   encoding="utf-8") as f:
            f.write("%s summarized to %d tokens\n"
                    % (time.strftime("%H:%M:%S"),
                       estimate_tokens(self.messages)))

    # ---- main loop --------------------------------------------------------
    def run(self, task: str) -> str:
        """Run the task. Returns the agent's final report."""
        if not task or not task.strip():
            raise AgentError("empty task")
        self.task = task.strip()
        self._append({"role": "user", "content": task.strip()})
        empty_stalls = 0
        tool_calls_made = 0
        no_tool_nudges = 0
        # Reasoning models burn tokens on their thinking trace; the
        # loop needs headroom for reasoning AND tool calls.
        max_tokens = max(self.cfg.cloud_max_tokens, 8000)
        for step in range(self.max_steps):
            self._maybe_summarize()
            try:
                resp = retry(
                    lambda: self.backend.complete_with_tools(
                        self.messages, self.schemas,
                        max_tokens=max_tokens,
                        temperature=self.cfg.cloud_temperature),
                    attempts=3, exceptions=(BackendError,),
                    on_retry=lambda n, e: self._note_retry(n, e))
            except BackendError as e:
                return "[agent] cloud call failed after retries: %s" % e
            self._append({"role": "assistant",
                          "content": resp.get("content") or "",
                          "tool_calls": resp.get("tool_calls") or []})
            calls = resp.get("tool_calls") or []
            if not calls:
                content = (resp.get("content") or "").strip()
                if not content and empty_stalls < 2:
                    # the model stalled: nudge it back to work
                    empty_stalls += 1
                    self._append({
                        "role": "user",
                        "content": "[nudge] Your last response was empty. "
                                   "Keep working on the task: call tools "
                                   "to make progress, or reply in plain "
                                   "text when it is complete."})
                    continue
                if not content:
                    return content or "(no response)"
                if tool_calls_made == 0:
                    # A small model can hallucinate tool use: it claims
                    # "the file is created" without ever calling a tool.
                    # Never return that as success. Nudge once, then
                    # fail loudly instead of reporting a lie.
                    if no_tool_nudges < 1:
                        no_tool_nudges += 1
                        self._append({
                            "role": "user",
                            "content": "[nudge] You have not called any "
                                       "tools yet. Complete the task by "
                                       "calling the appropriate tool — "
                                       "do not just describe the result."})
                        continue
                    return ("[agent] task not completed: model claimed "
                            "completion without calling any tools")
                return content
            tool_calls_made += len(calls)
            for tc in calls:
                self._append({"role": "tool",
                              "tool_call_id": tc.get("id"),
                              "name": tc.get("name"),
                              "content": self._exec_tool(tc)})
        return ("[agent] stopped after %d steps without finishing; "
                "session kept at %s" % (self.max_steps, self.dir))

    def _note_retry(self, n: int, e: BaseException) -> None:
        self._append({"role": "system",
                      "content": "[retry %d after cloud error: %s]"
                                 % (n, e)})


def run_task(cfg, backend, task: str, **kwargs) -> tuple[str, str]:
    """One-shot: create a session, run the task, close it.

    Returns (result, session_id)."""
    sess = AgentSession(cfg, backend, **kwargs)
    try:
        return sess.run(task), sess.session_id
    finally:
        sess.close()
