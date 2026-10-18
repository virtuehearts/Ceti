"""Tool registry: the agent's hands.

Each tool is a Tool object — name, description, JSON-schema
parameters, side effects, sandbox flag, and an execute() that
returns a ToolResult. Only registered tools can be invoked;
unregistered code doesn't run.

to_openai_schema() exports OpenAI function-calling schemas so a
cloud model (the oracle) can drive the tools directly. The agent
loop (ceti/agent_loop.py) does exactly that.

Execution policy:
- run_python / run_python_bg: ALWAYS in the sandbox
  (ceti/sandbox.py). The agent never executes code on the host.
- File tools (read/write/edit/list): host-side, but jailed to the
  work root (CETI_WORK_ROOT or data/scratch). Symlink escapes and
  ".." are refused.
- job_poll / job_kill: manage background sandbox jobs.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from . import sandbox as _sandbox
from . import trim as _trim
from .config import data_dir_default


class ToolError(Exception):
    """A tool call failed (bad args, jail refusal, ...)."""


@dataclass
class ToolResult:
    """The outcome of one tool execution."""
    success: bool
    content: str = ""
    error: str | None = None


class Tool:
    """One callable tool. execute() never raises ToolError —
    failures come back as ToolResult(success=False)."""

    def __init__(self, name: str, description: str,
                 parameters: dict, func,
                 side_effects: str = "none",
                 sandboxed: bool = False):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.side_effects = side_effects
        self.sandboxed = sandboxed
        self.func = func

    def execute(self, **kwargs) -> ToolResult:
        try:
            return ToolResult(success=True,
                              content=str(self.func(**kwargs)))
        except ToolError as e:
            return ToolResult(success=False, error=str(e))
        except TypeError as e:
            return ToolResult(success=False,
                              error="bad arguments: %s" % e)

    def call(self, **kwargs):
        """Raw call; raises ToolError on failure (REPL path)."""
        return self.func(**kwargs)

    def manifest(self) -> dict:
        props = self.parameters.get("properties", {})
        return {"name": self.name, "description": self.description,
                "args": {k: v.get("description", "")
                         for k, v in props.items()},
                "side_effects": self.side_effects,
                "sandboxed": self.sandboxed}

    def to_openai_schema(self) -> dict:
        """OpenAI function-calling schema for the oracle."""
        return {"type": "function",
                "function": {"name": self.name,
                             "description": self.description,
                             "parameters": self.parameters}}


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties,
            "required": required, "additionalProperties": False}


def work_root() -> str:
    """The only directory file tools may touch."""
    root = os.environ.get("CETI_WORK_ROOT") or os.path.join(
        data_dir_default(), "scratch")
    os.makedirs(root, exist_ok=True)
    return os.path.realpath(root)


def _resolve(path: str) -> str:
    """Resolve path inside the work root; refuse escapes."""
    root = work_root()
    # normalize: strip leading /, collapse ..
    rel = os.path.normpath(path.lstrip("/"))
    dest = os.path.realpath(os.path.join(root, rel))
    if dest != root and not dest.startswith(root + os.sep):
        raise ToolError("refused: %r escapes the work root" % path)
    return dest


def _coerce_timeout(value, default: int, lo: int, hi: int) -> int:
    try:
        t = int(value)
    except (TypeError, ValueError):
        raise ToolError("timeout_s must be an integer")
    return max(lo, min(t, hi))


def _format_result(res: "_sandbox.RunResult") -> str:
    out = []
    if res.stdout.strip():
        out.append(res.stdout.strip())
    if res.stderr.strip():
        out.append("stderr:\n" + res.stderr.strip())
    out.append("[exit %d%s]" % (
        res.exit_code, " — timed out" if res.timed_out else ""))
    return "\n".join(out)


def _tool_run_python(code: str, timeout_s: int = 10) -> str:
    if not isinstance(code, str) or not code.strip():
        raise ToolError("code must be a non-empty string")
    timeout_s = _coerce_timeout(timeout_s, 10, 1, 120)
    try:
        res = _sandbox.run_python(code, timeout_s=timeout_s)
    except _sandbox.SandboxError as e:
        raise ToolError("sandbox unavailable: %s" % e)
    return _format_result(res)


def _tool_run_python_bg(code: str, timeout_s: int = 300) -> str:
    if not isinstance(code, str) or not code.strip():
        raise ToolError("code must be a non-empty string")
    timeout_s = _coerce_timeout(timeout_s, 300, 1, 600)
    try:
        job = _sandbox.run_python_bg(code, timeout_s=timeout_s)
    except _sandbox.SandboxError as e:
        raise ToolError("sandbox unavailable: %s" % e)
    _JOBS[job.job_id] = job
    return "started %s — poll it with job_poll, stop it with job_kill" \
        % job.job_id


def _lookup_job(job_id: str) -> "_sandbox.BackgroundJob":
    job = _JOBS.get(job_id)
    if job is None:
        raise ToolError("no such job: %r (active: %s)"
                        % (job_id, ", ".join(sorted(_JOBS)) or "none"))
    return job


def _tool_job_poll(job_id: str) -> str:
    job = _lookup_job(job_id)
    res = job.poll()
    state = "done" if res.done else "running"
    out = ["%s: %s" % (job_id, state)]
    body = _format_result(res) if res.done else res.stdout.strip()
    if body:
        out.append(body[-4000:])
    if res.done:
        _JOBS.pop(job_id, None)  # reap finished jobs from the table
    return "\n".join(out)


def _tool_job_kill(job_id: str) -> str:
    job = _lookup_job(job_id)
    job.kill()
    _JOBS.pop(job_id, None)
    return "killed %s" % job_id


def _tool_read_file(path: str) -> str:
    dest = _resolve(path)
    if not os.path.isfile(dest):
        raise ToolError("no such file: %r" % path)
    if os.path.getsize(dest) > 512 * 1024:
        raise ToolError("file too large (>512KB): %r" % path)
    with open(dest, encoding="utf-8", errors="replace") as f:
        return f.read()


def _tool_write_file(path: str, content: str) -> str:
    if not isinstance(content, str):
        raise ToolError("content must be a string")
    if len(content) > 2 * 1024 * 1024:
        raise ToolError("content too large (>2MB)")
    dest = _resolve(path)
    os.makedirs(os.path.dirname(dest) or work_root(), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(content)
    return "wrote %d bytes to %s" % (len(content.encode()), path)


def _tool_edit_file(path: str, old: str, new: str) -> str:
    dest = _resolve(path)
    if not os.path.isfile(dest):
        raise ToolError("no such file: %r" % path)
    with open(dest, encoding="utf-8", errors="replace") as f:
        text = f.read()
    if old not in text:
        raise ToolError("pattern not found in %r" % path)
    text = text.replace(old, new, 1)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    return "edited %s (1 occurrence replaced)" % path


def _tool_trim(text: str, query: str, point: str = "balanced") -> str:
    """Compress text to the lines the query needs (see ceti/trim.py).

    Verbatim, in order, with "… N lines …" markers. Fails closed:
    short inputs, already-trimmed inputs and cuts too small to matter
    come back unchanged, and the note says so.
    """
    if not isinstance(text, str) or not isinstance(query, str):
        raise ToolError("text and query must be strings")
    res = _trim.compress(text, query, point=point)
    if res.unchanged:
        return ("trim: unchanged (%s) — %d bytes kept as-is"
                % (res.reason, len(text.encode("utf-8"))))
    return ("trim: kept %d of %d lines, %d of %d bytes (point=%s)\n%s"
            % (res.kept_lines, res.kept_lines + res.removed_lines,
               len(res.output.encode("utf-8")),
               len(text.encode("utf-8")), res.point, res.output))


def _tool_list_dir(path: str = ".") -> str:
    dest = _resolve(path)
    if not os.path.isdir(dest):
        raise ToolError("no such directory: %r" % path)
    entries = sorted(os.listdir(dest))
    lines = []
    for e in entries[:200]:
        full = os.path.join(dest, e)
        tag = "/" if os.path.isdir(full) else ""
        size = "" if os.path.isdir(full) else " (%dB)" % os.path.getsize(full)
        lines.append("  %s%s%s" % (e, tag, size))
    if len(entries) > 200:
        lines.append("  ... (%d more)" % (len(entries) - 200))
    return "\n".join(lines) or "(empty)"


_JOBS: dict[str, "_sandbox.BackgroundJob"] = {}
_REGISTRY: dict[str, Tool] = {}


def _register(tool: Tool) -> None:
    _REGISTRY[tool.name] = tool


def register_tool(tool: Tool) -> None:
    """Add a tool to the registry (e.g. session-scoped notes tools)."""
    _register(tool)


def make_note_tools(record, recall) -> list[Tool]:
    """Build record_note / recall_notes tools bound to callables."""
    return [
        Tool(
            name="record_note",
            description="Record a timestamped note to this session's "
                        "local memory. Use it for facts, decisions, and "
                        "findings worth remembering later in the session.",
            parameters=_schema(
                {"content": {"type": "string",
                             "description": "the note to record"}},
                ["content"]),
            side_effects="write", sandboxed=False, func=record),
        Tool(
            name="recall_notes",
            description="Read back this session's recorded notes.",
            parameters=_schema(
                {"limit": {"type": "integer",
                           "description": "newest N notes (default: all)"}},
                []),
            side_effects="read", sandboxed=False, func=recall),
    ]


_register(Tool(
    name="run_python",
    description="Run Python code in the isolated sandbox. Returns "
                "stdout, stderr, and the exit code. No network, no "
                "host filesystem access, capped memory/CPU.",
    parameters=_schema(
        {"code": {"type": "string",
                  "description": "Python source to run"},
         "timeout_s": {"type": "integer",
                       "description": "wall-clock limit, 1-120 (default 10)",
                       "default": 10}},
        ["code"]),
    side_effects="none", sandboxed=True, func=_tool_run_python))
_register(Tool(
    name="run_python_bg",
    description="Launch Python code in the isolated sandbox WITHOUT "
                "waiting. Returns a job id; use job_poll to check "
                "output and job_kill to stop it.",
    parameters=_schema(
        {"code": {"type": "string",
                  "description": "Python source to run"},
         "timeout_s": {"type": "integer",
                       "description": "wall-clock limit, 1-600 "
                                      "(default 300)",
                       "default": 300}},
        ["code"]),
    side_effects="none", sandboxed=True, func=_tool_run_python_bg))
_register(Tool(
    name="job_poll",
    description="Check a background job: running state plus output "
                "so far, or the final result. Finished jobs are "
                "reaped from the table.",
    parameters=_schema(
        {"job_id": {"type": "string",
                    "description": "job id from run_python_bg"}},
        ["job_id"]),
    side_effects="read", sandboxed=False, func=_tool_job_poll))
_register(Tool(
    name="job_kill",
    description="Terminate a background job.",
    parameters=_schema(
        {"job_id": {"type": "string",
                    "description": "job id from run_python_bg"}},
        ["job_id"]),
    side_effects="write", sandboxed=False, func=_tool_job_kill))
_register(Tool(
    name="read_file",
    description="Read a text file from the work root.",
    parameters=_schema(
        {"path": {"type": "string",
                  "description": "path relative to the work root"}},
        ["path"]),
    side_effects="read", sandboxed=False, func=_tool_read_file))
_register(Tool(
    name="write_file",
    description="Write (create or overwrite) a text file in the work "
                "root. Parent directories are created.",
    parameters=_schema(
        {"path": {"type": "string",
                  "description": "path relative to the work root"},
         "content": {"type": "string",
                     "description": "file content"}},
        ["path", "content"]),
    side_effects="write", sandboxed=False, func=_tool_write_file))
_register(Tool(
    name="edit_file",
    description="Replace the first occurrence of `old` with `new` in "
                "a work-root file. Fails loudly if not found.",
    parameters=_schema(
        {"path": {"type": "string",
                  "description": "path relative to the work root"},
         "old": {"type": "string",
                 "description": "exact text to find"},
         "new": {"type": "string",
                 "description": "replacement text"}},
        ["path", "old", "new"]),
    side_effects="write", sandboxed=False, func=_tool_edit_file))
_register(Tool(
    name="list_dir",
    description="List a directory in the work root.",
    parameters=_schema(
        {"path": {"type": "string",
                  "description": "directory relative to the work root "
                                 "(default '.')"}},
        []),
    side_effects="read", sandboxed=False, func=_tool_list_dir))


_register(Tool(
    name="trim",
    description="Compress text to the lines the current ask needs: "
                "verbatim, in original order, with markers where lines "
                "were removed. Best on logs and long tool output; "
                "repetitive lines cut hard, prose is kept generously. "
                "Fails closed (returns the input with a note) when the "
                "cut would be too small to matter. point: conservative, "
                "balanced (default) or aggressive.",
    parameters=_schema(
        {"text": {"type": "string",
                  "description": "text to compress (a long tool result, "
                                 "a file, a log)"},
         "query": {"type": "string",
                   "description": "the current ask — what the kept lines "
                                  "must answer"},
         "point": {"type": "string",
                   "description": "how hard to cut: conservative, "
                                  "balanced (default), aggressive",
                   "default": "balanced"}},
        ["text", "query"]),
    side_effects="none", sandboxed=False, func=_tool_trim))


def list_tools() -> list[Tool]:
    return list(_REGISTRY.values())


def all_tools() -> dict[str, Tool]:
    """A fresh name->Tool table (sessions extend it with their own)."""
    return dict(_REGISTRY)


def manifest() -> list[dict]:
    """Machine-readable tool manifests for the agent."""
    return [t.manifest() for t in list_tools()]


def openai_schemas() -> list[dict]:
    """OpenAI function-calling schemas, for the oracle."""
    return [t.to_openai_schema() for t in list_tools()]


def call(tool_name: str, **kwargs):
    """Invoke a registered tool by name. Unknown names raise."""
    tool = _REGISTRY.get(tool_name)
    if tool is None:
        raise ToolError("no such tool: %r (see /tools)" % tool_name)
    return tool.call(**kwargs)


def execute(tool_name: str, **kwargs) -> ToolResult:
    """Invoke a registered tool, returning a ToolResult (no raise)."""
    tool = _REGISTRY.get(tool_name)
    if tool is None:
        return ToolResult(success=False,
                          error="no such tool: %r (see /tools)"
                          % tool_name)
    return tool.execute(**kwargs)


# ---- C2 console tools: dictionary / thesaurus / memopad / weather --
# The day-of-use console: a command-and-control system needs a
# dictionary, a thesaurus, a memopad, and the weather — the same
# four things a sysop's desk has. Dictionary/thesaurus/weather
# fetch from free no-key APIs; a local word list covers the day's
# vocabulary when the network is down. Weather fails LOUD (a
# fabricated forecast is worse than none).

_LOCAL_WORDS = {
    "modem": ("A device that modulates a digital signal onto a "
              "telephone line and demodulates it back — the bridge "
              "between a computer and the phone network.",
              ["coupler", "interface"]),
    "baud": ("Signalling rate: symbols per second. A 2400 baud "
             "modem sends 2400 symbols per second.",
             ["rate", "speed"]),
    "bulletin": ("A short public notice posted for a group to read.",
                 ["notice", "announcement"]),
    "cipher": ("A method for transforming a message so only the "
               "intended reader can read it.",
               ["code", "cryptogram"]),
    "cypherpunk": ("Someone who uses cryptography to defend privacy "
                   "and free speech — 'privacy for the weak, "
                   "transparency for the powerful'.",
                   ["crypto-anarchist", "privacy advocate"]),
    "commodore": ("Commodore International: maker of the PET, VIC-20, "
                  "Commodore 64, and Amiga.",
                  ["c64 maker"]),
    "bbs": ("Bulletin Board System: a computer you dial into to "
           "read messages and share files.",
           ["board", "bulletin board"]),
    "sysop": ("The system operator — the person who runs a BBS.",
              ["operator", "host"]),
    "gold": ("In Ultima Online, the in-game currency players farmed "
             "and sold for real money — virtual goods with real value.",
             ["currency", "loot"]),
    "manifesto": ("A public declaration of principles and intent.",
                  ["declaration", "creed"]),
    "key": ("In cryptography, the secret that makes a signature or "
            "an encryption yours.",
            ["secret", "private key"]),
    "signature": ("A cryptographic proof that a message came from the "
                  "holder of a key — and was not altered.",
                  ["seal", "attestation"]),
}


def _fetch_json(url: str, timeout_s: int = 10) -> object:
    req = urllib.request.Request(url, headers={
        "User-Agent": "CETI/0.3.3 (sovereign local brain)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        raise ToolError("fetch failed: HTTP %s" % e.code)
    except Exception as e:
        raise ToolError("fetch failed: %s" % e)


def _tool_dictionary(word: str) -> str:
    word = (word or "").strip().lower()
    if not word:
        raise ToolError("dictionary needs a word")
    api_error: Exception | None = None
    data = None
    try:
        data = _fetch_json(
            "https://api.dictionaryapi.dev/api/v2/entries/en/"
            + urllib.parse.quote(word))
    except Exception as e:
        api_error = e  # API miss or unreachable: try the local list
    if data is not None:
        entry = data[0] if isinstance(data, list) and data else None
        if entry:
            lines = [entry.get("word", word)]
            for meaning in entry.get("meanings", [])[:2]:
                pos = meaning.get("partOfSpeech", "")
                for d in meaning.get("definitions", [])[:2]:
                    lines.append("- %s: %s" % (
                        pos, d.get("definition", "")))
            return "\n".join(lines)
    if word in _LOCAL_WORDS:
        definition, _syn = _LOCAL_WORDS[word]
        return "%s\n- %s" % (word, definition)
    if api_error is not None and "HTTP 404" not in str(api_error):
        # Reachable-API 404 means "no such word"; anything else
        # means the fetch itself failed — say that, don't claim
        # the word has no definition.
        raise ToolError(
            "dictionary API unreachable (%s); no definition found "
            "for %r in local word list" % (api_error, word))
    raise ToolError("no definition found for %r" % word)


def _tool_thesaurus(word: str) -> str:
    word = (word or "").strip().lower()
    if not word:
        raise ToolError("thesaurus needs a word")
    api_error: Exception | None = None
    data = None
    try:
        data = _fetch_json(
            "https://api.dictionaryapi.dev/api/v2/entries/en/"
            + urllib.parse.quote(word))
    except Exception as e:
        api_error = e  # API miss or unreachable: try the local list
    if data is not None:
        entry = data[0] if isinstance(data, list) and data else None
        if entry:
            synonyms: list[str] = []
            antonyms: list[str] = []
            for meaning in entry.get("meanings", []):
                for d in meaning.get("definitions", []):
                    synonyms += d.get("synonyms", []) or []
                    antonyms += d.get("antonyms", []) or []
            if not synonyms and not antonyms:
                return "%s: no synonyms or antonyms listed" % word
            out = [word]
            if synonyms:
                out.append("synonyms: " + ", ".join(sorted(set(synonyms))[:10]))
            if antonyms:
                out.append("antonyms: " + ", ".join(sorted(set(antonyms))[:10]))
            return "\n".join(out)
    if word in _LOCAL_WORDS:
        _definition, syns = _LOCAL_WORDS[word]
        if syns:
            return "%s\nsynonyms: %s" % (word, ", ".join(syns))
        return "%s: no synonyms listed" % word
    if api_error is not None and "HTTP 404" not in str(api_error):
        # Reachable-API 404 means "no such word"; anything else
        # means the fetch itself failed — say that, don't claim
        # the word has no entry.
        raise ToolError(
            "thesaurus API unreachable (%s); no entry found "
            "for %r in local word list" % (api_error, word))
    raise ToolError("no entry found for %r" % word)


def _memopad_path() -> str:
    return os.path.join(work_root(), "memopad.txt")


def _tool_memopad(action: str, text: str = "",
                  limit: int = 20) -> str:
    action = (action or "").strip().lower()
    if action == "append":
        text = (text or "").strip()
        if not text:
            raise ToolError("memopad append needs text")
        stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
        with open(_memopad_path(), "a", encoding="utf-8") as f:
            f.write("[%s] %s\n" % (stamp, text))
        return "noted."
    if action == "read":
        try:
            with open(_memopad_path(), encoding="utf-8") as f:
                lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        except OSError:
            return "(memopad is empty)"
        if not lines:
            return "(memopad is empty)"
        return "\n".join(lines[-max(1, int(limit)):])
    if action == "clear":
        with open(_memopad_path(), "w", encoding="utf-8") as f:
            f.write("")
        return "memopad cleared."
    raise ToolError("memopad action must be append, read, or clear")


def _tool_weather(place: str) -> str:
    place = (place or "").strip()
    if not place:
        raise ToolError("weather needs a place")
    data = _fetch_json(
        "https://wttr.in/" + urllib.parse.quote(place) + "?format=j1")
    try:
        cur = data["current_condition"][0]
        area = data.get("nearest_area", [{}])[0]
        name = area.get("areaName", [{}])[0].get("value", place)
        country = area.get("country", [{}])[0].get("value", "")
        desc = cur.get("weatherDesc", [{}])[0].get("value", "?")
        return ("%s, %s: %s, %s°C, humidity %s%%, wind %s km/h %s"
                % (name, country, desc, cur.get("temp_C", "?"),
                   cur.get("humidity", "?"),
                   cur.get("windspeedKmph", "?"),
                   cur.get("winddir16Point", ""))).strip()
    except (KeyError, IndexError, TypeError):
        raise ToolError("weather: unexpected response shape")


_register(Tool(
    name="dictionary",
    description="Look up a word's definitions (free dictionary API; "
                "a local word list covers the day's vocabulary "
                "when the network is down).",
    parameters=_schema(
        {"word": {"type": "string", "description": "the word to define"}},
        ["word"]),
    side_effects="read", sandboxed=False, func=_tool_dictionary))
_register(Tool(
    name="thesaurus",
    description="Look up a word's synonyms and antonyms (same free "
                "dictionary API; local word list as fallback).",
    parameters=_schema(
        {"word": {"type": "string",
                  "description": "the word to find synonyms for"}},
        ["word"]),
    side_effects="read", sandboxed=False, func=_tool_thesaurus))
_register(Tool(
    name="memopad",
    description="The console memopad: append a timestamped line, "
                "read the last N lines, or clear it. Lives in the "
                "work root (memopad.txt).",
    parameters=_schema(
        {"action": {"type": "string",
                    "description": "append, read, or clear"},
         "text": {"type": "string",
                  "description": "the line to append (append only)"},
         "limit": {"type": "integer",
                   "description": "newest N lines (read only, "
                                  "default 20)",
                   "default": 20}},
        ["action"]),
    side_effects="write", sandboxed=False, func=_tool_memopad))
_register(Tool(
    name="weather",
    description="Current weather for a place (wttr.in, no key). "
                "Fails loudly when the fetch fails — never invents "
                "a forecast.",
    parameters=_schema(
        {"place": {"type": "string",
                   "description": "city or place name"}},
        ["place"]),
    side_effects="read", sandboxed=False, func=_tool_weather))


# ---- TTS: speak through the PocketTTS bridge --------------------
# The bridge is the sovereign voice layer (self-hosted, local-only,
# OpenAI-compatible /v1/audio/speech). Default voice mya-03.
# Audio lands in the work root (tts/). Fail LOUD: a down bridge,
# an unknown voice, or empty audio comes back as success=False —
# never faked, never silence-as-speech.

def _tool_tts(text: str, voice: str | None = None,
              filename: str | None = None) -> str:
    from . import tts as _tts
    if not isinstance(text, str) or not text.strip():
        raise ToolError("tts needs non-empty text")
    try:
        res = _tts.synthesize_to_file(
            text, voice=(voice or None),
            out_dir=os.path.join(work_root(), "tts"),
            filename=(filename or None))
    except _tts.TTSError as e:
        raise ToolError("TTS FAILED: %s" % e)
    return _tts.describe(res)


_register(Tool(
    name="tts",
    description="Speak text through the local PocketTTS bridge and "
                "save the WAV to the work root (tts/). Default voice "
                "mya-03. Fails loudly when the bridge is down, the "
                "voice is unknown, or the bridge returns no audio — "
                "never fakes audio, never returns silence as speech.",
    parameters=_schema(
        {"text": {"type": "string",
                   "description": "the text to speak aloud"},
         "voice": {"type": "string",
                   "description": "bridge voice (default mya-03)",
                   "default": "mya-03"},
         "filename": {"type": "string",
                      "description": "plain .wav file name to save as "
                                     "(default: timestamped)"}},
        ["text"]),
    side_effects="write", sandboxed=False, func=_tool_tts))
