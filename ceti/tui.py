"""tui — a full-screen terminal interface for CETI, Hermes-style.

A BBS, not a chat window. Launch lands on the Main Menu (like
Hermes: numbered doors, single-key choices, typing at the prompt).
Door 1 is the AI — typing there asks CETI, inside the TUI. The
line REPL stays as the fallback when curses is unavailable, and
for the few doors that still live there (/task, /story, /share,
/download, live /chat, /ingest).

Layout:
  header    CETI + fingerprint + tier status
  pane      Main Menu / submenu, or transcript (AI door / output)
  input     prompt (menu choice, or AI question / command)
  footer    key hints

Windows overflow fix: curses on Windows (windows-curses /
PDCurses) misbehaves when text is written to the last column —
writing exactly width chars at x=0, or anything into the
bottom-right cell, scrolls or raises, and the old render()
swallowed the error, leaving text hanging off the terminal.
Two rules now, everywhere: (1) wrap, never truncate — long
lines wrap to width-1 with word boundaries; (2) never write
the last column — put() trims to width-x-1 and skips
out-of-bounds rows. Tabs are expanded, \r / control chars
stripped, before wrapping.

Slash commands: typing / filters the command list into a popup
under the input. Tab completes, Up/Down picks, Enter accepts.
Up/Down (outside the popup) cycles input history. Ctrl-C
abandons a running answer and drops back to input.

Sessions: every chat is a saved session (one JSON file per
session under data/tui-sessions/). /new starts a fresh chat,
/sessions lists saved chats, /resume <id> reopens one,
/delete <id> removes one. Launching the TUI always starts
a fresh chat; resume is explicit. Sessions with no user line
are never written — an empty chat leaves no file behind.

The loop takes a screen object so tests can drive it with a
fake screen and a scripted key queue — no real terminal needed.

Streaming: when the brain offers stream_tokens(), questions
stream instead of blocking. Model tiers yield real chunks;
retrieval tiers yield one block, revealed at SIM_TPS (the
same 499 t/s display cap Answer uses, so the tag and the
reveal agree). The generator is pumped on the main thread —
Brain's sqlite handles are thread-affine — one event per
frame; the backend yields heartbeat Nones while waiting, so
keys stay responsive. Ctrl-C cancels the stream and keeps
the partial answer (not learned); Enter while streaming
cancels the old answer and starts the new one.
"""
from __future__ import annotations

import os
import textwrap
import time

from . import tui_sessions

SIM_TPS = 499.0  # simulated reveal rate, words/sec (Answer.MAX_TPS)

COMMANDS = (
    "/new", "/sessions", "/resume", "/delete", "/help",
    "/ask", "/gopher", "/peers", "/peer", "/board", "/share",
    "/fetch", "/nodelist", "/gossip", "/chat", "/mail", "/upload",
    "/download", "/tools", "/call", "/task", "/skills", "/ingest",
    "/story", "/stats", "/trace", "/tiers", "/heartbeat", "/tasks",
    "/agent", "/identity", "/benchmark", "/tui", "/quit",
    "/menu", "/ai", "/evals", "/bus",
)


# ---- BBS menus (Hermes-style) -------------------------------------------
# A menu is (title, [(key, label, action)]). Action is:
#   ("menu", name)  -> open submenu
#   ("ai",)         -> AI door: leave menu, typing asks CETI
#   ("cmd", "/...") -> run that slash command inside the TUI
#   ("quit",)       -> leave the TUI
# The AI is door 1 — the only place the model is reached. Every
# other door is local data / tools / status. Typing at the prompt
# in a menu picks a door; typing in the AI door asks a question.
MENUS: dict[str, tuple[str, list[tuple[str, str, tuple]]]] = {
    "main": ("CETI // MAIN MENU", [
        ("1", "AI — Ask CETI (typing here asks the model)", ("ai",)),
        ("2", "Board — Read signed posts", ("menu", "board")),
        ("3", "Mail — Inbox", ("menu", "mail")),
        ("4", "Files — Work root / drop", ("menu", "files")),
        ("5", "Node — Peers, identity, nodelist", ("menu", "node")),
        ("6", "System — Stats, evals, bus, benchmark", ("menu", "system")),
        ("7", "Library — Ingest / research status", ("menu", "library")),
        ("0", "Goodbye — Leave the BBS", ("quit",)),
    ]),
    "board": ("BOARD", [
        ("1", "List posts", ("cmd", "/board")),
        ("2", "Read post 1 (oldest)", ("cmd", "/board read 1")),
        ("0", "Back to Main Menu", ("menu", "main")),
    ]),
    "mail": ("MAIL", [
        ("1", "Inbox", ("cmd", "/mail")),
        ("0", "Back to Main Menu", ("menu", "main")),
    ]),
    "files": ("FILES", [
        ("1", "List work root", ("cmd", "/files")),
        ("0", "Back to Main Menu", ("menu", "main")),
    ]),
    "node": ("NODE", [
        ("1", "Identity", ("cmd", "/identity")),
        ("2", "Peers", ("cmd", "/peers")),
        ("3", "Board", ("cmd", "/board")),
        ("0", "Back to Main Menu", ("menu", "main")),
    ]),
    "system": ("SYSTEM", [
        ("1", "Stats", ("cmd", "/stats")),
        ("2", "Evals — coding / tool / knowledge", ("cmd", "/evals")),
        ("3", "Bus — POST memory map", ("cmd", "/bus post")),
        ("4", "Benchmark — self-exam", ("cmd", "/benchmark")),
        ("5", "Tiers", ("cmd", "/tiers")),
        ("0", "Back to Main Menu", ("menu", "main")),
    ]),
    "library": ("LIBRARY", [
        ("1", "Library status", ("cmd", "/library")),
        ("0", "Back to Main Menu", ("menu", "main")),
    ]),
}


def menu_lines(name: str) -> list[str]:
    """Render a menu as BBS text lines (no curses needed, tested)."""
    title, options = MENUS[name]
    lines = ["", f"  {title}", "  " + "=" * min(44, len(title) + 2)]
    for key, label, _action in options:
        lines.append(f"  [{key}] {label}")
    lines.append("")
    lines.append("  Your choice, SysOp? (type a number, /menu, /ai, /quit)")
    return lines


def complete_command(prefix: str) -> list[str]:
    """Commands matching a typed / prefix, in definition order."""
    return [c for c in COMMANDS if c.startswith(prefix)]


# ---- wrapping / safe output (Windows overflow fix) ----------------------
def clean_line(line: str) -> str:
    """Tabs -> spaces, strip \\r and control chars (keep \\n splits
    done by the caller). Curses on Windows renders tabs and control
    chars unpredictably; plain text is the only safe input."""
    line = line.replace("\t", "    ").replace("\r", "")
    return "".join(ch for ch in line if ch >= " " or ch == "\x1b")


def wrap_lines(lines: list[str], width: int) -> list[str]:
    """Wrap every logical line to width-1 (never the last column),
    word boundaries, long words broken. Short lines pass through
    unchanged — Transcript.render for short lines is identical to
    the old truncate behavior, which the tests pin down."""
    if width <= 1:
        return []
    avail = max(1, width - 1)
    out: list[str] = []
    for raw in lines:
        line = clean_line(raw)
        if len(line) <= avail:
            out.append(line)
            continue
        # textwrap handles word boundaries; break_long_words covers
        # URLs / long tokens that would otherwise overflow.
        wrapped = textwrap.wrap(
            line, width=avail, break_long_words=True,
            break_on_hyphens=False, expand_tabs=False,
            replace_whitespace=False, drop_whitespace=True)
        out.extend(wrapped or [""])
    return out


# ---- colour -----------------------------------------------------------
# Phosphor BBS palette for the curses TUI. init_palette() runs once
# inside run_tui (real curses only); tests drive TUI.run with a fake
# screen and never call it, so _attrs stays all-zero there and every
# addstr falls back to the default attribute. NO_COLOR disables it.
# The transcript stores plain strings — colour is a render-time
# classification, so saved sessions stay format-compatible.
_ATTRS: dict[str, int] = {}
_PALETTE_INIT = False


def init_palette() -> None:
    """Define colour pairs. Safe to call anywhere: no curses, no
    colour support, or NO_COLOR all leave the palette all-zero."""
    global _PALETTE_INIT
    if _PALETTE_INIT:
        return
    _PALETTE_INIT = True
    if os.environ.get("NO_COLOR") is not None:
        return
    try:
        import curses
        if not curses.has_colors():
            return
        curses.start_color()
        try:
            curses.use_default_colors()
        except Exception:
            pass
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(6, curses.COLOR_RED, curses.COLOR_BLACK)
        _ATTRS.update({
            "header": curses.color_pair(1) | curses.A_BOLD,
            "user": curses.color_pair(2),
            "ceti": curses.color_pair(3),
            "tag": curses.color_pair(4),
            "system": curses.color_pair(5) | curses.A_DIM,
            "alert": curses.color_pair(6) | curses.A_BOLD,
            "popup_sel": curses.color_pair(2) | curses.A_BOLD,
        })
    except Exception:
        _ATTRS.clear()


def attr_for(line: str) -> int:
    """Render-time role colour for one transcript line."""
    if line.startswith("you> "):
        return _ATTRS.get("user", 0)
    if line.startswith("["):
        return _ATTRS.get("tag", 0)
    if line.startswith("error:") or line.startswith("(cancelled)") \
            or line.lstrip().startswith("!"):
        return _ATTRS.get("alert", 0)
    if line.startswith("(") or line.startswith("  ") \
            or line.startswith("keys:") or line.startswith("fingerprint:") \
            or line.startswith("node_id:"):
        return _ATTRS.get("system", 0)
    return _ATTRS.get("ceti", 0)


class History:
    """Input history ring: Up/Down cycles, newest last.
    pos is an index into items; pos == len(items) means blank
    (the saved in-progress input)."""

    def __init__(self) -> None:
        self.items: list[str] = []
        self.pos: int = 0
        self._saved: str = ""

    def add(self, text: str) -> None:
        if text.strip():
            self.items.append(text)
        self.pos = len(self.items)
        self._saved = ""

    def prev(self, current: str) -> str:
        if not self.items:
            return current
        if self.pos == len(self.items):
            self._saved = current
        if self.pos > 0:
            self.pos -= 1
        return self.items[self.pos]

    def next(self, current: str) -> str:
        if not self.items:
            return current
        if self.pos < len(self.items):
            self.pos += 1
            if self.pos < len(self.items):
                return self.items[self.pos]
            return self._saved
        return current


class LineEditor:
    """Multiline input editor. Text is a list of lines; cursor is
    (row, col). Enter sends (handled by the caller); Ctrl-J
    inserts a newline. Pure logic — no curses calls."""

    def __init__(self, text: str = "") -> None:
        self.lines = text.split("\n") if text else [""]
        self.row = len(self.lines) - 1
        self.col = len(self.lines[-1])

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def clear(self) -> None:
        self.lines = [""]
        self.row = 0
        self.col = 0

    def set_text(self, text: str) -> None:
        self.lines = text.split("\n") if text else [""]
        self.row = len(self.lines) - 1
        self.col = len(self.lines[-1])

    def insert(self, ch: str) -> None:
        line = self.lines[self.row]
        self.lines[self.row] = line[:self.col] + ch + line[self.col:]
        self.col += 1

    def newline(self) -> None:
        line = self.lines[self.row]
        self.lines[self.row] = line[:self.col]
        self.lines.insert(self.row + 1, line[self.col:])
        self.row += 1
        self.col = 0

    def backspace(self) -> None:
        if self.col > 0:
            line = self.lines[self.row]
            self.lines[self.row] = line[:self.col - 1] + line[self.col:]
            self.col -= 1
        elif self.row > 0:
            prev = self.lines[self.row - 1]
            cur = self.lines.pop(self.row)
            self.row -= 1
            self.col = len(prev)
            self.lines[self.row] = prev + cur

    def delete(self) -> None:
        line = self.lines[self.row]
        if self.col < len(line):
            self.lines[self.row] = line[:self.col] + line[self.col + 1:]
        elif self.row < len(self.lines) - 1:
            self.lines[self.row] = line + self.lines.pop(self.row + 1)

    def move_left(self) -> None:
        if self.col > 0:
            self.col -= 1
        elif self.row > 0:
            self.row -= 1
            self.col = len(self.lines[self.row])

    def move_right(self) -> None:
        if self.col < len(self.lines[self.row]):
            self.col += 1
        elif self.row < len(self.lines) - 1:
            self.row += 1
            self.col = 0

    def move_up(self) -> None:
        if self.row > 0:
            self.row -= 1
            self.col = min(self.col, len(self.lines[self.row]))

    def move_down(self) -> None:
        if self.row < len(self.lines) - 1:
            self.row += 1
            self.col = min(self.col, len(self.lines[self.row]))

    def move_home(self) -> None:
        self.col = 0

    def move_end(self) -> None:
        self.col = len(self.lines[self.row])


class Transcript:
    """Transcript lines with a scroll offset. render() returns the
    window of lines that fits the pane. on_change fires on every
    add — the TUI uses it to auto-save the session."""

    def __init__(self, on_change=None) -> None:
        self.lines: list[str] = []
        self.offset: int = 0  # 0 == pinned to the bottom
        self.on_change = on_change

    def add(self, text: str) -> None:
        self.lines.extend(text.split("\n"))
        self.offset = 0
        if self.on_change is not None:
            self.on_change()

    def scroll_up(self, n: int = 1) -> None:
        self.offset = min(self.offset + n, max(0, len(self.lines) - 1))

    def scroll_down(self, n: int = 1) -> None:
        self.offset = max(0, self.offset - n)

    def render(self, height: int, width: int) -> list[str]:
        if height <= 0:
            return []
        # Scroll offset stays in LOGICAL lines (tests pin this);
        # wrapping happens after windowing, so short lines render
        # exactly as before and long lines wrap instead of truncating.
        end = len(self.lines) - self.offset
        start = max(0, end - height)
        wrapped = wrap_lines(self.lines[start:end], width)
        # A wrapped block can exceed the pane: show its tail.
        return wrapped[-height:] if len(wrapped) > height else wrapped


class TUI:
    """The app: owns editor, history, transcript, completion state.
    run(screen) is the curses loop; screen only needs getch,
    getmaxyx, addstr, erase, refresh, and curs_set (all present
    on a real curses window and on the test fake)."""

    def __init__(self, brain, ask_fn=None, data_dir=None) -> None:
        self.brain = brain
        self.ask_fn = ask_fn or (lambda q: brain.ask(q))
        self.data_dir = data_dir or getattr(
            getattr(brain, "cfg", None), "data_dir", None)
        self.session = tui_sessions.new_session()
        self.editor = LineEditor()
        self.history = History()
        self.transcript = Transcript(on_change=self._save_session)
        self.completions: list[str] = []
        self.comp_idx: int = 0
        self.running = True
        # BBS: launch lands on the Main Menu. None = AI door (typing
        # asks CETI). /menu returns here from anywhere.
        self.menu: str | None = "main"
        self._esc = False  # Alt-Enter lookahead: ESC then Enter
        # streaming state (see _pump_stream)
        self._stream = None
        self._stream_tag_idx = None
        self._stream_buf = ""      # revealed text, not yet in transcript
        self._stream_reveal = ""   # block remainder, revealed at SIM_TPS
        self._stream_reveal_rate = 0.0  # chars/sec
        self._stream_carry = 0.0
        self._stream_end_args = None
        self._stream_last_pump = 0.0
        fp = getattr(getattr(brain, "identity", None),
                     "fingerprint", "?")
        self.transcript.add(
            f"CETI // {fp} — BBS. Door 1 is the AI. "
            f"/menu main, /ai chat, /help keys, /quit leaves.")

    # ---- sessions ---------------------------------------------------
    def _save_session(self) -> None:
        if self.data_dir is None:
            return
        self.session.lines = list(self.transcript.lines)
        tui_sessions.save_session(self.data_dir, self.session)

    def _open_session(self, session) -> None:
        """Swap the visible transcript to a session (new or
        resumed). The transcript list is replaced, not appended."""
        self.session = session
        self.transcript.lines = list(session.lines)
        self.transcript.offset = 0

    # ---- BBS menus ----------------------------------------------------
    def _choose_menu(self, text: str) -> bool:
        """Handle one line typed at a menu prompt. Returns True if
        the line was consumed as a menu choice. Menu keys ("0".."7")
        come from MENUS — on main "0" is Goodbye, on a submenu "0"
        is Back. Slash escapes (/quit, /menu, /ai, any /command)
        work from any menu."""
        raw = text.strip()
        if not raw:
            return True
        low = raw.lower()
        if low in ("/quit", "q", "goodbye"):
            self.running = False
            return True
        if low in ("/menu", "menu"):
            self.menu = "main"
            return True
        if low in ("/ai", "ai"):
            self.menu = None
            self.transcript.add("(AI door — type to ask. /menu to go back.)")
            return True
        if self.menu is None:
            return False
        _title, options = MENUS[self.menu]
        for key, _label, action in options:
            if raw == key:
                kind = action[0]
                if kind == "menu":
                    self.menu = action[1]
                elif kind == "ai":
                    self.menu = None
                    self.transcript.add(
                        "(AI door — type to ask. /menu to go back.)")
                elif kind == "quit":
                    self.running = False
                elif kind == "cmd":
                    self.menu = None
                    self.transcript.add(f"you> {action[1]}")
                    self._run_command(action[1])
                    self.transcript.add(
                        "(type /menu to return to the Main Menu)")
                return True
        # Not a menu key: a slash command runs and leaves the menu
        # (output shows in the transcript); anything else complains.
        if raw.startswith("/"):
            self.menu = None
            self.transcript.add(f"you> {raw}")
            self._run_command(raw)
            self.transcript.add(
                "(type /menu to return to the Main Menu)")
            return True
        self.transcript.add(
            f"(unknown choice '{raw}' — type a number, /menu, /ai, /quit)")
        return True

    # ---- completion -------------------------------------------------
    def _update_completions(self) -> None:
        text = self.editor.text
        if text.startswith("/") and "\n" not in text:
            word = text.split()[0] if text.split() else "/"
            self.completions = complete_command(word)
            self.comp_idx = 0
        else:
            self.completions = []

    def _apply_completion(self) -> None:
        if self.completions:
            self.editor.set_text(self.completions[self.comp_idx] + " ")
            self.completions = []

    # ---- answering --------------------------------------------------
    def _submit(self, text: str) -> None:
        """Send input: slash commands run inline, questions stream
        (or run synchronously when the brain has no stream_tokens,
        e.g. in tests). A submit while streaming cancels the old
        answer first — interrupt-and-redirect."""
        self._cancel_stream()
        self.history.add(text)
        self.transcript.add(f"you> {text}")
        if text.strip() == "/quit":
            self.running = False
            return
        if text.strip().startswith("/"):
            self._run_command(text.strip())
            return
        stream_fn = getattr(self.brain, "stream_tokens", None)
        if stream_fn is not None:
            self._stream = stream_fn(text)
            self._stream_last_pump = time.monotonic()
            return
        try:
            res = self.ask_fn(text)
        except KeyboardInterrupt:
            self.transcript.add("(cancelled)")
            return
        except Exception as e:  # never die on an answer
            self.transcript.add(f"error: {e}")
            return
        tag = getattr(res, "display_tag", "")
        body = getattr(res, "text", str(res))
        self.transcript.add(f"{tag}\n{body}" if tag else body)

    # ---- streaming --------------------------------------------------
    def _streaming(self) -> bool:
        return (self._stream is not None
                or self._stream_end_args is not None
                or bool(self._stream_buf or self._stream_reveal))

    def _pump_stream(self) -> None:
        """Advance the active stream by one event (or one reveal
        slice). Called from run() each frame while streaming."""
        now = time.monotonic()
        dt = max(0.0, now - self._stream_last_pump)
        self._stream_last_pump = now
        # reveal a slice of a pending block at the capped rate
        if self._stream_reveal:
            budget = self._stream_reveal_rate * dt + self._stream_carry
            take = min(len(self._stream_reveal), int(budget))
            self._stream_carry = budget - take
            self._stream_buf += self._stream_reveal[:take]
            self._stream_reveal = self._stream_reveal[take:]
            if self._stream_reveal:
                return  # still revealing; don't pull the generator yet
            if self._stream_end_args is not None:
                self._finish_stream()
                return
        if self._stream is None:
            return
        try:
            ev = next(self._stream)
        except StopIteration:
            self._stream = None
            if (self._stream_end_args is not None
                    and not self._stream_reveal):
                self._finish_stream()
            return
        kind, payload = ev
        if kind == "heartbeat":
            return
        if kind == "tag":
            self._stream_tag_idx = len(self.transcript.lines)
            self.transcript.add(payload)
        elif kind == "chunk":
            self._stream_buf += payload
        elif kind == "block":
            self._stream_reveal = payload
            words = max(1, len(payload.split()))
            # chars/sec that equals SIM_TPS words/sec for this text
            self._stream_reveal_rate = SIM_TPS * len(payload) / words
            self._stream_carry = 0.0
        elif kind == "stream_end":
            self._stream_end_args = payload
            if not self._stream_reveal:
                self._finish_stream()
        elif kind == "error":
            self._stream = None
            if self._stream_buf or self._stream_reveal:
                self.transcript.add(
                    self._stream_buf + self._stream_reveal)
            self._stream_buf = ""
            self._stream_reveal = ""
            self._stream_end_args = None
            self.transcript.add(f"error: {payload}")

    def _finish_stream(self) -> None:
        """Stream done: finish on the main thread (sqlite lives
        here), swap the provisional tag for the measured one, and
        commit the full text to the transcript (one save)."""
        args = self._stream_end_args
        self._stream = None
        self._stream_end_args = None
        full = self._stream_buf + self._stream_reveal
        self._stream_buf = ""
        self._stream_reveal = ""
        ans = self.brain.finish_stream(args)
        if self._stream_tag_idx is not None:
            self.transcript.lines[self._stream_tag_idx] = ans.display_tag
            self._stream_tag_idx = None
        self.transcript.add(full)

    def _cancel_stream(self, keep_partial: bool = True) -> None:
        """Stop the active stream. Partial text is kept in the
        transcript (redirect or Ctrl-C); the answer is not
        learned, and the provisional tag stays as-is."""
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if keep_partial and (self._stream_buf or self._stream_reveal):
            self.transcript.add(self._stream_buf + self._stream_reveal)
        self._stream_buf = ""
        self._stream_reveal = ""
        self._stream_end_args = None
        self._stream_tag_idx = None

    def _run_command(self, text: str) -> None:
        """Slash commands that stay inside the TUI. Anything else
        prints a pointer to the line REPL."""
        parts = text.split(None, 1)
        cmd, arg = parts[0], (parts[1] if len(parts) > 1 else "")
        if cmd == "/help":
            self.transcript.add(
                "keys: Enter send · Alt-Enter newline · Tab complete\n"
                "      Up/Down history or completions · Ctrl-C cancel\n"
                "bbs: /menu main menu · /ai AI door (typing asks)\n"
                "     1-7 pick a door · 0 back / goodbye\n"
                "chats: /new fresh chat · /sessions list saved\n"
                "       /resume <id> reopen · /delete <id> remove\n"
                "doors: /board /mail /files /identity /peers\n"
                "       /stats /evals /bus /benchmark /tiers\n"
                "/quit leaves the TUI")
        elif cmd == "/menu":
            self.menu = "main"
            self.transcript.add("(Main Menu — pick a door)")
        elif cmd == "/ai":
            self.menu = None
            self.transcript.add("(AI door — type to ask. /menu to go back.)")
        elif cmd == "/evals":
            from . import evals as _ev
            _suite = arg.strip() or "all"
            if _suite not in ("all", "knowledge", "coding", "tool"):
                self.transcript.add("usage: /evals [all|knowledge|coding|tool]")
                return
            self.transcript.add("evals running — coding hits the local model...")
            try:
                _rep = _ev.EvalRunner(
                    self.brain, gap_log=getattr(self.brain, "gap_log", None)
                ).run(None if _suite == "all" else _suite)
                for _name, _d in _rep["suites"].items():
                    self.transcript.add(
                        f"  {_name}: {_d['passed']}/{_d['total']} ({_d['accuracy']:.0%})")
                self.transcript.add(
                    f"  overall: {_rep['passed']}/{_rep['total']} "
                    f"({_rep['accuracy']:.0%}) in {_rep['elapsed_s']}s")
                for _r in _rep["results"]:
                    _mark = "PASS" if _r["passed"] else "FAIL"
                    self.transcript.add(f"  [{_mark}] {_r['id']}: {_r['detail'][:120]}")
            except Exception as e:
                self.transcript.add(f"evals error: {e}")
        elif cmd == "/bus":
            from . import bus as _bus
            _parts = text.split()
            _sub = _parts[1] if len(_parts) > 1 else "post"
            _q = " ".join(_parts[2:]) if len(_parts) > 2 else ""
            try:
                from . import ingest as _ing
                try:
                    _lib = _ing.Library(self.brain.cfg.data_dir)
                except Exception:
                    _lib = None
                _b = _bus.MemoryBus.from_brain(self.brain, library=_lib)
                try:
                    if _sub == "post":
                        import json as _json
                        self.transcript.add(_json.dumps(_b.post(), indent=1))
                    elif _sub == "read":
                        if not _q:
                            self.transcript.add("usage: /bus read <question>")
                        else:
                            _r = _b.read(_q)
                            for _f in _r.frames:
                                self.transcript.add(f"  [{_f.store}] {_f.text[:300]}")
                            for _g in _r.gaps:
                                self.transcript.add(f"  [gap] {_g.question}")
                    elif _sub == "compile":
                        if not _q:
                            self.transcript.add("usage: /bus compile <topic>")
                        else:
                            _c = _b.compile(_q, intent="article")
                            self.transcript.add(
                                f"  local coverage: {_c.local_coverage:.0%} "
                                f"frames={len(_c.frames)} gaps={len(_c.gaps)}")
                            for _g in _c.gaps:
                                self.transcript.add(f"  [gap] {_g.question}")
                    else:
                        self.transcript.add("usage: /bus post|read|compile")
                finally:
                    if _lib is not None:
                        _lib.close()
            except Exception as e:
                self.transcript.add(f"bus error: {e}")
        elif cmd == "/stats":
            try:
                import json as _json
                self.transcript.add(_json.dumps(
                    getattr(self.brain, "stats", {}), indent=1)[:1500])
            except Exception as e:
                self.transcript.add(f"stats error: {e}")
        elif cmd == "/tiers":
            try:
                from . import decide as _dec
                for _intent in ("smalltalk", "command", "factual", "chat"):
                    self.transcript.add(
                        f"  {_intent}: {', '.join(_dec.plan(_intent, self.brain.cfg))}")
            except Exception as e:
                self.transcript.add(f"tiers error: {e}")
        elif cmd == "/benchmark":
            try:
                from . import benchmark as _bm
                import json as _json
                self.transcript.add("self-exam running...")
                self.transcript.add(_json.dumps(
                    _bm.run(self.brain, store=getattr(self.brain, "gap_log", None)),
                    indent=1)[:2000])
            except Exception as e:
                self.transcript.add(f"benchmark error: {e}")
        elif cmd == "/mail":
            try:
                from .mail import MailBox
                _ident = getattr(self.brain, "identity", None)
                if _ident is None:
                    self.transcript.add("no identity: ecdsa not installed")
                else:
                    _valid, _rej = MailBox(self.brain.cfg.data_dir).inbox(
                        _ident.fingerprint)
                    for _i, _m in enumerate(_valid, 1):
                        self.transcript.add(
                            f"  {_i}. [{_m['from']['fingerprint']}] "
                            f"{_m['subject']}")
                    if _rej:
                        self.transcript.add(
                            f"  ! {len(_rej)} rejected (bad signature)")
                    if not _valid:
                        self.transcript.add("  (inbox empty)")
            except Exception as e:
                self.transcript.add(f"mail error: {e}")
        elif cmd == "/files":
            try:
                from . import tools as _tools
                self.transcript.add(_tools.execute("list_dir", path=".").content)
            except Exception as e:
                self.transcript.add(f"files error: {e}")
        elif cmd == "/library":
            try:
                from . import ingest as _ing
                _lib = _ing.Library(self.brain.cfg.data_dir)
                try:
                    _docs = _lib.docs()
                    for _d in _docs:
                        self.transcript.add(
                            f"  {_d['filename']}: {_d['chunks']} chunks")
                    self.transcript.add(
                        f"  total: {_lib.count_chunks()} chunks")
                    if not _docs:
                        self.transcript.add("  (library empty — /ingest in the REPL)")
                finally:
                    _lib.close()
            except Exception as e:
                self.transcript.add(f"library error: {e}")
        elif cmd == "/new":
            self._save_session()
            self._open_session(tui_sessions.new_session())
            self.transcript.add(f"(new chat — {self.session.id})")
        elif cmd == "/sessions":
            if not self.data_dir:
                self.transcript.add("(sessions unavailable: no data dir)")
                return
            found = tui_sessions.list_sessions(self.data_dir)
            if not found:
                self.transcript.add("  (no saved chats yet)")
            for s in found:
                mark = " *" if s["id"] == self.session.id else ""
                self.transcript.add(
                    f"  {s['id']}  {s['title']} "
                    f"({s['n_lines']} lines){mark}")
        elif cmd == "/resume":
            if not arg:
                self.transcript.add("(usage: /resume <id>)")
                return
            if not self.data_dir:
                self.transcript.add("(sessions unavailable: no data dir)")
                return
            sid = tui_sessions.resolve_id(self.data_dir, arg)
            if sid is None:
                self.transcript.add(
                    f"(no unique session matches '{arg}')")
                return
            loaded = tui_sessions.load_session(self.data_dir, sid)
            if loaded is None:
                self.transcript.add(f"(could not load '{arg}')")
                return
            self._save_session()
            self._open_session(loaded)
            self.transcript.add(f"(resumed — {loaded.title})")
        elif cmd == "/delete":
            if not arg:
                self.transcript.add("(usage: /delete <id>)")
                return
            if not self.data_dir:
                self.transcript.add("(sessions unavailable: no data dir)")
                return
            sid = tui_sessions.resolve_id(self.data_dir, arg)
            if sid is None:
                self.transcript.add(
                    f"(no unique session matches '{arg}')")
                return
            tui_sessions.delete_session(self.data_dir, sid)
            if sid == self.session.id:
                self._open_session(tui_sessions.new_session())
                self.transcript.add("(deleted — new chat)")
            else:
                self.transcript.add(f"(deleted {sid})")
        elif text == "/tui":
            self.transcript.add("(already in the TUI)")
        elif text == "/gopher":
            # Gopher menu IS the BBS Main Menu now — same doors.
            self.menu = "main"
            self.transcript.add("(Gopher menu — Main Menu, pick a door)")
        elif text == "/identity":
            ident = getattr(self.brain, "identity", None)
            self.transcript.add(
                f"fingerprint: {ident.fingerprint}\n"
                f"node_id:     {ident.node_id}" if ident
                else "no identity: ecdsa not installed")
        elif text == "/peers":
            from .peers import PeerBook
            plist = PeerBook(self.brain.cfg.data_dir).list()
            for p in plist:
                self.transcript.add(f"  {p['fingerprint']}")
            if not plist:
                self.transcript.add("  (no peers yet)")
        elif cmd == "/board":
            from .board import Board
            _b = Board(self.brain.cfg.data_dir)
            if arg.startswith("read"):
                _n = arg.split()[1] if len(arg.split()) > 1 else ""
                try:
                    _num = int(_n)
                except ValueError:
                    self.transcript.add("usage: /board read <n>")
                    return
                _post = _b.read(_num)
                if _post is None:
                    self.transcript.add(f"(no post {_num})")
                else:
                    self.transcript.add(
                        f"  [{_post['author']['fingerprint']}] "
                        f"{_post['subject']}\n  {_post['body'][:1500]}")
                return
            valid, rejected = _b.list()
            for i, p in enumerate(valid, 1):
                self.transcript.add(
                    f"  {i}. [{p['author']['fingerprint']}] "
                    f"{p['subject']}")
            if rejected:
                self.transcript.add(
                    f"  ! {len(rejected)} rejected (bad signature)")
            if not valid:
                self.transcript.add("  (board empty)")
        elif text == "/quit":
            self.running = False
        else:
            self.transcript.add(
                f"{text}: use the line REPL for this command "
                f"(type /quit, then run it there)")

    # ---- key handling -----------------------------------------------
    def handle_key(self, key: int) -> None:
        """One keystroke. Keys: -1 none, 10/13/343 Enter (send),
        27 Alt (lookahead: Alt-Enter inserts a newline), 9 Tab,
        263 Backspace, 330 Delete, 259 Up, 258 Down, 260 Left,
        261 Right, 1 Home, 5 End, 3 Ctrl-C (raw mode). In cbreak
        mode Ctrl-C arrives as KeyboardInterrupt instead — the
        run loop converts it to a cancel."""
        if key == -1:
            return
        if key == 410:  # KEY_RESIZE: terminal resized — next render
            # reflows to the new size; nothing else to do.
            return
        if self._esc:
            self._esc = False
            if key in (10, 13, 343):
                self.editor.newline()
                self._update_completions()
            return
        if key == 27:  # Alt-...: wait for the next key
            self._esc = True
            return
        if key == 3:  # Ctrl-C: abandon a running answer
            if self._streaming():
                self._cancel_stream()
            self.transcript.add("(cancelled)")
            return
        if key in (343, 13, 10):  # Enter: send
            text = self.editor.text
            self.editor.clear()
            self.completions = []
            if not text.strip():
                return
            # BBS: at a menu, Enter picks a door; in the AI door,
            # Enter asks. History records the line either way.
            if self.menu is not None:
                self.history.add(text)
                self._choose_menu(text)
                return
            self._submit(text)
            return
        if key == 9:  # Tab: complete
            self._apply_completion()
            return
        if key == 259:  # Up
            if self.completions:
                self.comp_idx = (self.comp_idx - 1) % len(
                    self.completions)
            else:
                self.editor.set_text(
                    self.history.prev(self.editor.text))
            return
        if key == 258:  # Down
            if self.completions:
                self.comp_idx = (self.comp_idx + 1) % len(
                    self.completions)
            else:
                self.editor.set_text(
                    self.history.next(self.editor.text))
            return
        if key == 260:
            self.editor.move_left()
            return
        if key == 261:
            self.editor.move_right()
            return
        if key == 1:
            self.editor.move_home()
            return
        if key == 5:
            self.editor.move_end()
            return
        if key in (263, 127):
            self.editor.backspace()
            self._update_completions()
            return
        if key == 330:
            self.editor.delete()
            self._update_completions()
            return
        if 32 <= key < 0x10FFFF:
            try:
                self.editor.insert(chr(key))
            except (ValueError, OverflowError):
                pass
            self._update_completions()

    # ---- rendering --------------------------------------------------
    def render(self, screen) -> None:
        try:
            screen.erase()
            height, width = screen.getmaxyx()
            if height < 4 or width < 20:
                return

            def put(y, x, s, attr=0):
                # SAFE put — the Windows overflow fix. Never write
                # the last column (x + len(s) >= width scrolls or
                # raises on windows-curses), never write outside
                # the screen, never write an empty string at the
                # bottom-right cell. 3-arg addstr when colour is
                # off: keeps the minimal screen contract (old
                # fakes take 3 args).
                if y < 0 or y >= height or x < 0 or x >= width:
                    return
                avail = width - x - 1  # leave the last column empty
                s = s[:max(0, avail)]
                if not s:
                    return
                if attr:
                    try:
                        screen.addstr(y, x, s, attr)
                    except TypeError:
                        screen.addstr(y, x, s)
                else:
                    screen.addstr(y, x, s)

            fp = getattr(getattr(self.brain, "identity", None),
                         "fingerprint", "?")
            header = f" CETI // {fp} // {self.session.title} "
            put(0, 0, header.ljust(width),
                _ATTRS.get("header", 0))
            # pane: menu (BBS) or transcript (AI door / output)
            input_h = max(3, len(self.editor.lines) + 1)
            pane_h = height - 1 - input_h - 1
            if self.menu is not None:
                # Menu + the last transcript lines above it (context:
                # unknown-choice notes, command output) — menu last
                # so it stays visible at the bottom of the pane.
                ctx = self.transcript.render(
                    max(0, pane_h - len(menu_lines(self.menu)) - 1),
                    width)
                lines = ctx + menu_lines(self.menu)
                lines = wrap_lines(lines, width)[-pane_h:]
            else:
                lines = self.transcript.render(pane_h, width)
            if self._streaming():
                # in-progress answer, below the tag line, with a
                # live cursor — not in the transcript yet, so a
                # cancel leaves no half-saved text behind
                buf_lines = (self._stream_buf + "▍").split("\n")
                lines = (lines + wrap_lines(buf_lines, width))
                lines = lines[-pane_h:]
            for i, ln in enumerate(lines):
                put(1 + i, 0, ln, attr_for(ln))
            # input box
            y0 = height - input_h - 1
            put(y0, 0, "─" * width, _ATTRS.get("system", 0))
            _prompt = "choice> " if self.menu is not None else "you> "
            for i, ln in enumerate(self.editor.lines):
                put(y0 + 1 + i, 0, _prompt + ln,
                    _ATTRS.get("user", 0))
            # completion popup
            if self.completions:
                for i, c in enumerate(self.completions[:5]):
                    marker = ">" if i == self.comp_idx else " "
                    attr = _ATTRS.get(
                        "popup_sel" if i == self.comp_idx else "system",
                        0)
                    put(y0 + 1 + len(self.editor.lines) + i,
                        2, f"{marker} {c}", attr)
            # footer
            _foot = (" Enter pick · /ai chat · /menu · /quit leaves."
                     if self.menu is not None else
                     " Enter send · Alt-Enter newline · /menu · "
                     "Tab complete · Up/Down history · "
                     "Ctrl-C cancel · /quit leaves.")
            put(height - 1, 0, _foot, _ATTRS.get("system", 0))
            try:
                screen.curs_set(1)
            except Exception:
                pass
            screen.refresh()
        except Exception:
            pass

    def run(self, screen) -> None:
        """Curses loop. screen.getch() must not block forever —
        callers set a timeout (real curses: screen.timeout(100))."""
        while self.running:
            self.render(screen)
            if self._streaming():
                try:
                    self._pump_stream()
                except KeyboardInterrupt:
                    # Ctrl-C during a blocking ask: the ask is
                    # abandoned, the shell stays alive
                    self._cancel_stream()
                    self.transcript.add("(cancelled)")
            try:
                key = screen.getch()
            except KeyboardInterrupt:
                # cbreak mode: Ctrl-C raises instead of key 3
                self.handle_key(3)
                continue
            except Exception:
                key = -1
            try:
                self.handle_key(key)
            except KeyboardInterrupt:
                # Ctrl-C during a blocking ask: the ask is
                # abandoned, the shell stays alive
                self.transcript.add("(cancelled)")


def run_tui(brain, ask_fn=None) -> None:
    """Entry point: curses.wrapper around TUI.run. Falls back to
    a plain message when curses is unavailable."""
    try:
        import curses
    except ImportError:
        print("tui needs curses (stdlib on unix; not on this build)")
        return

    def _main(stdscr) -> None:
        try:
            stdscr.timeout(100)
            curses.curs_set(1)
            # Windows-safe: never auto-scroll on bottom-right writes,
            # and decode arrow/function keys (keypad) so Up/Down,
            # resize (KEY_RESIZE), and Alt-Enter arrive as codes.
            try:
                stdscr.scrollok(False)
            except Exception:
                pass
            try:
                stdscr.keypad(True)
            except Exception:
                pass
        except Exception:
            pass
        init_palette()
        TUI(brain, ask_fn=ask_fn).run(stdscr)

    curses.wrapper(_main)


def main() -> int:
    """python -m ceti.tui — standalone TUI with a real Brain."""
    from .brain import Brain
    brain = Brain()
    try:
        run_tui(brain)
    finally:
        brain.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
