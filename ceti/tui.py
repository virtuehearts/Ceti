"""tui — a full-screen terminal interface for CETI, Hermes-style.

Full TUI with multiline editing, slash-command autocomplete,
conversation history, interrupt-and-redirect, and a transcript
pane. stdlib-only: curses, no new dependencies. The line REPL
stays as the fallback when curses is unavailable.

Layout:
  header    CETI + fingerprint + tier status
  transcript  scrollable Q&A / command output (grows up)
  input     multiline editor (Enter sends, Alt-Enter newline)
  footer    key hints

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
)


def complete_command(prefix: str) -> list[str]:
    """Commands matching a typed / prefix, in definition order."""
    return [c for c in COMMANDS if c.startswith(prefix)]


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
        end = len(self.lines) - self.offset
        start = max(0, end - height)
        return [ln[:width] for ln in self.lines[start:end]]


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
            f"CETI // {fp} — /help for keys + commands, "
            f"Ctrl-C cancels, /quit leaves.")

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
                "chats: /new fresh chat · /sessions list saved\n"
                "       /resume <id> reopen · /delete <id> remove\n"
                "node: /identity /peers /board · /gopher in the REPL\n"
                "/quit leaves the TUI")
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
            self.transcript.add("gopher: use the line REPL for the "
                                "menu, or /board /chat /mail here")
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
        elif text == "/board":
            from .board import Board
            valid, rejected = Board(self.brain.cfg.data_dir).list()
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
            if text.strip():
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
            fp = getattr(getattr(self.brain, "identity", None),
                         "fingerprint", "?")
            header = f" CETI // {fp} // {self.session.title} "
            screen.addstr(0, 0, header.ljust(width)[:width])
            # transcript pane: everything between header and input
            input_h = max(3, len(self.editor.lines) + 1)
            pane_h = height - 1 - input_h - 1
            lines = self.transcript.render(pane_h, width)
            if self._streaming():
                # in-progress answer, below the tag line, with a
                # live cursor — not in the transcript yet, so a
                # cancel leaves no half-saved text behind
                buf_lines = (self._stream_buf + "▍").split("\n")
                lines = (lines + [ln[:width] for ln in buf_lines])
                lines = lines[-pane_h:]
            for i, ln in enumerate(lines):
                screen.addstr(1 + i, 0, ln)
            # input box
            y0 = height - input_h - 1
            screen.addstr(y0, 0, "─" * width)
            for i, ln in enumerate(self.editor.lines):
                screen.addstr(y0 + 1 + i, 0,
                              ("you> " + ln)[:width])
            # completion popup
            if self.completions:
                for i, c in enumerate(self.completions[:5]):
                    marker = ">" if i == self.comp_idx else " "
                    screen.addstr(y0 + 1 + len(self.editor.lines) + i,
                                  2, f"{marker} {c}"[:width])
            # footer
            screen.addstr(height - 1, 0,
                          " Enter send · Alt-Enter newline · /new /help · "
                          "Tab complete · Up/Down history · "
                          "Ctrl-C cancel · /quit leaves."[:width])
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
        except Exception:
            pass
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
