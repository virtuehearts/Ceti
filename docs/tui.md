# TUI — full-screen terminal interface, Hermes-style

`python -m ceti.tui`, or `/tui` from the line REPL. The line
REPL stays as the fallback when curses is unavailable.

## Layout

```
 CETI // <fingerprint>
 <transcript — scrollable Q&A / command output>
 ───────────────
 you> <input>
 Enter send · Alt-Enter newline · / commands · Tab complete ·
 Up/Down history · Ctrl-C cancel · /quit leave
```

## What it does (Hermes parity, CETI-shaped)

- **Multiline editing** — Enter sends, Alt-Enter inserts a
  newline. Arrow keys, Home/End, Backspace/Delete all work.
- **Slash-command autocomplete** — typing `/` opens a popup
  of matching commands under the input. Tab completes,
  Up/Down picks, Enter accepts.
- **Conversation history** — Up/Down (outside the popup)
  cycles previous inputs.
- **Interrupt-and-redirect** — Ctrl-C abandons a running
  answer and drops back to input. Answers run on the main
  thread (Brain's sqlite handles are thread-affine), so
  Ctrl-C raises KeyboardInterrupt inside the ask and the
  shell carries on.
- **Transcript pane** — every question, answer, and command
  output lands in the scrollable transcript.

## Streaming

Questions stream instead of blocking. Model tiers
(local Qwen, oracle) yield real token chunks as they
arrive; retrieval tiers (calc, AIML, memory) yield one
block, revealed at 499 t/s — the same cap `Answer` uses
for its display tag, so the tag and the reveal agree.
The generator is pumped on the main thread (Brain's
sqlite handles are thread-affine); the backend reads the
SSE stream on a daemon thread and feeds a queue, so keys
stay responsive between chunks.

- Ctrl-C cancels the stream and keeps the partial answer
  (not learned).
- Enter while streaming cancels the old answer and starts
  the new one (interrupt-and-redirect).
- The memory tier buffers its handler output and checks
  grounding before anything is shown — an ungrounded
  handler answer never reaches the screen.
- A mid-stream backend failure is terminal: the partial
  text stays, `(error: ...)` is shown, no finish runs.
  A failure before the first chunk falls through to the
  next tier, like `ask()`.

`/identity`, `/peers`, `/board`, `/gopher` (pointer),
`/quit`, `/tui`. Everything else prints a pointer to the
line REPL — the TUI is the asking surface, the REPL keeps
the full command set. Questions go through the same
`Brain.ask` as the line REPL.

## Sessions — new chats, like Hermes

Every chat is a saved session: one JSON file per session
under `data/tui-sessions/`, titled from its first question.

- `/new` — save the current chat, start a fresh one.
- `/sessions` — list saved chats (newest first, `*` marks
  the current one).
- `/resume <id>` — reopen a chat. A unique prefix of the
  id works; the full id always does.
- `/delete <id>` — remove a chat. Deleting the current
  chat starts a fresh one.
- `/help` — keys + commands, in the transcript.

Launching the TUI always starts a fresh chat; resuming is
explicit. A chat with no real question is never written —
an empty chat (or one that only ran commands) leaves no
file behind. The header shows the current chat's title.
Sessions are separate from the agent loop's
`data/sessions/` (tool-driving runs, different format).

## Honest limits

- Answers arrive as blocks: `Brain.ask` returns a whole
  answer, so the TUI prints it whole. Streaming token-by-
  token output is a future pass, not this one.
- The TUI needs a real terminal (`TERM` set, curses
  available). No `TERM`, no TUI — it prints a message and
  the line REPL carries on.
- Tests drive the loop with a fake screen and a scripted
  key queue (`tests/test_tui.py`), plus a real pty smoke
  test during development. The pure logic (editor, history,
  completion, transcript) is fully unit-tested.

## BBS menus (Hermes-style)

Launch lands on the Main Menu — numbered doors, type a number
at the `choice>` prompt. The AI is door 1: typing there asks
CETI, inside the TUI. No need to leave the TUI to ask, check
the board, mail, node, or system.

```
 CETI // MAIN MENU
 [1] AI — Ask CETI (typing here asks the model)
 [2] Board — Read signed posts
 [3] Mail — Inbox
 [4] Files — Work root / drop
 [5] Node — Peers, identity, nodelist
 [6] System — Stats, evals, bus, benchmark
 [7] Library — Ingest / research status
 [0] Goodbye — Leave the BBS
```

Each door has a submenu (Board: list / read, Node: identity /
peers / board, System: stats / evals / bus / benchmark / tiers,
etc.). `0` is Back on a submenu, Goodbye on Main. `/menu`
returns to Main from anywhere, `/ai` goes to the AI door,
`/quit` leaves. Slash commands (`/board`, `/evals`, `/bus`,
`/stats`, `/mail`, ...) work from any menu and leave the menu
(output shows in the transcript); `/menu` brings the menu back.

Still in the line REPL only: `/task`, `/story`, `/share`,
`/download`, live `/chat`, `/ingest` — multi-step / backend
doors that have not moved into the TUI yet.

## Windows overflow fix

curses on Windows (windows-curses / PDCurses) scrolls or raises
when text is written to the last column — writing exactly
`width` chars at x=0, or anything into the bottom-right cell.
The old render swallowed the error, leaving text hanging off
the terminal. Two rules now, everywhere:

1. Wrap, never truncate — long lines wrap to width-1 with word
   boundaries (long tokens broken). Tabs expanded, CR / control
   chars stripped, before wrapping.
2. Never write the last column — `put()` trims to width-x-1
   and skips out-of-bounds rows. `scrollok(False)` +
   `keypad(True)` in `run_tui`; KEY_RESIZE (410) just re-renders
   (reflow to the new size).
