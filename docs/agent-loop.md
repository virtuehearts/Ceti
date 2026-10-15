# Agent loop

The oracle (cloud API) drives tools to complete tasks. This is how
CETI does real work: plan → write → run in the sandbox → observe →
iterate, with the transcript feeding the session log.

## `/task`

```
/task <what the agent should do>
```

Example session:

```
you> /task write a python script that prints the first 20 primes, run it, and save the output to primes.txt
[agent] session 20261014-001523 (openrouter)
wrote the script, ran it in the sandbox, saved output to primes.txt. First 20 primes: 2 3 5 ...
```

The loop needs a cloud backend with an API key (`OPENROUTER_API_KEY`
or `NVIDIA_API_KEY`). Without one it refuses loudly — the local
model is a background compiler, not a tool-driver.

## How it works (`ceti/agent_loop.py`)

`AgentSession` is one task run:

1. The task goes in as a user message with a system prompt
   describing CETI, the tools, and the rules.
2. The oracle replies with text and/or tool calls (OpenAI
   function-calling schemas, from `Tool.to_openai_schema()`).
3. Tool calls execute — `run_python`/`run_python_bg` in the
   namespace sandbox, file tools jailed to the work root —
   and results go back as tool messages.
4. The loop repeats until the oracle answers with no tool calls
   (task done) or `max_steps` (default 25) is reached.
5. Transient cloud failures retry with exponential backoff
   (`ceti/retry.py`); the brain's cloud tier uses the same helper.

## Session memory (local, per session)

Every session gets its own directory:

```
data/sessions/<session_id>/
  messages.jsonl   # every message, appended as it happens
  notes.json       # the agent's own scratch memory
  summaries.log    # when summarization fired
```

The agent has two tools for its memory:

- `record_note(content)` — timestamped note to itself: findings,
  decisions, facts worth keeping.
- `recall_notes(limit)` — read them back.

Notes are plain JSON on local disk. No server, no model, no
network. The REPL can read them directly.

## Context summarization

Long tool sessions can outgrow the context window. When the
estimated token count (chars/4) passes `token_limit` (default
24000), the session compresses itself:

- **User messages are always kept** — intent is never summarized
  away.
- Assistant/tool rounds between user messages are replaced by a
  summary (written by the oracle; if it fails, the raw transcript
  is truncated with a notice).
- The active round is summarized too, so the loop keeps going.

## Background jobs

`run_python_bg` launches sandboxed code without waiting and
returns a job id:

- `job_poll(job_id)` — running state + output so far, or the
  final result. Finished jobs are reaped from the table.
- `job_kill(job_id)` — terminate.

This is how the agent handles long-running code: launch, do
other work, poll. The sandbox's wall-clock deadline still
applies (default 300s for background jobs).

## Security model

- Code runs **only** in the sandbox — never on the host. The
  sandbox has no network, a private filesystem, and rlimit caps.
- File tools are jailed to `CETI_WORK_ROOT` (default
  `data/scratch`). `..` and symlink escapes are refused loudly.
- Unknown tools are an error message to the oracle, not a crash.
- Malformed tool arguments come back as `error: bad arguments`,
  and the oracle can retry.
- The human reviews: the operator runs `/task` and watches the
  session. Nothing is kept without review.
