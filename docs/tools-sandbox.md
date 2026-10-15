# Tools & sandbox

The agent is growing hands: a **tool/skill system** backed by a
**sandbox** for anything that executes.

## Status: v1 built

**Sandbox** (`ceti/sandbox.py`) — pure-Python Linux namespaces via
ctypes. No root, no daemon, no host modifications, near-zero
overhead:

- mount namespace: `/usr /lib /lib64 /bin` read-only, tmpfs `/tmp`,
  private writable `/work`; nothing propagates to the host
- user namespace: guest drops to uid 65534 when mapped (host root);
  otherwise the kernel locks unprivileged mounts read-only
- network namespace: empty, loopback down — no network, ever
- PID namespace: the guest is PID 1, sees no host processes
- rlimits: memory, CPU seconds, process count, file size; wall-clock
  kill as backstop

Verified by `tests/test_sandbox.py`: hello-world, no host write,
read-only binds hold (remount attack fails), no network, memory cap,
timeout, CPU-hog kill, PID 1, output truncation.

**Tools** (`ceti/tools.py`) — registry with manifests (name,
description, args, side effects, sandboxed flag):

| Tool         | Side effects | Runs in |
|--------------|--------------|---------|
| `run_python` | none         | sandbox |
| `read_file`  | read         | host    |
| `write_file` | write        | host    |
| `edit_file`  | write        | host    |
| `list_dir`   | read         | host    |
| `trim`       | none         | host    |

`trim` compresses text to the lines the current ask needs (see
[Trim](trim.md)): verbatim, in order, with markers where lines were
removed. Best on logs and long tool output; fails closed.

Execution rule: the agent **never** runs code on the host —
`run_python` is the only execution tool and it is always sandboxed.
File tools are jailed to the work root (`CETI_WORK_ROOT`, default
`data/scratch`): `..` escapes and symlink escapes are refused.
`/tools` in the REPL lists the registry.

## Still to come

- QEMU microVM backend behind the same `SandboxPolicy`
  (stronger isolation, same contract)
- Skill bundles in `skills/` (markdown + code, hot-loadable)
- The agent loop: plan → write code → run in sandbox → observe →
  iterate, as a task kind

## Tool registry

Tools declare a manifest — name, argument schema, side effects,
sandbox policy. The agent can only invoke registered tools;
unregistered code doesn't run.

## Skills

Markdown + code bundles in `skills/`, hot-loadable like cassettes.
A seed set is planned: file operations, web fetch through the
research stack, sqlite queries — plus docs for writing your own.

## Sandbox

A minimal QEMU microVM is the *only* place agent code runs:

- snapshot/restore per invocation
- no network by default
- CPU/RAM caps, scratch wiped after
- the agent never executes on the host

## The loop

Plan → write code → run in the sandbox → observe the result →
iterate, with the transcript feeding the task log. Coding becomes a
task kind like any other: queued, scheduled, retried, benchmarked.
