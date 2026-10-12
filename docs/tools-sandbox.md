# Tools & sandbox

The agent is growing hands: a **tool/skill system** backed by a
**sandbox** for anything that executes.

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
