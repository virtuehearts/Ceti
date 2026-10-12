# Development

Extending CETI follows the existing seams:

- **A new tier** goes in `ceti/brain.py`'s ladder: write a
  `_tier_<name>` method returning text or `None`, register it in
  `decide.plan`, add the backend to `ceti/backends.py`, and cover
  the fall-through in `tests/test_brain.py`.
- **A new task kind** goes in `ceti/tasks.py` (queue schema) with
  its executor in `ceti/agent.py`'s tick dispatch. Tasks are just
  rows: kind, title, payload JSON, schedule.
- **A new tool** (once the registry lands) is a manifest plus an
  entrypoint — see [Tools & sandbox](tools-sandbox.md).
- **A new check** for the BIOS POST is a `check_<name>` function in
  `ceti/boot.py` returning `Check(name, status, detail)`; add it to
  the list in `boot()`. Core failures refuse to start; warnings
  degrade.

## Conventions

- stdlib-only in the normal path
- degrade gracefully when a backend is missing
- never create files as a boot side effect (except the identity,
  which is the point)
- every new behavior gets a unit test with faked backends

## Glossary

- **Cassette** — a hot-loadable AIML topic file. The unit of
  shareable knowledge.
- **Tier** — one rung of the answer ladder: AIML → memory → local →
  cloud (oracle).
- **Handler** — the local model in handler mode: it presents
  retrieved knowledge, it doesn't freelance.
- **Oracle** — the cloud tier: frontier models used for hard
  answers, distillation, and judging.
- **Release** — a signed, chained, merkle-rooted tarball of
  knowledge. The commit.
- **Merge** — ingesting a peer's bundle: verify, dedup by record
  identity, resolve conflicts by score.
- **Tick** — one agent loop iteration: run the next due task.
- **Cassette rack** — the hot-reloadable collection of cassettes.
