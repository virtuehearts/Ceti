---
name: sandbox-python
description: Run Python in the isolated sandbox, foreground or background.
---
# Sandbox Python

`run_python` / `run_python_bg` execute code in a Linux namespace
sandbox — not on the host. Know the boundaries:

- **No network.** Sockets fail. If you need web data, see the
  `web-fetch` skill.
- **Filesystem:** read-only system (`/usr`, `/lib`, `/bin`),
  tmpfs `/tmp`, writable `/work` — but `/work` is per-run staging
  and vanishes when the run ends. Feed input data by embedding it
  in your code string; get results out with `print()` and save
  them via `write_file`. The sandbox is for compute; the work
  root is for artifacts.
- **Limits:** 256MB RAM, 32 processes, 64KB output cap per
  stream (excess is truncated, flagged `truncated`), wall-clock
  timeout (default 10s foreground, 300s background, hard max
  600s). CPU-heavy loops get killed — chunk the work.
- **Stdlib only.** No pip, no site-packages beyond the stdlib.
  `sqlite3`, `json`, `re`, `hashlib`, `math` are all available.

For long jobs: `run_python_bg(code)` returns a job id. Poll it
with `job_poll(job_id)` — you get output so far while it runs.
Kill runaways with `job_kill(job_id)`. Finished jobs are reaped
from the table automatically.

To keep results: have the code `print()` them, then save the
output with `write_file`. The sandbox is for compute; the work
root is for artifacts.
