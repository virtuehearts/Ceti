---
name: file-ops
description: Patterns for reading, writing, and editing files in the work root.
---
# File operations

You have four file tools, all jailed to the work root (`CETI_WORK_ROOT`,
default `data/scratch`). Paths are relative to the work root. `..`
and symlink escapes are refused — don't try to be clever, it fails
loudly.

- `list_dir(path=".")` — survey before you act. Lists up to 200
  entries with sizes; directories get a trailing `/`.
- `read_file(path)` — read a text file (max 512KB).
- `write_file(path, content)` — create or overwrite (max 2MB).
  Parent directories are created automatically.
- `edit_file(path, old, new)` — replace the FIRST occurrence of the
  exact string `old` with `new`. Fails loudly if `old` is not found,
  so read the file first and copy the exact text.

Workflow: list → read → write/edit → read back to verify. Never
assume a write landed; read the file after. For multi-step edits,
prefer several small `edit_file` calls over one big rewrite — each
one is verified or fails loudly.
