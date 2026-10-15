---
name: sqlite
description: Query sqlite databases with the stdlib sqlite3 module in the sandbox.
---
# SQLite

Python's stdlib `sqlite3` works in the sandbox — it's just file
I/O. Patterns via `run_python`:

```python
import sqlite3

# inspect schema first
db = sqlite3.connect("/work/data.db")
print([r[0] for r in db.execute(
    "SELECT name FROM sqlite_master WHERE type='table'")])
print(db.execute("PRAGMA table_info(mytable)").fetchall())

# parameterized queries — never format values into SQL
rows = db.execute(
    "SELECT id, title FROM mytable WHERE score > ? ORDER BY score DESC LIMIT 20",
    (0.5,)).fetchall()
for r in rows:
    print(r)
db.close()
```

Notes:

- The database file must be *inside the sandbox run*. Since
  `/work` is per-run staging, embed small datasets as SQL or
  Python literals in your code rather than expecting a file to
  be there.
- Results leave via `print()` — then save them with
  `write_file` or summarize with `record_note`.
- Writes inside the sandbox vanish with the run. If you need a
  persistent database, say so in your report; the operator
  decides what lives in the work root.
- Keep result sets small (`LIMIT`); the output cap is 64KB per
  stream.
