# Skills

Skills are markdown + code bundles the agent can read: reusable
guides for getting things done with CETI's tools. They live in
`skills/`, one directory per skill, hot-loadable like cassettes —
add a directory, `/skills reload`, and the agent knows it.

## Format

Each skill is a directory containing a `SKILL.md`:

```markdown
---
name: sqlite
description: Query sqlite databases with the stdlib sqlite3 module.
---
# SQLite

...the guide...
```

- Frontmatter is a flat `key: value` subset (stdlib-only, no
  PyYAML). Lines starting with `#` are ignored; values may be
  quoted.
- `name` and `description` are required. `name` must be
  lowercase, digits, `-`/`_` only.
- Any other keys are kept as metadata.
- The markdown body is the guide itself — write it for the
  oracle: concrete patterns, real snippets, honest limits.
- Extra files in the directory (scripts, examples) ride along;
  `read_skill` lists them.

Malformed bundles are skipped loudly — recorded in the loader's
`errors`, printed at boot and by `/skills`, never silent.
Duplicate names: first loaded wins, the rest are reported.

## Discovery

The agent finds skills two ways:

- `list_skills` — one-line listing (registered tool).
- `read_skill(name)` — the full guide (registered tool).

The agent loop also injects the skill list into its system
prompt, so the oracle knows what guides exist before it starts.

## REPL

```
/skills           # list loaded bundles
/skills reload    # rescan the skills/ directory
```

`/call list_skills` and `/call read_skill name=sqlite` work too.

## Writing a skill

1. Create `skills/<name>/SKILL.md` with the frontmatter above.
2. Write the guide as if briefing a smart contractor: what the
   tools do, the exact patterns that work, the limits that bite.
   Include short code snippets — the oracle copies them.
3. Be honest about what's *not* available. A skill that says
   "the sandbox has no network, here are your three options"
   beats one that pretends otherwise.
4. Drop in helper scripts or examples as extra files; mention
   them in the guide.
5. `/skills reload`, then `/call read_skill name=<name>` to
   proofread what the agent will see.

## Seed set

- `file-ops` — the four file tools, the jail, the
  list → read → write/edit → verify workflow.
- `sandbox-python` — what the sandbox is and isn't: no
  network, per-run `/work`, stdlib only, limits, background
  jobs for long work.
- `web-fetch` — getting web data given the sandbox has no
  network: ask the operator, the research queue, staged
  material. Rules for if fetching ever becomes available.
- `sqlite` — querying sqlite via stdlib `sqlite3` in the
  sandbox: schema inspection, parameterized queries, getting
  results out.
