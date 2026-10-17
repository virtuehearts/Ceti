# Eval loop — coding / tool / knowledge benchmarks

`ceti/benchmark.py` is the self-exam: quiz the brain on its OWN
knowledge (every Q&A pair, three phrasings), accuracy + tier dist +
latency, misses → gaps. `ceti/evals.py` is different: a FIXED,
curated suite that measures CETI the system on the three things it
must be good at. Same gap filing, different target.

## Suites (9 cases, 3 per domain)

| Suite | How it verifies | Why it's trustworthy |
|---|---|---|
| knowledge | `Brain.ask`, expected facts present, tier != failed | Seed AIML answers — deterministic, no LLM |
| coding | `Brain.ask` → extract code block → run in sandbox → harness output | Code runs or it doesn't. No judge, no vibes. |
| tool | Drive the tool registry directly: write→read, `run_python`, jail refusal | The tools either work or they don't. No model in the loop. |

Coding cases: `add(a,b)`, `reverse_string(s)`, `is_even(n)` — each
with a harness (`print(add(2,3))` → `5`, etc.). The runner extracts
the first fenced Python block (or a bare `def`), appends the
harness, runs it via `run_python` (sandboxed), and checks exit 0 +
expected output.

Tool cases: `write_file` → `read_file` round trip, `run_python`
computes 40+2, `read_file("../escape")` must be refused. The jail
test is the point: a tool suite that never tests refusal is theatre.

## Rules

- No network, no cloud, no new dependencies. Stdlib only.
- Evals never learn (`Brain.ask(learn=False)`) — measuring must
  not pollute memory.
- Verifiers are deterministic: substring checks + sandbox exit
  codes, never a model judging a model.
- Every miss is filed to the gap log (curiosity feed) so research
  can close it. No DB writes — the report is ephemeral, the gaps
  persist.

## REPL

```
/evals                 all three suites (~1 min live)
/evals knowledge      knowledge only (AIML-instant)
/evals coding         coding only (3 local-model calls)
/evals tool           tool only (registry, no model)
```

Output: per-suite accuracy, overall accuracy, per-case PASS/FAIL +
detail, misses filed as gaps.

## The loop that makes CETI smarter

```
propose -> verify (sandbox / tests / judge) -> promote if good,
gap if bad -> research fills gaps -> /evals measures the whole
thing
```

If coding accuracy goes up after a change, CETI got smarter. If
it goes down, the change regressed something — the evals catch it
before the user does. Run `/evals` after any change to the bus,
the handler prompt, the tool registry, or the model. That is the
"smarter" metric: not a bigger model, a system where every answer
either becomes verified local knowledge or becomes a known gap.
