# Trim: query-aware context compression

CETI's compression technique. Given a context (a tool result, a file, a
log, a long chat) and the current ask, keep the lines the ask needs and
drop the rest. Kept lines come back verbatim, in original order, with
one `… N lines …` marker where lines were removed. Nothing is rewritten
or summarized.

## Why a model that isn't a model

The selection is made by a small embedded gradient-boosted tree model:
712 trees, 1.2 MB, deterministic, milliseconds on one CPU core, no GPU,
no network. It is not a language model — it spends no model tokens, it
just scores lines. That is why it belongs here: it shrinks what the
oracle has to read, and it fits the zero-model-tokens-at-serve-time
design. It is a compiler-tier tool, same class as the calc tier.

Upstream: [trim-1](https://github.com/caveman-ai/trim-1) (Apache-2.0,
see `ceti/LICENSE-trim-1`). `ceti/trim.py` is a faithful Python port
of its Go source, stdlib only. The model file `ceti/trim_model.bin`
is vendored from the same repository.

## Use

```python
from ceti import trim
res = trim.compress(context, query, point="balanced")
print(res.output)          # compressed text, or context if unchanged
print(res.unchanged, res.reason)
```

`trim.score(context, query)` returns every line with its keep
probability, if you want to make the selection yourself.

### Operating points

`point` decides how hard trim cuts: `conservative`, `balanced` (the
default) or `aggressive`. Cutting harder costs quality, because every
removed line is one the model can no longer read. Long-document
retrieval is what it is for — use `balanced`. Coding-agent tool outputs
are short and dense: stay at `balanced` or `conservative` and expect
small savings.

### Fail-closed

Trim returns the context unchanged, and says why, when:

- the point name is unknown,
- the query is blank,
- the context is under 2,000 bytes,
- the context is over 8 MiB or 200,000 lines,
- the context already has a removal-marker line,
- the cut would remove less than 10% of the bytes,
- the model file is missing or corrupt.

## Where CETI uses it

- **Agent loop** (`ceti/agent_loop.py`): before the oracle summarizer
  reads a block of history, trim cuts it to the lines the task needs.
  Fewer oracle tokens, same signal. The block builder used to truncate
  tool results to 500 chars — which hid the tail, where errors live.
  Now each result enters the transcript up to 20,000 chars, trim does
  the real cutting, and an 8,000-char fallback bounds the summarizer
  input when trim fails closed.
- **`trim` tool** (`ceti/tools.py`): the agent can call it to get a
  focused view of a long tool result before working from it.

Deliberately **not** automatic on tool results: a trimmed file is
unsafe to edit precisely. The agent trims for reading; edits always
run against the full file.

## How it works

Each line gets 67 features — BM25 against the query, term coverage,
identifier and bigram/trigram hits, line kinds (error, test, path,
heading, diff, signature, ...), duplication and template counts,
position, indent and block structure, neighbour and window features,
and feedback terms from the best-covering lines. The trees turn the
row into a keep probability. The policy then selects: the top-K lines
plus everything scoring at least `floor` × the best line, ±1 line of
context around each, enclosing block headers, section openers, and
short-gap fill. One marker per removed run.

## Honest limits

- **Prose is kept generously.** The model cuts repetitive log lines
  hard (a 19 KB log compresses to ~400 bytes) but keeps irrelevant
  prose sections — it learned that prose lines are usually worth
  reading. For log-shaped tool output it is excellent; for prose,
  expect modest cuts.
- **The model is upstream's, from 2026.** It has no long track record;
  the benchmark numbers are theirs. The port is verified (below), the
  model's judgment is taken on trust.
- **It selects, never rewrites.** It cannot fix, explain or summarize
  — it only decides which lines the oracle gets to see.

## Verification

`tests/test_trim.py` ports the upstream tests:

- **Parity**: the tree evaluator matches LightGBM's raw scores
  bit-for-bit on `tests/testdata_trim/parity.bin`.
- **Golden**: the full output on the sample log pins to
  `tests/testdata_trim/golden.txt`.
- Refusals, determinism (same input → same output), verbatim
  in-order output, the final-newline rule, the query cap, model-load
  failure.
