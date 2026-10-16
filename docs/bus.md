# Unified Memory Bus — CETI as BIOS

CETI is not an app that calls an LLM. It is a BIOS: a thin, fast,
hardware-like layer under everything, and the LLM is just one slow
peripheral on the bus.

## Layers

| Layer | Code | Job |
|---|---|---|
| POST | `ceti/boot.py` | Verify config, persona, AIML, knowledge DB, embedding server, Ollama, cloud keys, signed dataset, topic queue. Refuse to start on core failure. |
| BUS | `ceti/bus.py` | Memory controller. The CPU never touches a store directly. |
| CPU | `ceti/brain.py`, `ceti/agent_loop.py`, `ceti/story.py` | Issue bus opcodes, assemble answers. |
| DMA | `ceti/agent.py` tick / heartbeat researcher | Prefetch + fill memory without the CPU. Curiosity tasks are DMA requests. |
| IRQ | gaps / adjudication | Miss → curiosity interrupt. Low confidence → adjudication interrupt. Bad signature → fault, never displayed. |

## Cache hierarchy

Cheapest first, like a real chip:

- L0 — session registers (caller-held, not a store)
- L1 — AIML, ~0ms (seed + learned + cassettes; cassettes ride inside AIML, priority learned > cassette > seed)
- L2 — memory, vector (768d) or TF-IDF Q&A
- L3 — library, `doc_chunks` from `data/library/`
- L4 — cloud oracle, slowest / most expensive, like disk/network. Last resort only.

## Common shape

Every store answers as a `Frame`:

```python
Frame(text, store, question, confidence, cost, signature, latency_ms, metadata)
```

`cost`: 0 = local, 1 = oracle. `signature`: True = signed/verified, None = n/a.

## Opcodes

| Opcode | Call | Returns |
|---|---|---|
| `READ` | `bus.read(query, intent)` | `BusResult` — every local hit, plus gaps if none hit |
| `COMPILE` | `bus.compile(topic, intent, subquestions)` | `Compilation` — local hits + explicit gaps |
| `WRITE` | `bus.write(frame)` | bool — via the brain's learn gate when attached |
| `PREFETCH` | `bus.prefetch(topic)` | gaps only, for the researcher (DMA) |
| `INVALIDATE` | `bus.invalidate(query)` | 0 — stores are append-only in v1, fail-loud, never silently drop knowledge |

REPL: `/bus post | read <q> | compile <topic>`

`POST` is the BIOS memory map: AIML count, memory count, library chunks, cassette count.

## Compile-first

The old ladder was fall-through: first tier that hit won, the rest
were never asked, and an article either came from local OR went
wholesale to the oracle.

The bus aggregates instead:

1. **Fan-out** — `Bus.compile(topic)` decomposes the topic into
   sub-questions. Default fan-out is deterministic (no LLM, no tokens):
   what / when+who / architecture / specs / vs alternatives / impact.
   Callers that know the domain pass their own sub-questions — e.g.
   for the C64: "How did the C64 fare against the Amiga?" and
   "... against the Atari?"
2. **Read every store per sub-question** — AIML + memory + library,
   deduped, provenance-tagged. No first-hit-wins.
3. **Gaps** — sub-questions no local store answered, returned
   explicitly. The bus never calls the oracle.
4. **Gap-fill only** — the caller (`story.ArticleCompiler`) calls the
   oracle once per gap, each prompted "answer ONLY this question".
   UNKNOWN stays a gap, never prose. Local frames are never sent to
   the oracle as a writing prompt.
5. **Assemble locally** — local frames + gap answers, tagged
   `[aiml]` / `[memory]` / `[library]` / `[oracle gap-fill]`,
   open gaps listed as such. Then write-back via the learn gate.

Example — Commodore 64 article:

- Local hits: 1982, Commodore, MOS 6510, 64K RAM, VIC-II, SID …
- Gaps: C64 vs Amiga, C64 vs Atari, sales figures …
- Oracle: called with ONLY the gaps, answers verified, promoted.
- Next time: the article is 100% local.

Context is a bus parameter: same topic, different intent —
factual Q → AIML first, short; article → library + vector first,
long compile; task → tools + skills first, oracle only for planning.

## Compile-first articles

`/story article <topic> [q1; q2; ...]`

- Backend optional: with no oracle it writes local-only, gaps listed.
- With a backend: one oracle call per gap, never the whole article.
- Output: `data/stories/<slug>-article/article.md` +
  `compilation.json` (frames, gaps, coverage, provenance).

`StoryWriter` (bible → oracle chapters) is unchanged — the article
flow is the bus path, the novel flow is the oracle path.

## Rules

- The bus never calls a model. The caller owns the oracle, its PII
  stripping, and its cost.
- No network in `ceti/bus.py`. No new dependencies. Stdlib only.
- A miss is a gap, not an empty answer. Gaps go to the curiosity
  feed, never to silent fabrication.
- Stores are append-only in v1. Invalidation is a no-op that says so.
