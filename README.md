# CETI

**A sovereign local brain.**  
The datasets are the mind. The models are only the handlers.

CETI answers from knowledge that lives on your iron first — AIML cassettes, then a local vector memory of curated Q&A. A small local model presents what was retrieved. Cloud models are a last-resort oracle, and even then they receive only a PII-stripped query. Every answer worth keeping is classified, distilled, signed, and returned to the local knowledge base.

No accounts. No telemetry. No landlord. It runs on a tired laptop.

![CETI launch art — pixel art of the brain booting up](docs/images/ceti-launch-v3.webp)

---

## Quick start

```bash
unzip ceti-0.1.0.zip
cd ceti-0.1.0

./install.sh
source ./ceti-env.sh
./run.sh
```

`install.sh` sets up the virtual environment, installs the single hard dependency (`ecdsa`), optionally installs Ollama and pulls the handler model, fetches a `llama-server` binary for embeddings, writes `ceti-env.sh`, and runs the boot self-test.

For a non-interactive or minimal install:

```bash
# Full stack, no prompts
./install.sh --yes

# Core only (AIML + sqlite memory). No Ollama, no models.
./install.sh --yes --no-ollama --no-models

# Bring your own models
./install.sh --yes \
  --embedding-model-url=https://example.com/embeddinggemma-2.gguf \
  --research-model-url=https://example.com/research-model.gguf
```

`run.sh` starts any local services that are present (Ollama, embedding server) and opens the REPL. Missing pieces degrade gracefully; the boot screen reports exactly what is available.

### After boot

```
CETI BIOS v0.1.0 — power-on self test
[  ok  ] config     data_dir=...
[  ok  ] persona    Persona (...)
[  ok  ] aiml       ... categories ...
[  ok  ] knowledge  ...
...
```

Type a question. Useful REPL commands:

| Command      | Purpose                                      |
|--------------|----------------------------------------------|
| `/tiers`     | Show the answer ladder and which tiers are live |
| `/trace`     | How the last answer was produced             |
| `/stats`     | Tier counts, learned totals, uptime          |
| `/identity`  | Node fingerprint and identity file           |
| `/agent`     | Agent name, pending tasks, knowledge gaps    |
| `/benchmark` | Self-exam of current knowledge               |
| `/heartbeat` | Toggle idle research tick                    |
| `/quit`      | Exit                                         |

---

## Requirements

- **Python 3.10+**
- **Linux or macOS** (the researcher’s memory gate is Linux-specific; elsewhere it is skipped)
- ~2 GB disk if you want the full model stack. The brain itself is a few hundred kilobytes.
- RAM: the core runs in tens of megabytes. Research cycles prefer ~1 GB free and will skip if memory is low.

Optional backends (all degrade cleanly):

| Backend              | Purpose                          | Notes                                      |
|----------------------|----------------------------------|--------------------------------------------|
| Ollama               | Local handler model              | Default: `qwen2.5-coder:1.5b` (~1 GB)     |
| embedding GGUF       | Vector memory                    | 768-dim, e.g. embeddinggemma-2. Falls back to sqlite TF-IDF |
| Research model       | Idle research distillation       | Any ~2–3B instruct model registered in Ollama |

Nothing phones home.

---

## How it answers

Every question walks a ladder. The first tier that knows the answer wins.

| Tier   | Backend                                      | Typical latency | Notes                                      |
|--------|----------------------------------------------|-----------------|--------------------------------------------|
| AIML   | Hot-reloadable pattern cassettes             | ~0 ms           | Typo- and paraphrase-tolerant              |
| Memory | Local vector search over curated Q&A         | ~1–3 s          | sqlite TF-IDF fallback if no embeddings    |
| Local  | Small model via Ollama (handler mode)        | ~5–30 s         | Sees the raw query; never leaves the box   |
| Cloud  | OpenRouter / Nvidia (oracle)                 | —               | Receives only the PII-stripped query       |

In handler mode the local model is instructed to present retrieved knowledge rather than generate freely. A deterministic numbers-grounding guard rejects rewrites that invent figures and falls back to the raw retrieved text.

---

## The learning loop

1. The ladder answers.
2. Every generated answer is scored by the relevance classifier:
   - **≥ 0.65** → promote into `learned.aiml` (next time it answers in ~0 ms)
   - **0.45–0.65** → local adjudication
   - **< 0.45** → discard
3. Promoted knowledge accumulates until you cut a signed release.
4. Questions that repeatedly fail become curiosity tasks for the idle researcher.

The idle researcher walks `data/topics.txt` (broad, encyclopedic subjects). It researches, distills into Q&A, classifies, and writes accepted pairs into a topic cassette and the vector store. Completed topics are skipped. Cassettes hot-reload.

---

## Identity and signed knowledge

On first run CETI generates a node identity (`data/identity.dat`, mode 0600). Only the public fingerprint is ever shared. The private key signs dataset releases.

Knowledge ships as signed, chained, merkle-rooted bundles:

1. sha256 per file, merkle root over sorted hashes
2. each manifest carries `prev_hash` (version ledger)
3. ECDSA (deterministic secp256k1) signature over the canonical manifest
4. gzip-packed tarball + pure-stdlib torrent/magnet

Peers verify the signature before merge. No signature, no trust. Merge is by record identity with score-based conflict resolution.

---

## Agent

CETI is a named, persistent agent with a task queue:

- Research, benchmark, curiosity, ingest, and release tasks
- Tasks persist in sqlite and survive restarts
- The REPL heartbeat triggers a tick after ten quiet minutes
- Self-benchmark quizzes every known question (exact, lowercased, and with a real typo)
- Curiosity turns repeated failures into research tasks

---

## Configuration

Environment variables (set in `ceti-env.sh` or exported):

| Variable                     | Default                          | Purpose                          |
|------------------------------|----------------------------------|----------------------------------|
| `CETI_DATA_DIR`              | `./data`                         | Knowledge and identity location  |
| `CETI_FINGERPRINT_PREFIX`    | `CETI`                           | Identity tag (`JERO` for the Jero network) |
| `OPENROUTER_API_KEY`         | (unset)                          | Enables cloud oracle             |
| `NVIDIA_API_KEY`             | (unset)                          | Enables cloud oracle             |
| `CETI_EMBEDDING_MODEL`       | `./models/gguf/embeddinggemma-2.gguf` | Vector tier model           |
| `CETI_LLAMA_SERVER`          | `./bin/llama-server`             | Embedding server binary          |

The cloud tier stays disabled until keys are present.

---

## Project layout

```
ceti/               core package
  agent.py          named agent, task queue, tick loop
  aiml.py           pattern matching and cassettes
  backends.py       Ollama, OpenRouter, Nvidia
  boot.py           power-on self-test
  brain.py          answer ladder and learning loop
  classify.py       relevance classifier
  memory.py         knowledge store
  researcher.py     idle research
  signing.py        identity, ECDSA, merkle
  vectormem.py      local embeddings
data/               persona, seed, learned, cassettes, topics
tests/              unit tests
install.sh          one-shot setup
run.sh              launcher
```

---

## Development

```bash
# Run tests
.venv/bin/python -m unittest discover -s tests

# Boot self-test only
.venv/bin/python -m ceti.boot
```

The brain is stdlib-only except for `ecdsa` (signing). All other dependencies are optional backends.

---

## Security model

- No telemetry, no accounts, no remote control plane.
- Queries to the cloud oracle are PII-stripped first.
- Node private key never leaves the machine.
- Dataset releases are signed; peers reject unsigned or invalid bundles.
- Identity file is created mode 0600.

---

## Roadmap

- Stronger promotion judge so researcher-grown knowledge outranks freelanced local answers
- Multi-signer web of trust and key revocation
- Agent-wired release → torrent → seed → verify → merge
- BBS front end with signed posts and file serving
- Tool registry and sandboxed agent loop

---

## Deeper docs

- [Oracle](docs/oracle.md) — the cloud tier's three jobs and two rules
- [Distribution](docs/distribution.md) — the torrent lifecycle and merge protocol
- [BBS front end](docs/bbs.md) — the three-phase board plan
- [Tools & sandbox](docs/tools-sandbox.md) — the agent's hands
- [Security model](docs/security.md) — the four boundaries
- [Development](docs/development.md) — extending every seam, plus glossary
- [Troubleshooting](docs/troubleshooting.md) — when something's off

---

## License

MIT. See `LICENSE`.
