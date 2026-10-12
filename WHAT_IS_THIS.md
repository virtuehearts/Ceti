# CETI — what this is meant to be

A separate, low-spec Python "brain": a small decision layer that answers
questions the way its keeper wants machines to think — **the datasets are
the mind; the LLMs are just the handlers.**

## The inversion

Normal assistants: the model IS the mind, knowledge is whatever it
memorized in training. This brain inverts it:

- **Local knowledge answers first** — AIML pattern cassettes (~0ms),
  then a vector memory of curated Q&A, both running on your own iron.
- **The model only presents** — in handler mode, local Qwen is
  instructed to answer strictly from retrieved knowledge. It wields
  the datasets; it does not freelance.
- **Frontier cloud models are the last-resort oracle** — they teach the
  local sets when nothing local knows the answer. Their answers get
  classified, and the good ones are distilled back into local knowledge.

## The tiers (see DIAGRAM.svg)

1. **AIML** — Alice-style pattern cassettes, including hot-loadable
   topic datasets ("cassettes"). Typo-tolerant, paraphrase-aware.
2. **Memory** — vector search (local embeddinggemma-2, 768d) over
   `knowledge.sqlite3`, with sqlite TF-IDF fallback.
3. **Local Qwen** — qwen2.5-coder:1.5b via Ollama. Sees the raw query
   (it never leaves the box).
4. **Cloud** — OpenRouter / Nvidia. Receives only the PII-stripped
   query. Nothing personal ever leaves the machine.

A relevance classifier scores every new answer: ≥0.65 promotes it into
`learned.aiml`, 0.45–0.65 goes to local-Qwen adjudication, below that
it's discarded. The brain only keeps what earns its place.

## The idle researcher

When the box is idle, a researcher works through `data/topics.txt`:
research a topic with the sovereign research stack, distill the report
into Q&A with local Qwen, classify, and write accepted pairs into a
topic cassette + the vector DB. Idle cycles pre-train the cassettes,
Wikipedia-style.

## Signed knowledge sets, seeded by torrent

Knowledge sets ship like blocks, regula_core-style:

1. **hash** — sha256 per file, merkle root over sorted hashes
2. **chain** — each manifest carries `prev_hash`: a version ledger
3. **sign** — node ECDSA (deterministic secp256k1, SHA256d) over the
   canonical manifest. No signature, no seed.
4. **pack** — gzip the bundle; the tarball is the seed artifact
5. **seed** — pure-stdlib `.torrent` + magnet, seeded by archivists on
   connected machines. The signature means no peer trusts the seeder.

## Persona

The brain speaks as Jero himself, in the manner of Khan (Wrath of
Khan): commanding, certain, theatrical — wrath aimed at problems,
never people. That persona is the brain's, not Mya's.

## Running it

```bash
python -m ceti.prototype --ask "who are you"
~/workspace/venvs/jeronode/bin/python -m unittest discover -s tests  # 89 tests
python -m ceti.seedtest <torrent> <tarball>   # loopback seeding proof
bash seedbox/setup-seedbox.sh                  # on the AlmaLinux VPS (as root)
```

Stdlib only in the normal code path. Runs on Jero-node class hardware.
