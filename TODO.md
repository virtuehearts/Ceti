# Brain TODO — honest list, 2026-10-12

## Agent evolution (done 2026-10-12)
- [x] Named agent (CETI — Jero chose it 2026-10-12), task queue
      with schedules, tick loop.
- [x] Self-benchmarking: quizzes itself on its knowledge, records
      accuracy/tiers/latency per run.
- [x] Curiosity: failed asks -> gaps -> research tasks -> cassettes.
- [x] Vector floor raised 0.60 -> 0.70 on measured evidence (true
      paraphrases 0.705-0.869, unrelated "what is X" 0.511-0.680).
- [x] Split-brain heal: fallback-DB rows migrated to vector on boot.

- [x] Handler constraint leak, mitigated 2026-10-12: the BBS-history
      cassette landed and the handler rewrote "February 16, 1978" as
      "1971". New deterministic guard `_numbers_grounded`: every
      number the handler states must appear in the retrieved
      knowledge (or the question); on violation it serves the raw
      retrieved text instead of the confabulation. False alarms cost
      prettiness, never truth. Live-verified both directions.
- [ ] Local-tier confabulation can still be promoted: a fresh
      install asked "what is usenet" (nothing local knew it) and the
      1.5B model invented "1979, UC Berkeley, Network News Group",
      which the classifier promoted. Seeded the correct answers in
      seed.aiml for the obvious questions; the real fix is a stronger
      promotion judge for local-tier answers (they should earn
      cassettes via the researcher, not freelancing).

## Identity / network
- [x] Node identity 2026-10-12: `data/identity.dat` (0600) generated on
      first run; boot POST reports the fingerprint; `Brain.identity`;
      `/identity` REPL command; legacy files auto-upgrade;
      `CETI_IDENTITY` overrides. Master key stays local — only the
      public key is ever shared. Fingerprint tag configurable via
      `CETI_FINGERPRINT_PREFIX` (default `CETI`; `JERO` for the
      Jero network). The raw 32-byte secret is the regula_core
      plug point.
- [ ] regula_core plug: network message signing, peer handshake,
      torrent seeding identity, BBS front end file/image serving —
      all signing as the node identity.

## Roadmap

### Oracle: OpenRouter / Nvidia cloud tier
State: backends are coded (`ceti/backends.py`: `OpenRouterBackend`,
`NvidiaBackend`, OpenAI-compatible), `cloud_model` defaults to
`qwen/qwen3-8b:free`, keys load from env, the tier is disabled without
keys, and the PII stripper already gates everything sent up.
- [ ] Key setup: document `OPENROUTER_API_KEY` / `NVIDIA_API_KEY`,
      add an `install.sh` prompt that appends them to `ceti-env.sh`
      (never into the repo).
- [ ] Oracle roles, explicit escalation policy:
      (a) research distillation when the local model is too small to
      distill faithfully; (b) ingestion judging — adjudicating new
      data before it enters a cassette; (c) complex answers beyond
      local capability, via an `/oracle` command and via automatic
      escalation when local fails twice.
- [ ] Cost controls: free-tier models by default, per-day call caps
      in `tasks.db`, spend log.
- [ ] The oracle teaches: cloud answers already flow through the
      classifier (`learn_from_cloud`) into local cassettes — verify
      this loop live once keys exist.

### Torrent distribution
State: `ceti/seeder.py` writes `.torrent` files, `seedbox/` has the
transmission-daemon setup, `seedtest.py` proved a loopback transfer.
- [ ] Agent wiring: `release` tasks cut the tarball → make the
      torrent → start seeding; `ingest` tasks fetch → `verify_tarball`
      → `merge`. The queue already has both task kinds — connect them.
- [ ] `/seed` REPL command: what we're seeding, peers, ratios.
- [ ] Seeding identity: announce/sign as the node identity
      (`identity.dat`), so peers know whose sets they're pulling.
- [ ] Peer discovery story: DHT/PEX (already in the seedbox script)
      plus optional tracker; document the second-machine download
      verification as the release checklist.

### BBS front end (message board)
- [ ] Phase 1 — local board: message bases in sqlite (boards,
      threads, posts), file areas, served through the REPL first
      (`/bbs`), ANSI-friendly terminal UI.
- [ ] Phase 2 — signed posts: every post signed with the node
      identity; moderation = key-based (sysop signs, members post).
- [ ] Phase 3 — federated boards: peer sync through the merge
      protocol (record identity = hash of post, conflicts logged),
      same as dataset merge. Serves files/images once regula_core
      plugs in.

### Tool/skill system + sandbox
- [ ] Tool registry: manifest format (name, args schema, side-effect
      declaration, sandbox policy). Skills as markdown + code bundles
      in `skills/`, hot-loadable like cassettes.
- [ ] Sandbox runner: minimal QEMU microVM for coding and running
      commands — snapshot/restore per invocation, no network by
      default, CPU/RAM caps, 0600 scratch. The agent NEVER runs code
      on the host; the sandbox is the only execution environment.
- [ ] Agent loop: plan → write code → run in sandbox → observe →
      iterate, with the transcript feeding the task log.
- [ ] Skill curation: a `skills/` seed set (file ops, web fetch via
      the lynx stack, sqlite queries) plus docs for writing your own.

## Bugs found live (not yet fixed)

- [ ] **Vector false positive.** "explain what a merkle tree is"
      matched the BBS Q&A at similarity **0.619**, above the 0.60
      floor. A question about Merkle trees should never match a
      bulletin-board answer. Suspect the floor is too low, or the
      asymmetric prefix matches question *shape* more than content.
      Fix: measure a spread of related/unrelated pairs, consider
      raising the floor toward 0.70, or add a margin/second-pass
      check.
- [ ] **Handler constraint leak.** On that false match, handler mode
      asked Qwen to answer the Merkle question using ONLY the BBS
      knowledge — and Qwen answered from its own weights anyway,
      ignoring the constraint. This breaks the core guarantee
      ("LLMs just the handlers"). The small local model does not
      reliably obey the handler prompt. Fix options: harden the
      prompt, add an answer-grounding check (does the presented
      answer actually draw on the retrieved text?), or skip
      handler-rewriting when query/knowledge similarity is marginal.
      Note: the hallucinated answer was NOT persisted (learning only
      runs on local/cloud tiers) — the dataset stayed clean.
- [ ] **Handler latency.** Every memory hit currently pays a full
      local-Qwen rewrite (2–5s observed). Consider caching handled
      answers, or handling once per knowledge entry.

## Incomplete by design (queued work)

- [ ] **Researcher retry.** The first idle-research run (topic:
      "bulletin board system history") died without a report — the
      box was OOM-starved (~360MB free) and the research compile
      likely got killed. The topic stays queued in
      `researcher_log.sqlite3` and retries on the next idle cycle.
      A scheduled idle-cycle cron was proposed but needs Jero's
      approval (recurring work).
- [ ] **Cassettes.** Only one exists
      (`bulletin-board-system-history.aiml`) and it's still a shell
      until the researcher lands its report. `topics.txt` has 12
      queued topics.
- [ ] **Seedbox not deployed.** `seedbox/setup-seedbox.sh` is written
      but not yet run on the AlmaLinux VPS. The v2 torrent exists
      and is proven valid (loopback transfer + peer-side verify),
      but no live swarm yet.

## Designed, not built

- [x] **OS-style boot.** `brain/boot.py`: BIOS POST verifying every
      subsystem on each run (config, persona, AIML parsed+counted,
      knowledge DB schema+counts, embedding server, Ollama model,
      cloud keys, latest dataset release fully signature-verified,
      topic queue). Core failures refuse to start; service gaps are
      loud but bootable. The REPL boots before prompting.
- [x] **Idle heartbeat.** The REPL fires one background research
      cycle after 10 quiet minutes (own Brain per thread — sqlite
      handles never shared); results delivered before the next
      prompt, cassettes hot-reloaded. `/heartbeat` toggles.
- [x] **Dataset merge on download.** Built `brain/merge.py`:
      verify-first ingest, content-addressed records
      (sha256 of normalized question), score-wins conflict policy
      with ties keeping local, local embedding recompute (peer
      vectors never trusted), aliases + AIML categories, ledger
      merge records. Deliberately not the git binary: sqlite doesn't
      merge, line-based 3-way is the wrong granularity for Q&A, and
      a git dependency breaks stdlib-only. (Salvaged from an external
      review list, 2026-10-11 — the one item on it that was real.)
- [ ] Multi-signer archivist web-of-trust + key revocation.
- [ ] Cross-node signed `ask()` jobs through the relay.
- [ ] Cloud tier live test (no OpenRouter/Nvidia keys configured;
      the tier is coded but never fired).
- [ ] Khan-voice cassette: persona currently lives in the system
      prompt + seed identity only; no cassette of persona-style
      responses yet.

## Done (this session)

- [x] Framework: tiers, classifier, PII stripper, paraphrase/typo
      AIML, vector memory, cassettes, idle researcher, handler mode.
- [x] Signed/chained/gzip-packed datasets; single-file torrents +
      magnets; loopback seeding proof (89/89 tests green).
- [x] Seedbox pack for the VPS.
