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
- [x] Local-tier confabulation can still be promoted: a fresh
      install asked "what is usenet" (nothing local knew it) and the
      1.5B model invented "1979, UC Berkeley, Network News Group",
      which the classifier promoted. Seeded the correct answers in
      seed.aiml for the obvious questions; the real fix is the
      learning gates (2026-10-15): a truth judge must verify
      factual promotes (unsure votes NO, fail-closed), and a
      contradiction check blocks writes that disagree with
      memory or AIML answers, filing a curiosity gap instead.
      Serving is unchanged — only learning is gated.

## Identity / network
- [x] Node identity 2026-10-12: `data/identity.dat` (0600) generated on
      first run; boot POST reports the fingerprint; `Brain.identity`;
      `/identity` REPL command; legacy files auto-upgrade;
      `CETI_IDENTITY` overrides. Master key stays local — only the
      public key is ever shared. Fingerprint tag configurable via
      `CETI_FINGERPRINT_PREFIX` (default `CETI`; `JERO` for the
      Jero network). The raw 32-byte secret is the regula_core
      plug point.
- [x] regula_core plug, v1 (Gopher hole, 2026-10-14): network
      message signing (signed board posts), peer handshake
      (identity cards, public-key address book), BBS front end
      (signed posts, file areas) — all signing as the node
      identity. Still open: live transport, torrent seeding
      identity, federated board sync.

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
- [x] Oracle-driven file editing (Jero's ask): the agent loop
      (`/task`) lets the cloud model draft READMEs, sites, and code
      directly into the work root through `write_file`/`edit_file`
      — the operator watches and reviews before anything is kept.
      Proved manually 2026-10-13: oracle-generated
      `your_files/ceti-launch-site/index.html`. Mechanism live
      2026-10-14.

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

### BBS front end (message board) — Gopher hole, v1 done 2026-10-14
- [x] Phase 1 — local board: one JSON file per post in
      `data/board/posts/`, served through the REPL (`/board`) and
      the Gopher menu (`/gopher`). File areas = shared bundles.
- [x] Phase 2 — signed posts: every post signed with the node
      identity; bad signatures rejected, never displayed.
      Identity cards + peer address book = the handshake.
- [x] Phase 3 — federated boards, v1 (2026-10-14): signed
      nodelist + gossip for discovery, mail, chat rooms,
      upload/download via the drop. Import is explicit —
      no auto-propagation, no live transport. Still open:
      live transport, torrent seeding of bundles, moderation.

### TUI (Hermes-style full-screen interface) — v1 done 2026-10-14
- [x] `ceti/tui.py` (stdlib curses): header + transcript +
      multiline input, slash-command autocomplete, history,
      Ctrl-C interrupt, `/tui` in the REPL + `python -m ceti.tui`.
      Line REPL stays as the fallback. Still open: tool calls
      inside the TUI (`/task` still bounces to the line REPL).
- [x] **BBS startup (2026-10-15, Jero: stay on the line REPL).**
      Startup: clear screen -> colour ASCII banner + boot POST
      self-test -> `/help` -> line REPL prompt. The TUI never
      auto-loads; `/tui` enters it explicitly. `ceti/colour.py`
      (TTY-only ANSI, NO_COLOR respected), `ceti/help.py` +
      `/help`, boot tags coloured (green/amber/red). TUI itself
      got a curses palette (header bold green, you> cyan, answers
      green, tags amber, system dim, errors red) — render-time
      only, saved sessions stay plain text.
- [x] Streaming (2026-10-15): `stream_tokens()` +
      `finish_stream()` in `brain.py`, `complete_stream()` in
      `backends.py`, TUI pump + simulated 499 t/s reveal for
      blocks, Ctrl-C cancel, Enter redirect. Verified against
      live Ollama (19 chunks, real measured t/s).

### Tool/skill system + sandbox
- [x] Tool registry: Tool class (name, description, JSON-schema
      parameters, side-effect declaration, sandbox flag),
      `to_openai_schema()` for oracle function calling, `execute()`
      returning ToolResult. Tools: run_python, run_python_bg,
      job_poll, job_kill, read_file, write_file, edit_file,
      list_dir. Retry-with-backoff on cloud calls (ceti/retry.py).
      Done 2026-10-14.
- [x] Skills as markdown + code bundles in `skills/`,
      hot-loadable like cassettes: SKILL.md with flat `key: value`
      frontmatter (stdlib-only, no PyYAML), `name` + `description`
      required. Discovered via `list_skills`/`read_skill` tools and
      listed in the agent loop's system prompt. `/skills` lists,
      `/skills reload` rescans. Bad bundles skipped loudly.
      Reference: MiniMax Mini-Agent's SKILL.md loader (MIT).
      Done 2026-10-14.
- [x] Sandbox runner: pure-Python Linux namespace backend
      (QEMU was a non-starter on this box) — ro binds, tmpfs /tmp,
      private /work, no network, rlimits + wall-clock kill, fixed
      remount escape. Foreground AND background jobs (poll/wait/
      kill). The agent NEVER runs code on the host. QEMU stays a
      future backend behind the same policy. Done 2026-10-13/14.
- [x] Agent loop: oracle (cloud API) drives tools to complete
      tasks — `/task <description>` in the REPL. Plan → write →
      run in sandbox → observe → iterate, with the transcript in
      the session log. Per-session local memory: notes.json
      (record_note/recall_notes tools) + messages.jsonl under
      data/sessions/<id>/. Context summarization past the token
      limit (user messages always kept). Refuses loudly without a
      cloud API key. Done 2026-10-14.
- [x] Skill curation: `skills/` seed set — file-ops (the four
      file tools + jail + verify workflow), sandbox-python (no
      network, per-run /work, stdlib only, limits, background
      jobs), web-fetch (honest options given no sandbox network),
      sqlite (stdlib sqlite3 patterns). Authoring guide in
      `docs/skills.md`. Done 2026-10-14.

### Library ingest + oracle paper writer
Done 2026-10-14 (Jero's ask — "drop a file, it's in vector memory",
StoryBlocks-style paper writer).
- [x] `ceti/ingest.py`: `data/library/` drop folder (.pdf via
      pdftotext, .txt/.md); paragraph-aware chunking (~600 tokens,
      ~120 overlap); embeddinggemma-2 embeddings; `doc_chunks`
      table in `data/library.sqlite3` (separate from Q&A);
      sha256 dedup; `/ingest` REPL command (+ heartbeat-ready
      `scan()`). Restarted the embedding server (:11435) after
      the VM replacement killed it.
- [x] `ceti/story.py`: stories as block sequences (~200
      words/page, retry-if-short like StoryBlocks); story bible
      first (title, characters, arc, 12 beats); rolling chapter
      summaries; top-k library chunks as style reference (voice
      inspiration, never reproduced); plain files under
      `data/stories/<id>/`; `/story new|write|status|compile|list`.
      Proven live: bible + chapter + compile via OpenRouter.
- [ ] Branching narratives (StoryBlocks' two-choice trees) — v2.

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

- [x] **Research engine config (2026-10-15, commit 35d5cdc).**
      `data/research.json` governs idle learning, managed from the
      REPL with `/research`: sites allowlist (default Wikipedia-only;
      content fetched ONLY from listed hosts — the stack drops
      off-list results before any fetch via `--allow-sites`),
      topics queue (`topics.txt`), evergreen interests (cycled
      least-recent-first when the queue drains), nightly budget
      (`max_topics`/`max_qa`/`max_minutes`, enforced by
      `run_nightly()` + the nightly driver). Replaces the old
      forced `CETI_WIKI_ONLY` switch; default config still uses
      the `--wiki-only` fast path, so an older stack keeps working.
      Stack side lives outside this repo:
      `sovereign/research/research.py` gained `--allow-sites`.
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
