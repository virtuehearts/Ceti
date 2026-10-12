# Troubleshooting

- **Boot refuses to start.** Core failures (config, persona, AIML)
  refuse; everything else is a loud warning and the tiers fall
  through. Read the POST lines — they name the gap.
- **Everything answers `[local]` slowly.** The embedding server is
  probably down — `run.sh` starts it, or check `./logs/embeddings.log`.
  Without vectors, memory uses TF-IDF and the floor is higher.
- **Research tasks skip.** The box needs ~1 GB free and load < 2.0
  (`Researcher(min_mem_gb=…, max_load=…)` tunes this), plus
  `CETI_RESEARCH_PY` pointing at a research stack.
- **Ollama hangs on 127.0.0.1.** Use the HTTP API with `--noproxy '*'`
  (or set `no_proxy`); the `ollama run` CLI can hang behind proxies.
- **AIML answers feel stale.** `/trace` shows the tier; `learned.aiml`
  is safe to delete — the brain re-learns.
- **`install.sh` can't reach the network.** The llama-server fetch
  uses the GitHub API and the Ollama install uses `ollama.com` —
  both need egress. Behind a proxy, export `https_proxy` first.
