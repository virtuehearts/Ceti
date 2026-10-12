# Oracle (cloud tier)

The top of the ladder is the **oracle**: frontier models reached
through OpenRouter and Nvidia's API (`ceti/backends.py`:
`OpenRouterBackend`, `NvidiaBackend` — both OpenAI-compatible).
It is coded, tested with fakes, and disabled until you add keys:

```bash
export OPENROUTER_API_KEY=…    # and/or
export NVIDIA_API_KEY=…
```

Defaults live in `ceti/config.py`: `cloud_model` is
`qwen/qwen3-8b:free`, backends are tried in `cloud_order`
(`openrouter`, then `nvidia`).

## The three jobs

In escalating order:

1. **Complex answers.** When the local model fails or the question
   exceeds it, the oracle answers — via automatic escalation or an
   explicit `/oracle` command (planned).
2. **Research distillation.** When a research report is too subtle
   for the 1.5B local model to distill faithfully, the oracle does
   the distillation instead.
3. **Ingestion judging.** New data — peer bundles, scraped sources —
   is adjudicated by the oracle before it earns a place in a
   cassette.

## The two rules

First, **the oracle never sees you**: every query passes through the
PII stripper first — the cloud gets the question, never the person.
Second, **the oracle teaches**: its answers flow through the same
relevance classifier as local ones (`learn_from_cloud`), so frontier
knowledge gets distilled down into local cassettes and the oracle is
needed less over time.

## Cost controls

Free-tier models by default. Per-day call caps and a spend log are
on the roadmap — for now, watch your provider dashboards.
