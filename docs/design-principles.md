# Design Principles

The rules CETI is built under. They are stated as goals — where the
code hasn't caught up yet, it says so.

## 1. Zero tokens at serve time

The local model never spends inference answering you live. It does its
work when idle — research queues, curiosity tasks, promotion and
adjudication — compiling answers into cassettes and memory ahead of
time. Serving is pure retrieval: AIML cassettes, then local memory.

A live model call at query time is a cache miss. The correct response
to a miss is to enqueue it, not to burn tokens. The next time you ask,
the answer is already there, in ~0ms.

*Status: stated goal. The serve-time local tier (handler mode) still
exists today; the roadmap is to retire it — "researcher-grown
knowledge outranks freelanced local answers."*

## 2. The datasets are the mind; the models are only the handlers

Knowledge lives in signed cassettes and the local memory store.
Models present, distill, and judge — they don't originate. If you took
the models away, the brain would still answer from what it knows. If
you took the datasets away, the models would have nothing worth
saying.

## 3. The cloud is the last-resort oracle

Local tiers answer first, always. The cloud only ever sees
PII-stripped queries, only when everything local has failed, and only
if you gave it keys. It teaches — distilling answers that get
promoted into local cassettes — it never answers you directly.

## 4. No signature, no trust

Knowledge ships as signed, chained, merkle-rooted bundles. Peers
verify the signature before merge; unsigned or invalid bundles are
rejected outright. Identity is generated on-device on first run —
the private key never leaves the machine, only the fingerprint is
ever shared.

## 5. Nothing phones home

No accounts. No telemetry. No remote control plane. The brain is a
few hundred kilobytes; the full model stack is optional. Missing
pieces degrade their tier instead of breaking the whole — the boot
screen reports exactly what is available, loudly.

## 6. The model presents and distills; it never freestyles

Whenever the model runs — idle research, adjudication, or the
transitional serve-time handler — it works from retrieved knowledge,
not from its weights. The numbers-grounding guard rejects any
rewrite that invents figures and falls back to the raw retrieved
text. Fluency is not evidence.

## 7. Curiosity is a first-class workload

Repeated failures are not errors to hide — they become research
tasks. The agent keeps a gaps log, the idle heartbeat works through
it, and the self-benchmark quizzes everything the brain claims to
know, including with typos. Not knowing is scheduled work, not
shame.
