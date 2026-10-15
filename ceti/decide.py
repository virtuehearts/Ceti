"""DecisionLayer — interprets a query and plans the tier order.

Tiers, cheapest first:
  aiml    instant pattern brain (~ms, no LLM)
  memory  fuzzy index over promoted Q&A (~ms, no LLM)
  local   Qwen on Jero's iron via Ollama (private, free)
  cloud   OpenRouter / Nvidia (PII-stripped query only)

Intents steer the plan: smalltalk never touches the cloud,
commands are never learned, factual questions get the full ladder.
"""
from __future__ import annotations

import re

_SMALLTALK = re.compile(
    r"^(hi|hey|hello|yo|sup|howdy|good\s?(morning|afternoon|evening|day)"
    r"|thanks?|thank\s?you|thx|bye|goodbye|good\s?night|see\s?you"
    r"|how\s?are\s?you|what'?s\s?up)[!.\s?]*$",
    re.IGNORECASE,
)

_COMMAND_LEAD = re.compile(
    r"^(run|open|close|delete|remove|send|play|set|remind|schedule|"
    r"book|buy|order|cancel|install|update|restart|reboot|ping)\b",
    re.IGNORECASE,
)

_QUESTION_LEAD = re.compile(
    r"^(what|why|how|when|where|which|who|whom|whose|is|are|was|were|"
    r"do|does|did|can|could|should|would|will|has|have|had|define|"
    r"explain|describe|list|name)\b",
    re.IGNORECASE,
)

# intents that may be persisted when the answer is relevant
_LEARNABLE = {"factual", "chat"}


def classify_intent(query: str) -> str:
    q = query.strip()
    if not q:
        return "empty"
    if _SMALLTALK.match(q):
        return "smalltalk"
    if _COMMAND_LEAD.match(q):
        return "command"
    if q.endswith("?") or _QUESTION_LEAD.match(q):
        return "factual"
    return "chat"


def plan(intent: str, cfg) -> list[str]:
    """Ordered tier list for an intent, honoring config switches."""
    # calc is always first: deterministic, stdlib-only, no switch.
    # Math is computed in Python — it never reaches a model.
    if intent == "smalltalk":
        tiers = ["calc", "aiml", "local"]
    elif intent == "command":
        tiers = ["calc", "aiml", "local"]
    elif intent in ("factual", "chat"):
        tiers = ["calc", "aiml", "memory", "local", "cloud"]
    else:  # empty / unknown
        tiers = ["calc", "aiml", "local"]
    switch = {"aiml": cfg.enable_aiml, "memory": cfg.enable_memory,
              "local": cfg.enable_local, "cloud": cfg.enable_cloud}
    return [t for t in tiers if switch.get(t, True)]


def learnable(intent: str) -> bool:
    return intent in _LEARNABLE
