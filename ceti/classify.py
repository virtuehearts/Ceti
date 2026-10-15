"""RelevanceClassifier — decides whether an answer is worth keeping.

After the LLM answers, the pair (question, answer) is scored 0..1 on
how likely it is to be durable general knowledge:

  promote    score >= promote_threshold   -> write to AIML + memory index
  adjudicate score in [adjudicate_low, threshold) -> ask Qwen, one cheap call
  discard    below that                    -> forget it

Rules-based scoring is free and instant; the LLM judge only sees the
borderline cases. Personal/time-sensitive/one-off content is refused
even if it scores well elsewhere.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import pii

_QUESTION_LEAD = re.compile(
    r"^(what|why|how|when|where|which|who|whom|whose|is|are|was|were|"
    r"do|does|did|can|could|should|would|will|has|have|had|define|"
    r"explain|describe|list|name|tell me)\b",
    re.IGNORECASE,
)

_TIME_SENSITIVE = re.compile(
    r"\b(today'?s|right now|current|latest|this week|this morning|"
    r"price of|stock price|weather|news|who won|score of)\b",
    re.IGNORECASE,
)

_COMMAND_LEAD = re.compile(
    r"^(run|open|close|delete|remove|send|play|set|remind|schedule|"
    r"book|buy|order|cancel|install|update|restart|reboot|ssh|ping|"
    r"curl|wget)\b",
    re.IGNORECASE,
)

_AI_HEDGE = re.compile(
    r"\bas an ai\b|\bi don'?t have (real-?time|access to)\b|"
    r"\bi'?m (just|only) a (language|ai) model\b|"
    r"\bmy (training|knowledge) (data|cutoff)\b",
    re.IGNORECASE,
)

_EXPLAINS = re.compile(r"because|:|^\d+\.|\n[-*]\s",
                         re.MULTILINE | re.IGNORECASE)

_PERSONAL_Q = re.compile(
    r"\b(my|mine|our)\b|\bi (live|work|am|have)\b", re.IGNORECASE)


@dataclass
class Verdict:
    score: float
    action: str  # "promote" | "adjudicate" | "discard"
    reasons: list[str] = field(default_factory=list)


class RelevanceClassifier:
    def __init__(self, promote_threshold: float = 0.65,
                 adjudicate_low: float = 0.45):
        self.promote_threshold = promote_threshold
        self.adjudicate_low = adjudicate_low

    def score(self, question: str, answer: str) -> tuple[float, list[str]]:
        reasons: list[str] = []
        s = 0.0
        q = question.strip()
        a = answer.strip()

        if _QUESTION_LEAD.match(q):
            s += 0.25
            reasons.append("question-shaped")
        if 40 <= len(a) <= 1200:
            s += 0.15
            reasons.append("answer length in band")
        elif len(a) < 20:
            s -= 0.20
            reasons.append("answer very short")
        elif len(a) < 40:
            s -= 0.10
            reasons.append("answer short")

        _clean_q, found_q = pii.strip(q)
        _clean_a, found_a = pii.strip(a)
        if not found_q and not found_a:
            s += 0.15
            reasons.append("no PII detected")
        else:
            s -= 0.35
            reasons.append(f"PII present: {sorted(set(found_q + found_a))}")

        if not pii.looks_personal(q):
            s += 0.10
            reasons.append("not personal")
        else:
            s -= 0.30
            reasons.append("personal query")

        if _TIME_SENSITIVE.search(q) or _TIME_SENSITIVE.search(a):
            s -= 0.30
            reasons.append("time-sensitive")
        else:
            s += 0.10
            reasons.append("not time-sensitive")

        if _AI_HEDGE.search(a):
            s -= 0.25
            reasons.append("model hedge/refusal language")
        else:
            s += 0.10
            reasons.append("no hedge language")

        if _COMMAND_LEAD.match(q):
            s -= 0.40
            reasons.append("command, not knowledge")

        if _EXPLAINS.search(a):
            s += 0.05
            reasons.append("explanatory structure")

        if _PERSONAL_Q.search(q):
            s -= 0.20
            reasons.append("first-person framing")

        s = max(0.0, min(1.0, s))
        return s, reasons

    def decide(self, question: str, answer: str) -> Verdict:
        s, reasons = self.score(question, answer)
        if s >= self.promote_threshold:
            action = "promote"
        elif s >= self.adjudicate_low:
            action = "adjudicate"
        else:
            action = "discard"
        return Verdict(score=s, action=action, reasons=reasons)

    # ---- LLM judge for borderline cases -------------------------------
    ADJUDICATE_PROMPT = (
        "You are a relevance judge. Reply with exactly YES or NO.\n"
        "Should this Q&A be saved as permanent general knowledge?\n"
        "Say YES only if the answer is general, factual, stable over time, "
        "and contains no personal data.\n"
        "Say NO for opinions, time-sensitive info, personal details, "
        "one-off requests, or hedged non-answers.\n"
        "Q: {q}\nA: {a}\n"
        "YES or NO:"
    )

    def adjudicate(self, question: str, answer: str,
                   llm_callable) -> bool:
        """llm_callable(prompt) -> str. Returns True to promote."""
        try:
            out = llm_callable(
                self.ADJUDICATE_PROMPT.format(
                    q=question[:500], a=answer[:800])).strip().upper()
        except Exception:
            return False  # judge failed: default to forgetting
        return out.startswith("YES")

    # ---- LLM truth judge for factual promotes -------------------------
    # The relevance scorer above checks shape, never truth: a
    # confident confabulation ("the C64 was introduced in 1977")
    # scores exactly like a true fact. So a factual answer that
    # earned "promote" still needs one verification call before
    # it enters permanent memory. The judge is the same small
    # model, so this is not a guarantee — it catches the cases
    # where the model does not believe its own answer on a second
    # ask, and "unsure" votes NO. Ground truth stays the job of
    # cassettes, research, and the contradiction check in _learn.
    TRUTH_PROMPT = (
        "You are a fact checker. Reply with exactly YES or NO.\n"
        "Is the factual claim in this answer correct, to the best "
        "of your knowledge?\n"
        "Say YES only if you are confident the claim is true.\n"
        "Say NO if the claim is wrong, or if you are unsure.\n"
        "Q: {q}\nA: {a}\n"
        "YES or NO:"
    )

    def verify_truth(self, question: str, answer: str,
                     llm_callable) -> bool:
        """llm_callable(prompt) -> str. Returns True to learn."""
        try:
            out = llm_callable(
                self.TRUTH_PROMPT.format(
                    q=question[:500], a=answer[:800])).strip().upper()
        except Exception:
            return False  # judge failed: default to forgetting
        return out.startswith("YES")
