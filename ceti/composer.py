"""Frontal lobe / Composer: the reasoner.

Derives intent from a question, gathers pieces through a caller-supplied
fetch callable, composes an answer from those pieces, and grounds it.

RULES ONLY — no model calls, no imports from ceti.brain (avoid cycles).
The composer never fetches anything itself: it receives pieces via the
callable passed to Composer.__init__, and it never invents content —
it only rearranges piece text plus words from the user's own question
plus fixed template glue ("The ... of ... is ...", all stopwords).

Grounding (see is_grounded): every number in the composed answer must
appear in at least one piece (or in the question itself, which the user
supplied), and at least 40% of the answer's content words must appear in
the pieces or the question. Because the composer only rearranges piece
text + question words, a composed draft passes grounding by
construction; the check is defense-in-depth, and on failure the
composer falls back to quoting the best piece verbatim with
grounded=False rather than shipping an ungrounded answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

__all__ = [
    "Intent",
    "Piece",
    "ComposedAnswer",
    "Composer",
    "derive_intent",
    "is_grounded",
]

_STOPWORDS = frozenset({
    "the", "a", "an", "of", "is", "was", "are", "were",
    "in", "on", "and", "or", "to", "for",
})

_GREETINGS = frozenset({"hello", "hi", "hey"})

_ENTITY_PREFIXES = (
    "what is the ", "what is ", "what's the ", "what's ",
    "whats the ", "whats ", "tell me about ", "who is ", "who was ",
    "when was ", "compare ", "difference between ",
)

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Intent:
    """What the composer thinks the user is asking for."""
    action: str          # lookup | explain | compare | compute | greet | unknown
    entity: str          # subject, "" when none ("France", "France vs Germany")
    attribute: str       # lookup attribute, "" otherwise ("capital", ...)
    question: str        # the original question, verbatim
    confidence: float


@dataclass
class Piece:
    """One chunk of retrieved knowledge handed to the composer."""
    text: str
    source: str
    confidence: float = 1.0


@dataclass
class ComposedAnswer:
    text: str
    grounded: bool
    pieces: list[Piece] = field(default_factory=list)
    intent: Intent | None = None


# ---------------------------------------------------------------------------
# Intent derivation (rule-based, deterministic)
# ---------------------------------------------------------------------------

def _clean_entity(raw: str) -> str:
    """Strip question-word prefixes, punctuation, and title-case."""
    s = raw.strip().lower()
    s = s.strip("?!.").strip()
    for prefix in _ENTITY_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.startswith("the "):
        s = s[len("the "):]
    s = re.sub(r"\s+", " ", s).strip()
    return s.title()


def _clean_side(raw: str) -> str:
    return _clean_entity(raw)


def derive_intent(question) -> Intent:
    """Rule-based intent derivation. Lowercase + keyword rules only."""
    q = question if isinstance(question, str) else str(question or "")
    original = q
    low = q.strip().lower().strip("?!.").strip()

    tokens = re.findall(r"[a-z]+", low)

    # --- greet: a bare greeting, optionally with one more word ("hi there")
    if tokens and tokens[0] in _GREETINGS and len(tokens) <= 2:
        return Intent("greet", "", "", original, 0.9)

    # --- compare: "difference between X and Y" / "compare X and Y" / "X vs Y"
    m = re.search(r"difference between (.+?) and (.+)$", low)
    if m:
        left, right = _clean_side(m.group(1)), _clean_side(m.group(2))
        return Intent("compare", f"{left} vs {right}", "", original, 0.9)
    if low.startswith("compare "):
        rest = low[len("compare "):]
        for sep in (" and ", " with "):
            if sep in rest:
                left, right = rest.split(sep, 1)
                left, right = _clean_side(left), _clean_side(right)
                return Intent("compare", f"{left} vs {right}", "",
                              original, 0.9)
    m = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+)$", low)
    if m and "capital of" not in low and "population of" not in low \
            and "currency of" not in low and "money of" not in low:
        left, right = _clean_side(m.group(1)), _clean_side(m.group(2))
        return Intent("compare", f"{left} vs {right}", "", original, 0.9)

    # --- attribute lookup: capital / currency / population / introduced ...
    if "capital of " in low:
        entity = _clean_entity(low.split("capital of ", 1)[1])
        return Intent("lookup", entity, "capital", original, 0.9)
    if "currency of " in low:
        entity = _clean_entity(low.split("currency of ", 1)[1])
        return Intent("lookup", entity, "currency", original, 0.9)
    if "money of " in low:
        entity = _clean_entity(low.split("money of ", 1)[1])
        return Intent("lookup", entity, "currency", original, 0.9)
    m = re.match(r"what currency does (.+?) use$", low)
    if m:
        return Intent("lookup", _clean_entity(m.group(1)), "currency",
                      original, 0.9)
    if "population of " in low:
        entity = _clean_entity(low.split("population of ", 1)[1])
        return Intent("lookup", entity, "population", original, 0.9)
    m = re.match(r"when was (.+?) (introduced|released|founded)\b", low)
    if m:
        return Intent("lookup", _clean_entity(m.group(1)), m.group(2),
                      original, 0.9)

    # --- compute: arithmetic expressions ("2+2", "sqrt", "% of")
    if re.search(r"\d\s*[-+*/^]\s*\d", low) \
            or "% of" in low \
            or re.search(r"\bsqrt\b", low) \
            or low.startswith(("calculate ", "compute ", "solve ")):
        return Intent("compute", "", "", original, 0.9)

    # --- explain: "what is X" / "tell me about X" / "who is X"
    m = re.match(r"what(?:'s|s)? (?:is )?(?:the |a |an )?(.+)$", low)
    if m and low.startswith(("what is ", "what's ", "whats ")):
        return Intent("explain", _clean_entity(m.group(1)), "", original, 0.8)
    if low.startswith("tell me about "):
        return Intent("explain",
                      _clean_entity(low[len("tell me about "):]),
                      "", original, 0.8)
    m = re.match(r"who (?:is|was) (.+)$", low)
    if m:
        return Intent("explain", _clean_entity(m.group(1)), "", original, 0.8)

    return Intent("unknown", "", "", original, 0.0)


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------

def _numbers(text: str) -> list[str]:
    """Number tokens with commas stripped ("1,000" -> "1000")."""
    return [m.group(0).replace(",", "") for m in _NUMBER_RE.finditer(text)]


def _content_words(text: str) -> set[str]:
    """Lowercase words, len >= 3, minus the stopword set."""
    return {w for w in _WORD_RE.findall(text.lower())
            if len(w) >= 3 and w not in _STOPWORDS}


def _vocab(text: str) -> set[str]:
    """All lowercase words len >= 3 in a source text."""
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) >= 3}


def _number_present(number: str, text: str) -> bool:
    """True if `number` occurs as a whole number token in `text`.

    Digit boundaries on both sides, so "2" does not match inside
    "1982" and "3.14" does not match inside "3.14159".
    """
    flat = text.replace(",", "")
    # Lookbehind: not part of a longer number / fraction. Lookahead:
    # not followed by a digit, nor by ".<digit>" (a longer decimal) —
    # but a sentence-final "." after the number is fine.
    return re.search(r"(?<![\d.])" + re.escape(number) + r"(?!\d)(?!\.\d)",
                     flat) is not None


def is_grounded(answer: str, pieces: list[Piece], question: str = "") -> bool:
    """Grounding check for a composed answer.

    - Every number in the answer must appear in at least one piece
      (or in the question, which the user supplied, not the composer).
    - At least 40% of the answer's content words (lowercase, len>=3,
      minus stopwords) must appear in the pieces or the question.

    Words from the question count as grounded: the composer only
    rearranges piece text plus question words plus stopword template
    glue, so question words in the answer are the user's own words,
    not invented content.
    """
    for number in _numbers(answer):
        if not _number_present(number, " ".join(p.text for p in pieces)) \
                and not _number_present(number, question):
            return False
    answer_words = _content_words(answer)
    if not answer_words:
        return True  # nothing content-bearing to ground
    known = set()
    for p in pieces:
        known |= _vocab(p.text)
    known |= _vocab(question)
    overlap = len(answer_words & known) / len(answer_words)
    return overlap >= 0.4


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------

def _looks_like_sentence(text: str) -> bool:
    """True if the piece already reads as a sentence: use it verbatim."""
    s = text.strip()
    if not s:
        return False
    if s[-1] in ".!?":
        return True
    words = s.split()
    return len(words) >= 4 and any(
        glue in f" {s.lower()} " for glue in (" is ", " was ", " are ",
                                              " were ", " has ", " have "))


class Composer:
    """Composes answers from pieces supplied by the caller's fetch.

    fetch: callable(Intent) -> list[Piece]. The composer never fetches
    anything itself and never calls a model.
    """

    def __init__(self, fetch: Callable[[Intent], list[Piece]]):
        self._fetch = fetch

    def compose(self, question: str) -> ComposedAnswer | None:
        """Compose an answer, or None to leave it to another lobe."""
        intent = derive_intent(question)
        # Greet / compute / unknown belong to other lobes; do not answer.
        if intent.action in ("greet", "compute", "unknown"):
            return None
        pieces = self._fetch(intent) or []
        if not pieces:
            return None

        if intent.action == "lookup":
            draft = self._lookup(intent, pieces)
        elif intent.action == "explain":
            draft = self._explain(pieces)
        elif intent.action == "compare":
            draft = self._compare(intent, pieces)
        else:  # pragma: no cover - derive_intent only yields known actions
            return None
        if draft is None:
            return None

        if is_grounded(draft, pieces, question=question):
            return ComposedAnswer(draft, True, pieces, intent)
        # Grounding failed: quote the best piece verbatim, don't invent.
        best = max(pieces, key=lambda p: p.confidence)
        return ComposedAnswer(best.text.strip(), False, pieces, intent)

    @staticmethod
    def _lookup(intent: Intent, pieces: list[Piece]) -> str | None:
        best = max(pieces, key=lambda p: p.confidence)
        text = best.text.strip()
        if not text:
            return None
        if _looks_like_sentence(text):
            return text
        if intent.attribute:
            return f"The {intent.attribute} of {intent.entity} is {text}."
        return text

    @staticmethod
    def _explain(pieces: list[Piece]) -> str | None:
        texts = [p.text.strip() for p in pieces[:3]]
        texts = [t for t in texts if t]
        if not texts:
            return None
        return " ".join(texts)

    @staticmethod
    def _compare(intent: Intent, pieces: list[Piece]) -> str | None:
        if " vs " not in intent.entity or len(pieces) < 2:
            return None
        left_name, right_name = intent.entity.split(" vs ", 1)
        half = len(pieces) // 2
        left = " ".join(p.text.strip() for p in pieces[:half])
        right = " ".join(p.text.strip() for p in pieces[half:])
        if not left or not right:
            return None
        return f"{left_name}: {left} | {right_name}: {right}"
