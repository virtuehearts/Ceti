"""Unified Memory Bus — CETI as BIOS.

CETI is not an app that calls an LLM. It is a BIOS: a thin, fast,
hardware-like layer under everything, and the LLM is just one slow
peripheral on the bus.

Layers (like a microchip):
  POST    brain/boot.py — verifies config, persona, AIML, knowledge
          DB, embedding server, Ollama, cloud keys, signed dataset.
  BUS     this file — the memory controller. The CPU never touches a
          store directly; it only issues bus opcodes.
  CPU     Brain.ask / agent loop / story writer — issues bus reads.
  DMA     the idle researcher / heartbeat — prefetches and fills
          memory without the CPU (curiosity tasks are DMA requests).
  IRQ     miss -> curiosity interrupt. Low confidence -> adjudication
          interrupt. Bad signature -> fault, never displayed.

Cache hierarchy (cheapest first, like a real chip):
  L0  session registers (caller-held, not a store)
  L1  AIML — ~0ms, exact patterns (seed + learned + cassettes;
      cassettes ride inside the AIML tier, priority-learned >
      cassette > seed — see ceti/aiml.py)
  L2  memory — vector (768d) or TF-IDF Q&A, semantic / fuzzy
  L3  library — doc_chunks from data/library/ (the big book)
  L4  cloud oracle — slowest, most expensive, like disk/network.
      Last resort only. The bus NEVER calls it on its own:
      Bus.compile() returns gaps; the caller decides whether and
      how to fill them.

Every store answers the same shape — Frame — so the CPU never
learns store internals:

    Frame(text, store, question, confidence, cost, signature, ...)

Compile-first (the point of the bus):
  The old ladder was fall-through: first tier that hit won, the
  rest were never asked, and an article either came from local
  OR went wholesale to the oracle. The bus aggregates instead:
  Bus.compile(topic) fans out to sub-questions, reads EVERY store
  for each one, returns local hits + an explicit gap list, and
  only the gaps ever go to the oracle — never the whole article.

Example — "Commodore 64 article":
  local hits: 1982, Commodore, MOS 6510, 64K RAM, VIC-II, SID ...
  gaps:       C64 vs Amiga, C64 vs Atari, sales figures ...
  oracle:     called with ONLY the gaps, answers verified,
              written back as signed knowledge. Next time the
              article is 100% local.

Opcodes (microchip feel, same names in the REPL /bus command):
  READ       one query across all stores -> BusResult
  COMPILE    topic -> Compilation (hits + gaps + coverage)
  WRITE      Frame -> promote via the brain's learn gate
  PREFETCH   topic -> gaps only, for the researcher (DMA)
  INVALIDATE store is append-only in v1: returns 0, fail-loud
             in docs, never silently drops knowledge.

Stdlib only. No network in this file — oracle calls live in the
caller (story.ArticleCompiler, agent loop), behind an explicit
gap_fill_fn the caller supplies.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

# Opcodes — the bus instruction set.
OP_READ = "READ"
OP_COMPILE = "COMPILE"
OP_WRITE = "WRITE"
OP_PREFETCH = "PREFETCH"
OP_INVALIDATE = "INVALIDATE"
OPCODES = (OP_READ, OP_COMPILE, OP_WRITE, OP_PREFETCH, OP_INVALIDATE)

# Store names, in bus (cache) order.
STORE_CALC = "calc"
STORE_AIML = "aiml"
STORE_MEMORY = "memory"
STORE_LIBRARY = "library"
STORE_ORACLE = "oracle"
STORES = (STORE_CALC, STORE_AIML, STORE_MEMORY, STORE_LIBRARY,
          STORE_ORACLE)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "",
                                      text.lower())).strip()


def _dedupe_key(text: str) -> str:
    return _norm(text)[:200]


@dataclass
class Frame:
    """One answer from one store, in the bus's common shape."""
    text: str
    store: str                       # calc | aiml | memory | library | oracle
    question: str = ""               # the sub-question this answers
    confidence: float = 1.0          # 0..1, store-native score
    cost: int = 0                    # 0 = local, 1 = oracle (slow/paid)
    signature: bool | None = None    # True = signed/verified, None = n/a
    latency_ms: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"text": self.text, "store": self.store,
                "question": self.question,
                "confidence": self.confidence, "cost": self.cost,
                "signature": self.signature,
                "latency_ms": self.latency_ms,
                "metadata": dict(self.metadata)}


@dataclass
class Gap:
    """A sub-question no local store could answer — an interrupt."""
    question: str
    intent: str = ""
    reason: str = "miss"             # miss | below-floor | error
    stores_checked: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"question": self.question, "intent": self.intent,
                "reason": self.reason,
                "stores_checked": list(self.stores_checked)}


@dataclass
class BusResult:
    """Result of one READ: every local hit, plus gaps if none hit."""
    query: str
    intent: str = ""
    frames: list[Frame] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)

    @property
    def hit(self) -> bool:
        return bool(self.frames)

    @property
    def best(self) -> Frame | None:
        """Highest-confidence frame, ties broken by bus order."""
        if not self.frames:
            return None
        order = {s: i for i, s in enumerate(STORES)}
        return sorted(self.frames,
                      key=lambda f: (-f.confidence,
                                     order.get(f.store, 99)))[0]


@dataclass
class Compilation:
    """Result of one COMPILE: local hits + explicit gaps."""
    topic: str
    intent: str = "article"
    frames: list[Frame] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    subquestions: list[str] = field(default_factory=list)

    @property
    def local_coverage(self) -> float:
        """Fraction of sub-questions with at least one LOCAL hit.
        Oracle gap-fill frames never count — after a fill, coverage
        must still read the pre-fill local share (e.g. 0.5, not 1.0)."""
        if not self.subquestions:
            return 0.0
        hit_qs = {_norm(f.question) for f in self.frames
                  if f.question and f.store != STORE_ORACLE}
        total = len(self.subquestions)
        hits = sum(1 for q in self.subquestions if _norm(q) in hit_qs)
        return hits / total if total else 0.0

    @property
    def oracle_needed(self) -> bool:
        return bool(self.gaps)

    def gap_questions(self) -> list[str]:
        return [g.question for g in self.gaps]

    def to_context(self, max_chars: int = 12000) -> str:
        """Local knowledge as one block, for a handler prompt or
        for article assembly. Deduped, grouped by sub-question,
        provenance-tagged. Never includes oracle text unless the
        caller filled the gaps first (those frames carry
        store='oracle')."""
        seen: set[str] = set()
        parts: list[str] = []
        by_q: dict[str, list[Frame]] = {}
        for f in self.frames:
            key = _dedupe_key(f.text)
            if key in seen:
                continue
            seen.add(key)
            by_q.setdefault(f.question or self.topic, []).append(f)
        for q, frames in by_q.items():
            lines = ["[%s] %s" % (f.store, f.text.strip())
                     for f in frames]
            parts.append("Q: %s\n%s" % (q, "\n".join(lines)))
        out = "\n\n".join(parts)
        return out[:max_chars]

    def to_dict(self) -> dict:
        return {"topic": self.topic, "intent": self.intent,
                "local_coverage": round(self.local_coverage, 3),
                "oracle_needed": self.oracle_needed,
                "frames": [f.to_dict() for f in self.frames],
                "gaps": [g.to_dict() for g in self.gaps],
                "subquestions": list(self.subquestions)}


# Default fan-out for an article/compile: deterministic, no LLM, so
# the bus stays testable offline and never spends tokens to learn
# what it doesn't know. Callers that know the domain (e.g. C64 vs
# Amiga/Atari) pass their own subquestions — that is the intended
# way to get domain-specific gaps.
DEFAULT_FANOUT = (
    "What is {topic}?",
    "When was {topic} created or released, and by whom?",
    "How does {topic} work — what is its architecture or design?",
    "What are the key facts, specs, or components of {topic}?",
    "How does {topic} compare to its alternatives or rivals?",
    "What was {topic}'s impact, reception, or legacy?",
)


def default_fanout(topic: str) -> list[str]:
    return [s.format(topic=topic) for s in DEFAULT_FANOUT]


class MemoryBus:
    """The memory controller. Construct from a Brain, or from raw
    stores for tests:

        bus = MemoryBus.from_brain(brain, library=lib)
        bus = MemoryBus(aiml=fake_aiml, memory=fake_mem, library=None)

    A store is anything with the method the bus calls:
      aiml:   .respond(query) -> str | None
      memory: .search(query, top_k=n) -> [(q, a, score, sim)]
      library:.search(query, top_k=n) -> [{text, score, filename, ...}]
    Calc is tried via ceti.calc when available; a missing calc
    module just means no calc frames, never a crash.
    """

    def __init__(self, aiml=None, memory=None, library=None,
                 cfg=None, brain=None, gap_log=None):
        self.aiml = aiml
        self.memory = memory
        self.library = library
        self.cfg = cfg
        self.brain = brain
        self.gap_log = gap_log

    @classmethod
    def from_brain(cls, brain, library=None):
        return cls(aiml=getattr(brain, "aiml", None),
                   memory=getattr(brain, "memory", None),
                   library=library,
                   cfg=getattr(brain, "cfg", None),
                   brain=brain,
                   gap_log=getattr(brain, "gap_log", None))

    # ---- POST ---------------------------------------------------------
    def post(self) -> dict:
        """BIOS POST memory map: counts per store. Fail-soft —
        a missing store reports 0, never raises."""
        out: dict = {"stores": {}}
        try:
            out["stores"]["aiml"] = self.aiml.count() if self.aiml else 0
        except Exception:
            out["stores"]["aiml"] = 0
        try:
            out["stores"]["memory"] = (
                self.memory.count() if self.memory else 0)
        except Exception:
            out["stores"]["memory"] = 0
        try:
            out["stores"]["library_chunks"] = (
                self.library.count_chunks() if self.library else 0)
        except Exception:
            out["stores"]["library_chunks"] = 0
        try:
            from .cassettes import CassetteRack
            import os
            if self.cfg is not None:
                rack = CassetteRack(os.path.join(
                    os.path.abspath(self.cfg.data_dir),
                    self.cfg.cassette_dir))
                out["stores"]["cassettes"] = len(rack.list())
            else:
                out["stores"]["cassettes"] = 0
        except Exception:
            out["stores"]["cassettes"] = 0
        out["stores"]["oracle"] = "last-resort, not counted"
        return out

    # ---- per-store reads ------------------------------------------------
    def _read_calc(self, query: str) -> Frame | None:
        try:
            from . import calc as _calc
            text = _calc.try_calculate(query)
        except Exception:
            return None
        if not text:
            return None
        return Frame(text=text, store=STORE_CALC, question=query,
                     confidence=1.0, cost=0)

    def _read_aiml(self, query: str) -> Frame | None:
        if not self.aiml:
            return None
        try:
            text = self.aiml.respond(query)
        except Exception:
            return None
        if not text:
            return None
        return Frame(text=text, store=STORE_AIML, question=query,
                     confidence=1.0, cost=0)

    def _read_memory(self, query: str, top_k: int = 3) -> list[Frame]:
        if not self.memory:
            return []
        try:
            hits = self.memory.search(query, top_k=top_k)
        except Exception:
            return []
        floor = 0.0
        if self.cfg is not None:
            try:
                floor = float(self.cfg.vector_min_score)
            except Exception:
                floor = 0.0
        frames: list[Frame] = []
        for q, a, _score, sim in hits or []:
            try:
                conf = float(sim)
            except Exception:
                conf = 0.0
            if conf < floor:
                continue
            frames.append(Frame(text=a, store=STORE_MEMORY,
                                question=query, confidence=conf, cost=0,
                                metadata={"matched_question": q}))
        return frames

    def _read_library(self, query: str, top_k: int = 3) -> list[Frame]:
        if not self.library:
            return []
        try:
            hits = self.library.search(query, top_k=top_k)
        except Exception:
            return []
        frames: list[Frame] = []
        for h in hits or []:
            try:
                text = h.get("text", "") if isinstance(h, dict) else ""
                conf = float(h.get("score", 0.0)) if isinstance(h, dict) else 0.0
                meta = {"filename": h.get("filename", ""),
                        "chunk_no": h.get("chunk_no", "")} \
                    if isinstance(h, dict) else {}
            except Exception:
                continue
            if not text:
                continue
            frames.append(Frame(text=text, store=STORE_LIBRARY,
                                question=query, confidence=conf, cost=0,
                                metadata=meta))
        return frames

    # ---- READ -----------------------------------------------------------
    def read(self, query: str, intent: str = "factual",
             top_k_per_store: int = 3) -> BusResult:
        """One query across ALL local stores. Aggregates — never
        first-hit-wins. Returns frames + gaps (gap when no store
        hit at all). Never touches the oracle."""
        t0 = time.monotonic()
        frames: list[Frame] = []
        calc = self._read_calc(query)
        if calc is not None:
            calc.latency_ms = int((time.monotonic() - t0) * 1000)
            frames.append(calc)
        aiml = self._read_aiml(query)
        if aiml is not None:
            aiml.latency_ms = int((time.monotonic() - t0) * 1000)
            frames.append(aiml)
        frames.extend(self._read_memory(query, top_k=top_k_per_store))
        frames.extend(self._read_library(query, top_k=top_k_per_store))
        # Dedupe identical text across stores: keep the first
        # (cheapest / highest-priority) copy, note the other store.
        seen: dict[str, Frame] = {}
        deduped: list[Frame] = []
        for f in frames:
            key = _dedupe_key(f.text)
            if key in seen:
                seen[key].metadata.setdefault("also_in", []).append(f.store)
                continue
            seen[key] = f
            deduped.append(f)
        gaps: list[Gap] = []
        if not deduped:
            gaps.append(Gap(question=query, intent=intent, reason="miss",
                            stores_checked=[STORE_CALC, STORE_AIML,
                                            STORE_MEMORY, STORE_LIBRARY]))
            if self.gap_log is not None:
                try:
                    self.gap_log.note_gap(query)
                except Exception:
                    pass
        return BusResult(query=query, intent=intent,
                         frames=deduped, gaps=gaps)

    # ---- COMPILE ---------------------------------------------------------
    def compile(self, topic: str, intent: str = "article",
                subquestions: list[str] | None = None,
                fanout_fn=None) -> Compilation:
        """Compile-first: fan out, read EVERY store per sub-question,
        return local hits + explicit gaps. No oracle call here."""
        topic = (topic or "").strip()
        if not topic:
            raise ValueError("empty topic")
        if subquestions is not None:
            subs = [s.strip() for s in subquestions if s and s.strip()]
        elif fanout_fn is not None:
            subs = [s.strip() for s in fanout_fn(topic)
                    if s and s.strip()]
        else:
            subs = default_fanout(topic)
        frames: list[Frame] = []
        gaps: list[Gap] = []
        for q in subs:
            res = self.read(q, intent=intent)
            frames.extend(res.frames)
            gaps.extend(res.gaps)
        return Compilation(topic=topic, intent=intent, frames=frames,
                           gaps=gaps, subquestions=subs)

    def fill_gaps(self, compilation: Compilation, gap_fill_fn) -> Compilation:
        """Fill ONLY the gaps, via a caller-supplied function.

        gap_fill_fn(gap_questions: list[str]) -> list[str] | dict:
          list -> answers aligned with gap_questions, in order
          dict -> {question: answer}
        Returns a NEW Compilation with oracle frames appended and
        gaps cleared for every gap the fn answered. The bus never
        calls a model itself — the caller owns the oracle, its
        PII stripping, and its cost.
        """
        gaps = compilation.gap_questions()
        if not gaps or gap_fill_fn is None:
            return compilation
        answers = gap_fill_fn(gaps)
        if isinstance(answers, dict):
            pairs = [(q, answers.get(q, "")) for q in gaps]
        else:
            pairs = list(zip(gaps, list(answers or [])))
        new_frames = list(compilation.frames)
        remaining: list[Gap] = []
        answered = set()
        for q, ans in pairs:
            ans = (ans or "").strip()
            if ans and ans.upper() != "UNKNOWN":
                new_frames.append(Frame(text=ans, store=STORE_ORACLE,
                                        question=q, confidence=0.7,
                                        cost=1,
                                        metadata={"gap_fill": True}))
                answered.add(_norm(q))
        for g in compilation.gaps:
            if _norm(g.question) not in answered:
                remaining.append(g)
        return Compilation(topic=compilation.topic,
                           intent=compilation.intent,
                           frames=new_frames, gaps=remaining,
                           subquestions=list(compilation.subquestions))

    # ---- WRITE / PREFETCH / INVALIDATE -----------------------------------
    def write(self, frame: Frame) -> bool:
        """Write-back through the brain's learn gate when a brain is
        attached (conflict check + classifier live there). Without
        a brain, write straight to AIML + memory. Returns True if
        the frame stuck somewhere."""
        if not frame.text or not frame.question:
            return False
        if self.brain is not None:
            try:
                return bool(self.brain._learn(
                    frame.question, frame.text, frame.confidence))
            except Exception:
                return False
        ok = False
        if self.aiml:
            try:
                ok = bool(self.aiml.learn(frame.question, frame.text)) or ok
            except Exception:
                pass
        if self.memory:
            try:
                ok = bool(self.memory.add(
                    frame.question, frame.text,
                    frame.confidence)) or ok
            except Exception:
                pass
        return ok

    def prefetch(self, topic: str, intent: str = "article",
                 subquestions: list[str] | None = None) -> list[Gap]:
        """DMA: compile without oracle, return ONLY the gaps — the
        researcher's shopping list. The CPU never blocks on this."""
        return self.compile(topic, intent=intent,
                            subquestions=subquestions).gaps

    def invalidate(self, _query: str = "") -> int:
        """Stores are append-only in v1: invalidation is a no-op
        that says so. Returns 0 invalidated. Never silently drops
        knowledge — a future version will tombstone by signature."""
        return 0

    # ---- opcode dispatcher (microchip feel) ------------------------------
    def execute(self, opcode: str, *args, **kwargs):
        """execute("READ", query) / ("COMPILE", topic) / ("WRITE", frame)
        / ("PREFETCH", topic) / ("INVALIDATE", query). Unknown opcodes
        raise ValueError — fail-loud, like a bad opcode on a chip."""
        op = (opcode or "").upper()
        if op == OP_READ:
            return self.read(*args, **kwargs)
        if op == OP_COMPILE:
            return self.compile(*args, **kwargs)
        if op == OP_WRITE:
            return self.write(*args, **kwargs)
        if op == OP_PREFETCH:
            return self.prefetch(*args, **kwargs)
        if op == OP_INVALIDATE:
            return self.invalidate(*args, **kwargs)
        raise ValueError("unknown bus opcode: %r (want one of %s)"
                         % (opcode, ", ".join(OPCODES)))
