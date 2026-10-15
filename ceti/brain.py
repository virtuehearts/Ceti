"""Brain — the framework facade.

    brain = Brain()                    # all defaults, low-spec
    ans = brain.ask("What is a BBS?")  # Answer(text, tier, latency_ms, ...)
    print(ans.tier)                    # aiml | memory | local | cloud

Pipeline per query:
  1. interpret intent (decision layer)
  2. AIML instant match            -> done in ms
  3. memory fuzzy match            -> done in ms
  4. local Qwen                    -> classify answer, maybe learn
  5. cloud (PII-stripped query)     -> classify answer, maybe learn

Learning persists the STRIPPED Q&A into learned.aiml + the memory
index, so next time the same question answers instantly with no LLM.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from . import calc, decide
from .aiml import AimlBrain
from .backends import (BackendError, ChatBackend, NvidiaBackend,
                       OllamaBackend, OpenRouterBackend)
from .cassettes import CassetteRack
from .classify import RelevanceClassifier
from .config import BrainConfig
from .memory import MemoryIndex
from .retry import retry
from .vectormem import EmbedError, VectorMemory
from . import pii

BASE_PROMPT = (
    "Answer directly and concisely. Do not mention being an AI language "
    "model unless asked. Never reveal or repeat personal data."
)

HANDLER_PROMPT = (
    "You are a handler, not an oracle. Answer the user's question using "
    "ONLY the knowledge below. Do not add facts from elsewhere. If the "
    "knowledge does not cover it, say so plainly in one sentence."
)

_NUM_PAT = re.compile(r"\d[\d,]*")


def _numbers_grounded(answer: str, knowledge: str, question: str) -> bool:
    """Every number the handler states must already appear in the
    retrieved knowledge (or the question). Small models confabulate
    dates and figures; this deterministic guard catches it and the
    caller falls back to the raw retrieved text. Conservative in the
    safe direction: a false alarm costs prettiness, never truth."""
    norm = lambda s: s.replace(",", "")
    known = {norm(n) for n in _NUM_PAT.findall(knowledge)}
    known |= {norm(n) for n in _NUM_PAT.findall(question)}
    return all(norm(n) in known for n in _NUM_PAT.findall(answer))


@dataclass
class Answer:
    text: str
    tier: str                      # aiml | memory | local | cloud | failed
    latency_ms: int
    intent: str = ""
    learned: bool = False          # was this answer promoted to AIML?
    classification: float | None = None
    pii_found: list[str] = field(default_factory=list)
    trace: dict = field(default_factory=dict)

    # User-facing tier names: everything served on the box presents as
    # [local] (the internal ladder is nobody's business); the cloud
    # fallback presents as [oracle].
    DISPLAY_TIERS = {"aiml": "local", "memory": "local", "local": "local",
                     "cloud": "oracle", "failed": "failed"}
    MAX_TPS = 499  # display cap: instant retrieval would otherwise read absurd

    @property
    def display_tier(self) -> str:
        return self.DISPLAY_TIERS.get(self.tier, self.tier)

    @property
    def tokens_per_sec(self) -> int | None:
        if self.tier == "failed":
            return None
        tokens = max(1, len(self.text.split()))
        secs = max(self.latency_ms / 1000.0, 1e-3)
        return min(self.MAX_TPS, int(tokens / secs))

    @property
    def display_tag(self) -> str:
        tag = f"[{self.display_tier}{' +learned' if self.learned else ''}]"
        tps = self.tokens_per_sec
        return f"{tag} ({tps} t/s)" if tps is not None else tag


class Brain:
    def __init__(self, config: BrainConfig | None = None,
                 local_backend: ChatBackend | None = None,
                 cloud_backends: list[ChatBackend] | None = None,
                 gap_log=None):
        self.cfg = config or BrainConfig()
        # gap_log: a TaskStore (or anything with note_gap). Failed
        # questions are logged as gaps — the agent's curiosity feed.
        self.gap_log = gap_log
        self.classifier = RelevanceClassifier(
            self.cfg.promote_threshold, self.cfg.adjudicate_low)

        self.rack = CassetteRack(os.path.join(
            os.path.abspath(self.cfg.data_dir), self.cfg.cassette_dir))

        self.aiml: AimlBrain | None = None
        if self.cfg.enable_aiml:
            paths = self.cfg.aiml_paths()
            self.aiml = AimlBrain(
                seed_path=paths[0], learned_path=paths[1],
                max_learned=self.cfg.max_learned_categories,
                extra_paths=self.rack.load_paths(self.cfg.cassettes))

        self.memory = None
        self.memory_kind = "none"
        if self.cfg.enable_memory:
            self.memory, self.memory_kind = self._make_memory()

        self.local = local_backend
        if self.local is None and self.cfg.enable_local:
            self.local = OllamaBackend(self.cfg.ollama_url,
                                       self.cfg.local_model,
                                       self.cfg.local_timeout)

        self.clouds: list[ChatBackend] = []
        if cloud_backends is not None:
            self.clouds = cloud_backends
        elif self.cfg.enable_cloud:
            for name in self.cfg.cloud_order:
                try:
                    if name == "openrouter":
                        self.clouds.append(OpenRouterBackend(
                            BrainConfig.openrouter_key(),
                            self.cfg.cloud_model,
                            self.cfg.openrouter_url,
                            self.cfg.cloud_timeout))
                    elif name == "nvidia":
                        self.clouds.append(NvidiaBackend(
                            BrainConfig.nvidia_key(),
                            self.cfg.cloud_model,
                            self.cfg.nvidia_url,
                            self.cfg.cloud_timeout))
                except BackendError:
                    continue  # no key: skip this cloud silently

        self.stats = {"aiml": 0, "memory": 0, "local": 0,
                      "cloud": 0, "failed": 0, "learned": 0,
                      "memory_backend": self.memory_kind,
                      "cassettes": len(self.rack.list())}
        self._learn_block_reason: str | None = None

        self._persona_text: str | None = None
        self._handled = False
        self._handler_ungrounded = False
        # node identity: generated on first run, reused forever. The
        # master key stays in identity.dat; only the public key is
        # ever shared. This is the identity that will sign network
        # messages, seed torrents, and serve the BBS front end once
        # regula_core plugs in.
        try:
            from . import signing as _signing
            self.identity = _signing.ensure_identity(
                self.cfg.data_dir)
        except Exception:
            self.identity = None  # ecdsa missing: brain runs, no signing

    # ---- persona --------------------------------------------------------
    def persona(self) -> str:
        if self._persona_text is None:
            p = os.path.join(os.path.abspath(self.cfg.data_dir),
                             self.cfg.persona_file)
            try:
                with open(p, encoding="utf-8") as f:
                    self._persona_text = f.read().strip()
            except OSError:
                self._persona_text = ""
        return self._persona_text

    def _system_prompt(self) -> str:
        p = self.persona()
        return (p + "\n\n" + BASE_PROMPT) if p else BASE_PROMPT

    def _make_memory(self):
        """Vector backend when the embedding server answers, else the
        stdlib TF-IDF index. 'vector' raises if the server is down."""
        want = self.cfg.memory_backend
        if want in ("auto", "vector"):
            vm = VectorMemory(self.cfg.knowledge_path(),
                              self.cfg.embed_url)
            try:
                if vm.ping():
                    return vm, "vector"
            except EmbedError:
                pass
            vm.close()
            if want == "vector":
                raise BackendError(
                    "memory_backend=vector but the embedding server"
                    " at %s is unreachable" % self.cfg.embed_url)
        return MemoryIndex(self.cfg.memory_path()), "sqlite"

    def reload_cassettes(self) -> int:
        """Rebuild the AIML brain so newly recorded cassettes take
        effect without restarting. Returns cassettes loaded."""
        if not self.aiml:
            return 0
        paths = self.cfg.aiml_paths()
        cassette_paths = self.rack.load_paths(self.cfg.cassettes)
        self.aiml = AimlBrain(
            seed_path=paths[0], learned_path=paths[1],
            max_learned=self.cfg.max_learned_categories,
            extra_paths=cassette_paths)
        self.stats["cassettes"] = len(cassette_paths)
        return len(cassette_paths)

    # ---- public API ---------------------------------------------------
    def ask(self, query: str, force_tier: str | None = None,
            learn: bool = True) -> Answer:
        t0 = time.monotonic()
        trace: dict = {}
        self._handled = False
        self._handler_ungrounded = False
        intent = decide.classify_intent(query)
        tiers = ([force_tier] if force_tier
                 else decide.plan(intent, self.cfg))
        trace["intent"] = intent
        trace["tiers"] = tiers

        _stripped_q, pii_found = pii.strip(query)
        trace["pii_found"] = pii_found

        for tier in tiers:
            try:
                if tier == "calc":
                    text = self._tier_calc(query)
                elif tier == "aiml":
                    text = self._tier_aiml(query)
                elif tier == "memory":
                    text = self._tier_memory(query)
                elif tier == "local":
                    text = self._tier_local(query)
                elif tier == "cloud":
                    text = self._tier_cloud(_stripped_q or query)
                else:
                    continue
            except BackendError as e:
                trace[tier] = f"error: {e}"
                continue  # fall through to the next tier
            if text:
                return self._finish(query, _stripped_q, text, tier,
                                    intent, learn, t0, trace, pii_found)
            trace[tier] = "miss"

        ms = int((time.monotonic() - t0) * 1000)
        self.stats["failed"] += 1
        if self.gap_log is not None:
            try:
                self.gap_log.note_gap(query)
            except Exception:
                pass  # gap logging must never break answering
        return Answer(
            text="I couldn't find or generate an answer for that.",
            tier="failed", latency_ms=ms, intent=intent,
            pii_found=pii_found, trace=trace)

    def stream_tokens(self, query: str, force_tier: str | None = None,
                      learn: bool = True):
        """Streaming version of ask(): yields ('tag', tag) then
        ('chunk' | 'block', text) events, then ('stream_end', args).
        Retrieval tiers yield one 'block' — the whole answer, for
        the caller to reveal at the capped 499 t/s. Model tiers
        yield real 'chunk' fragments as they arrive. 'heartbeat'
        means no chunk yet (pump again). 'error' is terminal: the
        partial text stays, no finish runs.
        This phase touches no sqlite (no learning, no classifier,
        no gap log, no stats) — the caller runs finish_stream(args)
        on the main thread after the stream ends."""
        t0 = time.monotonic()
        trace: dict = {}
        self._handled = False
        self._handler_ungrounded = False
        intent = decide.classify_intent(query)
        tiers = ([force_tier] if force_tier
                 else decide.plan(intent, self.cfg))
        trace["intent"] = intent
        trace["tiers"] = tiers
        stripped_q, pii_found = pii.strip(query)
        trace["pii_found"] = pii_found

        def end(tier, text):
            return {"query": query, "stripped_q": stripped_q,
                    "text": text, "tier": tier, "intent": intent,
                    "learn": learn, "t0": t0, "trace": trace,
                    "pii_found": pii_found}

        for tier in tiers:
            tag = f"[{Answer.DISPLAY_TIERS.get(tier, tier)}]"
            try:
                if tier == "calc":
                    text = self._tier_calc(query)
                    if text:
                        yield ("tag", tag)
                        yield ("block", text)
                        yield ("stream_end", end(tier, text))
                        return
                elif tier == "aiml":
                    text = self._tier_aiml(query)
                    if text:
                        yield ("tag", tag)
                        yield ("block", text)
                        yield ("stream_end", end(tier, text))
                        return
                elif tier == "memory":
                    if (yield from self._stream_memory(query, tag, end)):
                        return
                elif tier == "local":
                    if not self.local:
                        continue
                    streamed, _err = yield from self._stream_backend(
                        self.local, self._system_prompt(), query,
                        tag, end, tier, self.cfg.local_max_tokens,
                        self.cfg.local_temperature)
                    if streamed:
                        return
                elif tier == "cloud":
                    if (yield from self._stream_cloud(
                            stripped_q or query, tag, end)):
                        return
            except BackendError as e:
                trace[tier] = f"error: {e}"
                continue
        text = "I couldn't find or generate an answer for that."
        yield ("tag", "[failed]")
        yield ("block", text)
        yield ("stream_end", end("failed", text))

    def _stream_backend(self, backend, system, user, tag, end, tier,
                        max_tokens, temperature):
        """Stream one model call. Yields events; returns
        (streamed, error). streamed=True when an answer streamed
        (or a mid-stream error was yielded — terminal, the partial
        text stays, no finish runs). streamed=False + error set
        means a failure before the first chunk: the caller falls
        through to the next tier."""
        parts: list[str] = []
        tag_sent = False
        try:
            for ev in backend.complete_stream(
                    system, user, max_tokens=max_tokens,
                    temperature=temperature):
                if ev is None:
                    yield ("heartbeat", None)
                    continue
                if not tag_sent:
                    yield ("tag", tag)
                    tag_sent = True
                parts.append(ev)
                yield ("chunk", ev)
        except BackendError as e:
            if not tag_sent:
                return False, str(e)
            yield ("error", str(e))
            return True, None
        if not tag_sent:
            return False, "empty stream"
        yield ("stream_end", end(tier, "".join(parts)))
        return True, None

    def _stream_memory(self, query, tag, end):
        """Memory tier with handler streaming. The handler output
        is buffered and grounding-checked before anything is
        shown — an ungrounded handler answer must never reach the
        screen. Returns True when an answer was yielded."""
        if not self.memory:
            return False
        hits = self.memory.search(query, top_k=1)
        floor = (self.cfg.vector_min_score
                 if self.memory_kind == "vector"
                 else self.cfg.memory_min_score)
        if not hits or hits[0][3] < floor:
            return False
        knowledge = hits[0][1]
        if self.cfg.handler_mode and self.local:
            parts: list[str] = []
            try:
                for ev in self.local.complete_stream(
                        self._system_prompt() + "\n\n" + HANDLER_PROMPT,
                        "KNOWLEDGE:\n%s\n\nQUESTION: %s"
                        % (knowledge, query),
                        max_tokens=self.cfg.local_max_tokens,
                        temperature=0.2):
                    if ev is None:
                        yield ("heartbeat", None)
                        continue
                    parts.append(ev)
            except BackendError:
                pass  # fall back to the raw retrieved text
            presented = "".join(parts)
            if presented:
                if _numbers_grounded(presented, knowledge, query):
                    self._handled = True
                    yield ("tag", tag)
                    yield ("block", presented)
                    yield ("stream_end", end("memory", presented))
                    return True
                self._handler_ungrounded = True
        self._handled = False
        yield ("tag", tag)
        yield ("block", knowledge)
        yield ("stream_end", end("memory", knowledge))
        return True

    def _stream_cloud(self, stripped_query, tag, end):
        """Cloud tier: up to 2 attempts per backend, like ask()'s
        retry, then the next backend. Pre-chunk failures fall
        through; mid-stream failures are terminal errors. Raises
        the last error when every backend failed pre-chunk, so
        the ladder records it in the trace."""
        last: str | None = None
        for backend in self.clouds:
            for _ in range(2):
                streamed, err = yield from self._stream_backend(
                    backend, self._system_prompt(), stripped_query,
                    tag, end, "cloud", self.cfg.cloud_max_tokens,
                    self.cfg.cloud_temperature)
                if streamed:
                    return True
                last = err  # pre-chunk failure: retry same backend
        if last:
            raise BackendError(last)
        return False

    def finish_stream(self, args) -> Answer:
        """Main-thread finish for a completed stream: classification,
        learning, gap log, stats. Must run on the thread that owns
        the sqlite handles (the TUI main thread)."""
        if args["tier"] == "failed":
            ms = int((time.monotonic() - args["t0"]) * 1000)
            self.stats["failed"] += 1
            if self.gap_log is not None:
                try:
                    self.gap_log.note_gap(args["query"])
                except Exception:
                    pass  # gap logging must never break answering
            return Answer(
                text=args["text"], tier="failed", latency_ms=ms,
                intent=args["intent"], pii_found=args["pii_found"],
                trace=args["trace"])
        return self._finish(args["query"], args["stripped_q"],
                            args["text"], args["tier"],
                            args["intent"], args["learn"], args["t0"],
                            args["trace"], args["pii_found"])

    def _finish(self, query, stripped_q, text, tier, intent, learn,
                t0, trace, pii_found) -> Answer:
        ms = int((time.monotonic() - t0) * 1000)
        self.stats[tier] = self.stats.get(tier, 0) + 1
        if tier == "memory":
            trace["handled"] = self._handled
            trace["handler_ungrounded"] = self._handler_ungrounded
        learned = False
        score = None
        if learn and tier in ("local", "cloud") \
                and decide.learnable(intent):
            verdict = self.classifier.decide(query, text)
            score = verdict.score
            trace["classification"] = {
                "score": round(score, 3), "action": verdict.action,
                "reasons": verdict.reasons}
            do_learn = verdict.action == "promote"
            if verdict.action == "adjudicate" \
                    and self.cfg.adjudicate_with_llm and self.local:
                try:
                    judge = self.classifier.adjudicate(
                        stripped_q or query, text,
                        lambda prompt: self.local.complete(
                            "Reply with exactly YES or NO.", prompt,
                            max_tokens=8, temperature=0.0))
                    trace["classification"]["judge"] = judge
                    do_learn = judge
                except BackendError as e:
                    trace["classification"]["judge_error"] = str(e)
            # Truth gate: a relevance "promote" is not a truth
            # verdict. Factual answers get one verification call
            # before they enter permanent memory; a NO (or a
            # failed judge) blocks learning, fail-closed. Serving
            # is unchanged — the answer is still shown, only
            # learning is gated.
            if do_learn and intent == "factual" \
                    and self.cfg.adjudicate_with_llm and self.local:
                try:
                    ok = self.classifier.verify_truth(
                        stripped_q or query, text,
                        lambda prompt: self.local.complete(
                            "Reply with exactly YES or NO.", prompt,
                            max_tokens=8, temperature=0.0))
                    trace["truth_check"] = ok
                    do_learn = ok
                except BackendError as e:
                    trace["truth_check_error"] = str(e)
                    do_learn = False
            if do_learn:
                learned = self._learn(stripped_q or query, text, score)
                trace["learned"] = learned
                if not learned and self._learn_block_reason:
                    trace["learn_blocked"] = self._learn_block_reason
        return Answer(text=text, tier=tier, latency_ms=ms, intent=intent,
                      learned=learned, classification=score,
                      pii_found=pii_found, trace=trace)

    # ---- tiers ---------------------------------------------------------
    def _tier_calc(self, query: str) -> str | None:
        # Deterministic: math is computed in Python, never by a model.
        # Calc answers are never learned — they're recomputed every time.
        return calc.try_calculate(query)

    def _tier_aiml(self, query: str) -> str | None:
        if not self.aiml:
            return None
        return self.aiml.respond(query)

    def _tier_memory(self, query: str) -> str | None:
        if not self.memory:
            return None
        hits = self.memory.search(query, top_k=1)
        floor = (self.cfg.vector_min_score
                 if self.memory_kind == "vector"
                 else self.cfg.memory_min_score)
        if not hits or hits[0][3] < floor:
            return None
        knowledge = hits[0][1]
        # handler mode: the datasets answer; the model only wields them
        if self.cfg.handler_mode and self.local:
            try:
                presented = self.local.complete(
                    self._system_prompt() + "\n\n" + HANDLER_PROMPT,
                    "KNOWLEDGE:\n%s\n\nQUESTION: %s"
                    % (knowledge, query),
                    max_tokens=self.cfg.local_max_tokens,
                    temperature=0.2)
                if _numbers_grounded(presented, knowledge, query):
                    self._handled = True
                    return presented
                # the handler invented figures: fall back to the raw
                # retrieved text rather than serve a confabulation
                self._handler_ungrounded = True
            except BackendError:
                pass  # fall back to the raw retrieved text
        self._handled = False
        return knowledge

    def _tier_local(self, query: str) -> str | None:
        if not self.local:
            return None
        return self.local.complete(
            self._system_prompt(), query,
            max_tokens=self.cfg.local_max_tokens,
            temperature=self.cfg.local_temperature)

    def _tier_cloud(self, stripped_query: str) -> str | None:
        last: BackendError | None = None
        for backend in self.clouds:
            try:
                return retry(
                    lambda: backend.complete(
                        self._system_prompt(), stripped_query,
                        max_tokens=self.cfg.cloud_max_tokens,
                        temperature=self.cfg.cloud_temperature),
                    attempts=2, exceptions=(BackendError,))
            except BackendError as e:
                last = e
        if last:
            raise last
        return None

    # ---- learning ------------------------------------------------------
    @staticmethod
    def _norm_text(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "",
                                          s.lower())).strip()

    @classmethod
    def _answers_conflict(cls, a: str, b: str) -> bool:
        """True when two answers genuinely disagree. Equal answers
        agree; an elaboration (one contains the other) agrees.
        Everything else is a conflict — e.g. '1977' vs
        'January 1982'."""
        na, nb = cls._norm_text(a), cls._norm_text(b)
        if na == nb:
            return False
        if len(na) >= 10 and (na in nb or nb in na):
            return False
        return True

    def _find_conflict(self, question: str, answer: str) -> str | None:
        """Return a description of existing knowledge that answers
        the same question differently, or None. Checks the memory
        store (same normalized question) and the AIML engine
        (what it already answers for this question, cassettes
        included). A conflict blocks learning — curated knowledge
        is never silently overridden by a model guess."""
        norm_q = self._norm_text(question)
        if self.memory:
            try:
                for q, a, _s, _sim in self.memory.search(
                        question, top_k=3):
                    if (self._norm_text(q) == norm_q
                            and self._answers_conflict(a, answer)):
                        return f"memory already has {q!r} -> {a!r}"
            except Exception:
                pass  # search must never break learning
        if self.aiml:
            try:
                existing = self.aiml.respond(question)
            except Exception:
                existing = None
            if existing and self._answers_conflict(existing, answer):
                return f"AIML already answers {existing!r}"
        return None

    def _learn(self, question: str, answer: str, score: float) -> bool:
        """Persist the STRIPPED pair. Returns True if it stuck.
        A conflict with existing knowledge blocks the write and
        files a curiosity gap, so research can resolve it."""
        self._learn_block_reason = None
        conflict = self._find_conflict(question, answer)
        if conflict:
            self._learn_block_reason = conflict
            if self.gap_log is not None:
                try:
                    self.gap_log.note_gap(question)
                except Exception:
                    pass
            return False
        q = " ".join(question.split())
        a = " ".join(answer.split())
        if not q or not a or len(q) > 300:
            return False
        ok = False
        if self.aiml and self.aiml.learn(q, a):
            ok = True
        if self.memory:
            if self.memory.add(q, a, score):
                ok = True
            # learn acronym<->expansion pairs ("BBS <-> bulletin board
            # system") so paraphrases match later
            n_alias = self.memory.learn_aliases_from(a)
            if n_alias:
                self.stats["aliases"] = \
                    self.stats.get("aliases", 0) + n_alias
        if ok:
            self.stats["learned"] += 1
        return ok

    def close(self) -> None:
        if self.memory:
            self.memory.close()
