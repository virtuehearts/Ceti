"""Tests for the Unified Memory Bus (ceti/bus.py) + compile-first articles.

No network, no embedding server, no oracle — fake stores + fake
gap-fill. The bus itself never calls a model.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ceti.bus import (Frame, Gap, MemoryBus, default_fanout,
                      OP_READ, OP_COMPILE, OP_WRITE, OP_PREFETCH,
                      OP_INVALIDATE)
from ceti.config import BrainConfig
from ceti.story import ArticleCompiler, StoryError


class FakeAiml:
    def __init__(self, answers=None):
        self.answers = answers or {}
        self.learned = {}

    def respond(self, query):
        return self.answers.get(query)

    def learn(self, q, a):
        self.learned[q] = a
        return True

    def count(self):
        return len(self.answers) + len(self.learned)


class FakeMemory:
    def __init__(self, hits=None):
        # hits: {query: [(q, a, score, sim)]}
        self.hits = hits or {}
        self.added = {}

    def search(self, query, top_k=3):
        return self.hits.get(query, [])[:top_k]

    def add(self, q, a, score=0.0):
        self.added[q] = a
        return True

    def count(self):
        return len(self.hits) + len(self.added)


class FakeLibrary:
    def __init__(self, hits=None):
        self.hits = hits or {}

    def search(self, query, top_k=5):
        return self.hits.get(query, [])[:top_k]

    def count_chunks(self):
        return sum(len(v) for v in self.hits.values())


def make_bus(**kw):
    return MemoryBus(**kw)


class TestFrameGap(unittest.TestCase):
    def test_frame_to_dict(self):
        f = Frame(text="1982", store="aiml", question="When?",
                  confidence=1.0)
        d = f.to_dict()
        self.assertEqual(d["store"], "aiml")
        self.assertEqual(d["text"], "1982")

    def test_gap_to_dict(self):
        g = Gap(question="Q?", reason="miss")
        self.assertEqual(g.to_dict()["question"], "Q?")


class TestBusRead(unittest.TestCase):
    def test_aggregates_all_stores_not_first_hit(self):
        # AIML + memory + library ALL answer — old ladder would stop at AIML.
        bus = make_bus(
            aiml=FakeAiml({"What is C64?": "Commodore 64 home computer."}),
            memory=FakeMemory({"What is C64?": [
                ("What is C64?", "8-bit home computer, 1982.", 0.9, 0.85)]}),
            library=FakeLibrary({"What is C64?": [
                {"text": "MOS 6510 CPU.", "score": 0.8,
                 "filename": "c64.md", "chunk_no": 0}]}),
            cfg=BrainConfig(data_dir=tempfile.mkdtemp()))
        res = bus.read("What is C64?")
        stores = sorted(f.store for f in res.frames)
        self.assertEqual(stores, ["aiml", "library", "memory"])
        self.assertEqual(res.gaps, [])
        self.assertEqual(res.best.store, "aiml")  # highest confidence wins

    def test_miss_is_gap_not_empty(self):
        bus = make_bus(aiml=FakeAiml({}), memory=FakeMemory({}),
                       library=FakeLibrary({}),
                       cfg=BrainConfig(data_dir=tempfile.mkdtemp()))
        res = bus.read("What is the airspeed of a laden swallow?")
        self.assertEqual(res.frames, [])
        self.assertEqual(len(res.gaps), 1)
        self.assertEqual(res.gaps[0].reason, "miss")

    def test_dedupe_across_stores(self):
        bus = make_bus(
            aiml=FakeAiml({"Q?": "Same answer."}),
            memory=FakeMemory({"Q?": [("Q?", "Same answer.", 0.9, 0.9)]}),
            cfg=BrainConfig(data_dir=tempfile.mkdtemp()))
        res = bus.read("Q?")
        self.assertEqual(len(res.frames), 1)
        self.assertIn("memory", res.frames[0].metadata.get("also_in", []))

    def test_memory_below_floor_dropped(self):
        cfg = BrainConfig(data_dir=tempfile.mkdtemp())
        cfg.vector_min_score = 0.70
        bus = make_bus(
            memory=FakeMemory({"Q?": [("Q?", "Weak.", 0.5, 0.40)]}),
            cfg=cfg)
        res = bus.read("Q?")
        self.assertEqual(res.frames, [])
        self.assertEqual(len(res.gaps), 1)

    def test_graph_store_answers_module_facts(self):
        # The bus must consult the graph/fact store: without it,
        # module facts (countries) reported as gaps and went to the
        # oracle even though the brain knew them locally.
        from types import SimpleNamespace

        class FakeGraph:
            def answer(self, q):
                if "capital of france" in q.lower():
                    return "The capital of France is Paris."
                return None

            def all_active(self):
                return [("France", "capital", "Paris", 1.0, "countries")]

        brain = SimpleNamespace(_graph=FakeGraph(), _facts=None)
        bus = make_bus(aiml=FakeAiml({}), memory=FakeMemory({}),
                       library=FakeLibrary({}),
                       cfg=BrainConfig(data_dir=tempfile.mkdtemp()),
                       brain=brain)
        res = bus.read("What is the capital of France?")
        self.assertTrue(res.hit)
        self.assertEqual(res.frames[0].store, "graph")
        self.assertIn("Paris", res.frames[0].text)
        self.assertEqual(res.gaps, [])

    def test_graph_fallback_to_fact_store(self):
        from types import SimpleNamespace

        class FakeFacts:
            def answer(self, q):
                if "capital of france" in q.lower():
                    return "The capital of France is Paris."
                return None

        brain = SimpleNamespace(_graph=None, _facts=FakeFacts())
        bus = make_bus(aiml=FakeAiml({}), memory=FakeMemory({}),
                       cfg=BrainConfig(data_dir=tempfile.mkdtemp()),
                       brain=brain)
        res = bus.read("What is the capital of France?")
        self.assertTrue(res.hit)
        self.assertEqual(res.frames[0].store, "graph")

    def test_graph_miss_lists_graph_checked(self):
        from types import SimpleNamespace

        class FakeGraph:
            def answer(self, q):
                return None

            def all_active(self):
                return []

        brain = SimpleNamespace(_graph=FakeGraph(), _facts=None)
        bus = make_bus(aiml=FakeAiml({}), memory=FakeMemory({}),
                       cfg=BrainConfig(data_dir=tempfile.mkdtemp()),
                       brain=brain)
        res = bus.read("What is the airspeed of a laden swallow?")
        self.assertEqual(res.frames, [])
        self.assertIn("graph", res.gaps[0].stores_checked)

    def test_graph_reader_never_raises(self):
        from types import SimpleNamespace

        class BadGraph:
            def answer(self, q):
                raise RuntimeError("boom")

            def all_active(self):
                raise RuntimeError("boom")

        brain = SimpleNamespace(_graph=BadGraph(), _facts=None)
        bus = make_bus(aiml=FakeAiml({}), memory=FakeMemory({}),
                       cfg=BrainConfig(data_dir=tempfile.mkdtemp()),
                       brain=brain)
        res = bus.read("What is the capital of France?")
        self.assertEqual(res.frames, [])
        self.assertEqual(len(res.gaps), 1)
        self.assertEqual(bus.post()["stores"]["graph"], 0)


class TestBusCompile(unittest.TestCase):
    def test_compile_hits_plus_gaps(self):
        bus = make_bus(
            aiml=FakeAiml({"When was Commodore 64 released?":
                           "1982."}),
            memory=FakeMemory({}),
            library=FakeLibrary({}),
            cfg=BrainConfig(data_dir=tempfile.mkdtemp()))
        subs = ["When was Commodore 64 released?",
                "How did Commodore 64 fare against the Amiga?"]
        comp = bus.compile("Commodore 64", subquestions=subs)
        self.assertEqual(len(comp.frames), 1)
        self.assertEqual(len(comp.gaps), 1)
        self.assertIn("Amiga", comp.gaps[0].question)
        self.assertAlmostEqual(comp.local_coverage, 0.5)

    def test_default_fanout_is_deterministic(self):
        self.assertEqual(default_fanout("X"), default_fanout("X"))
        self.assertEqual(len(default_fanout("Commodore 64")), 6)

    def test_fill_gaps_only_calls_gaps(self):
        bus = make_bus(
            aiml=FakeAiml({"When was Commodore 64 released?": "1982."}),
            cfg=BrainConfig(data_dir=tempfile.mkdtemp()))
        comp = bus.compile("Commodore 64", subquestions=[
            "When was Commodore 64 released?",
            "How did Commodore 64 fare against the Amiga?"])
        seen = []

        def fake_fill(gaps):
            seen.extend(gaps)
            return ["It outsold the Amiga in units."]

        filled = bus.fill_gaps(comp, fake_fill)
        # gap-fill fn saw ONLY the gap, never the local hit
        self.assertEqual(seen, ["How did Commodore 64 fare against the Amiga?"])
        self.assertEqual(filled.gaps, [])
        oracle_frames = [f for f in filled.frames if f.store == "oracle"]
        self.assertEqual(len(oracle_frames), 1)
        self.assertEqual(oracle_frames[0].cost, 1)

    def test_fill_gaps_keeps_local_coverage(self):
        # After a fill, local_coverage must still read the LOCAL
        # share — oracle frames never count as local hits.
        bus = make_bus(
            aiml=FakeAiml({"When was Commodore 64 released?": "1982."}),
            cfg=BrainConfig(data_dir=tempfile.mkdtemp()))
        comp = bus.compile("Commodore 64", subquestions=[
            "When was Commodore 64 released?",
            "How did Commodore 64 fare against the Amiga?"])
        self.assertAlmostEqual(comp.local_coverage, 0.5)
        filled = bus.fill_gaps(comp, lambda gaps: ["It outsold it."])
        self.assertAlmostEqual(filled.local_coverage, 0.5)

    def test_fill_gaps_unknown_stays_gap(self):
        bus = make_bus(cfg=BrainConfig(data_dir=tempfile.mkdtemp()))
        comp = bus.compile("X", subquestions=["Unknown Q?"])
        filled = bus.fill_gaps(comp, lambda gaps: ["UNKNOWN"])
        self.assertEqual(len(filled.gaps), 1)
        self.assertEqual(
            [f for f in filled.frames if f.store == "oracle"], [])

    def test_prefetch_returns_gaps_only(self):
        bus = make_bus(cfg=BrainConfig(data_dir=tempfile.mkdtemp()))
        gaps = bus.prefetch("X", subquestions=["Q1?", "Q2?"])
        self.assertEqual(len(gaps), 2)

    def test_invalidate_is_noop_fail_loud(self):
        bus = make_bus()
        self.assertEqual(bus.invalidate("anything"), 0)

    def test_execute_opcodes(self):
        bus = make_bus(aiml=FakeAiml({"Hi": "Hello."}))
        self.assertEqual(len(bus.execute(OP_READ, "Hi").frames), 1)
        self.assertEqual(
            bus.execute(OP_COMPILE, "T", subquestions=["Hi"]).frames[0].store,
            "aiml")
        # PREFETCH with a miss returns the gap
        gaps = bus.execute(OP_PREFETCH, "T", subquestions=["Nope?"])
        self.assertEqual(len(gaps), 1)
        self.assertEqual(bus.execute(OP_INVALIDATE, "x"), 0)
        with self.assertRaises(ValueError):
            bus.execute("BOGUS", "x")

    def test_post_counts(self):
        bus = make_bus(aiml=FakeAiml({"A": "B"}),
                       memory=FakeMemory({"Q": []}),
                       library=FakeLibrary({"Q": [{"text": "t"}]}),
                       cfg=BrainConfig(data_dir=tempfile.mkdtemp()))
        post = bus.post()
        self.assertEqual(post["stores"]["aiml"], 1)
        self.assertEqual(post["stores"]["library_chunks"], 1)


class TestArticleCompiler(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ceti-bus-article-")
        self.cfg = BrainConfig(data_dir=self.tmp)

    def _compiler(self, bus, backend=None):
        return ArticleCompiler(self.cfg, bus, backend=backend)

    def test_local_only_no_backend(self):
        bus = make_bus(
            aiml=FakeAiml({"When was Commodore 64 released?": "1982."}),
            cfg=self.cfg)
        comp = self._compiler(bus, backend=None)
        r = comp.compile_article("Commodore 64", subquestions=[
            "When was Commodore 64 released?",
            "How did Commodore 64 fare against the Amiga?"])
        self.assertEqual(r["local_coverage"], 0.5)
        self.assertEqual(r["gaps_open"], 1)
        self.assertEqual(r["gaps_filled"], 0)
        with open(r["md"]) as f:
            text = f.read()
        self.assertIn("1982", text)
        self.assertIn("Open gaps", text)
        self.assertIn("Amiga", text)
        # no oracle text invented when there is no backend
        self.assertNotIn("[oracle", text)

    def test_gap_fill_only_via_backend(self):
        class FakeOracle:
            def __init__(self):
                self.seen = []

            def complete(self, system, user, max_tokens=400,
                         temperature=0.2):
                self.seen.append(user)
                return "It outsold the Amiga in units."

        oracle = FakeOracle()
        bus = make_bus(
            aiml=FakeAiml({"When was Commodore 64 released?": "1982."}),
            cfg=self.cfg)
        r = self._compiler(bus, backend=oracle).compile_article(
            "Commodore 64", subquestions=[
                "When was Commodore 64 released?",
                "How did Commodore 64 fare against the Amiga?"])
        self.assertEqual(r["gaps_filled"], 1)
        self.assertEqual(r["gaps_open"], 0)
        # oracle saw ONLY the gap question, never the local hit
        self.assertEqual(len(oracle.seen), 1)
        self.assertIn("Amiga", oracle.seen[0])
        self.assertNotIn("1982", oracle.seen[0])
        with open(r["md"]) as f:
            text = f.read()
        self.assertIn("[oracle gap-fill]", text)
        self.assertIn("1982", text)

    def test_empty_topic_refuses(self):
        bus = make_bus(cfg=self.cfg)
        with self.assertRaises(StoryError):
            self._compiler(bus).compile_article("   ")


if __name__ == "__main__":
    unittest.main()
