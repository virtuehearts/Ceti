"""Tests for the knowledge graph (ceti/graph.py).

The graph is a DERIVED INDEX built from module files (files are the
source of truth). It generalizes ceti/facts.py with the same
semantics: conflict at (subject, predicate), confidence, supersede,
provenance — plus dictionary encoding, entity lookup, and traversal.
"""
from types import SimpleNamespace

import pytest

from ceti.graph import KnowledgeGraph

C64_Q = "when was the commodore 64 introduced"
C64_A = "The Commodore 64 was introduced in January 1982."
C64_A_WRONG = "The Commodore 64 was introduced in 1977."


@pytest.fixture
def graph():
    return KnowledgeGraph()


# --- dictionary encoding --------------------------------------------------

def test_intern_stores_string_once(graph):
    assert graph.intern("Commodore 64") == graph.intern("Commodore 64")
    assert graph.stats()["strings"] == 1


def test_stats_prove_interning_saves_space(graph):
    # Three facts, same subject, distinct predicates/objects:
    # naive per-fact storage would hold 3*3 = 9 strings.
    graph.learn(C64_Q, C64_A)
    graph.learn("how much ram does the commodore 64 have",
                "The Commodore 64 has 64 KB of RAM.")
    graph.learn("what is the cpu of the commodore 64",
                "The Commodore 64 uses the MOS 6510 CPU.")
    stats = graph.stats()
    assert stats["facts"] == 3
    # 1 subject + 3 predicates + 3 objects = 7 distinct strings.
    assert stats["strings"] == 7
    assert stats["strings"] < 3 * stats["facts"]
    assert stats["dict_bytes"] > 0
    assert stats["dict_bytes"] == sum(
        len(s.encode("utf-8")) for s in graph._strings)


def test_blocked_fact_does_not_intern_its_object(graph):
    graph.learn(C64_Q, C64_A, score=0.8)
    before = graph.stats()["strings"]
    assert graph.learn(C64_Q, C64_A_WRONG, score=0.6) == "blocked"
    # The blocked object ("1977") was never stored.
    assert graph.stats()["strings"] == before
    assert "1977" not in graph._ids


# --- learn / lookup / answer ----------------------------------------------

def test_learn_and_lookup(graph):
    assert graph.learn(C64_Q, C64_A) == "learned"
    assert graph.lookup("Commodore 64", "introduced") == "January 1982"
    assert graph.lookup("Commodore 64", "ram") is None
    assert graph.lookup("Nobody", "introduced") is None


def test_learn_unstructured_returns_pair(graph):
    assert graph.learn("how do I call a bbs",
                       "Dial the number, wait for the carrier, log in.") == "pair"
    assert graph.all_active() == []


def test_answer_via_question(graph):
    graph.learn(C64_Q, C64_A)
    # Novel paraphrase — same (subject, predicate) via extract_query.
    assert graph.answer("what year did the commodore 64 come out") == \
        "The Commodore 64 was introduced in January 1982."
    assert graph.answer("how much ram does the commodore 64 have") is None
    assert graph.answer("how do I call a bbs") is None


def test_lookup_by_entity(graph):
    graph.learn(C64_Q, C64_A)
    graph.learn("how much ram does the commodore 64 have",
                "The Commodore 64 has 64 KB of RAM.")
    pairs = graph.lookup_by_entity("Commodore 64")
    assert ("introduced", "January 1982") in pairs
    assert ("ram", "64 KB") in pairs
    assert graph.lookup_by_entity("Nobody") == []


# --- conflict: blocked / superseded / idempotent --------------------------

def test_conflict_blocked_at_lower_confidence(graph):
    graph.learn(C64_Q, C64_A, score=0.8)
    assert graph.learn(C64_Q, C64_A_WRONG, score=0.6) == "blocked"
    assert graph.lookup("Commodore 64", "introduced") == "January 1982"


def test_conflict_blocked_at_equal_confidence(graph):
    graph.learn(C64_Q, C64_A, score=0.8)
    # Tie goes to the existing fact.
    assert graph.learn(C64_Q, C64_A_WRONG, score=0.8) == "blocked"
    assert graph.lookup("Commodore 64", "introduced") == "January 1982"


def test_higher_confidence_supersedes(graph):
    graph.learn(C64_Q, C64_A_WRONG, score=0.6)
    assert graph.learn(C64_Q, C64_A, score=0.9) == "superseded"
    assert graph.lookup("Commodore 64", "introduced") == "January 1982"
    # The old fact is kept (audit trail), marked superseded.
    assert ("Commodore 64", "introduced", "1977", 0.6, "local", True) \
        in graph.all_facts()
    # all_active shows the new fact, not the old one.
    assert ("Commodore 64", "introduced", "January 1982", 0.9, "local") \
        in graph.all_active()
    assert all(o != "1977" for _s, _p, o, _c, _src in graph.all_active())


def test_idempotent_relearn(graph):
    assert graph.learn(C64_Q, C64_A) == "learned"
    assert graph.learn("what year did the commodore 64 come out",
                       C64_A) == "learned"
    assert len(graph.all_active()) == 1
    assert graph.stats()["facts"] == 1


# --- add_fact / all_active -------------------------------------------------

def test_add_fact_and_all_active(graph):
    assert graph.add_fact("Paris", "located_in", "France",
                          0.9, "reference") == "learned"
    assert graph.lookup("Paris", "located_in") == "France"
    assert graph.all_active() == [
        ("Paris", "located_in", "France", 0.9, "reference")]
    # add_fact follows the same conflict rule.
    assert graph.add_fact("Paris", "located_in", "Germany",
                          0.5, "local") == "blocked"
    assert graph.add_fact("Paris", "located_in", "Germany",
                          1.0, "reference") == "superseded"
    assert graph.lookup("Paris", "located_in") == "Germany"


# --- traversal --------------------------------------------------------------

def _chain_graph():
    g = KnowledgeGraph()
    g.add_fact("Paris", "located_in", "France", 0.9, "reference")
    g.add_fact("France", "located_in", "Europe", 0.9, "reference")
    g.add_fact("Europe", "located_in", "Earth", 0.9, "reference")
    return g


def test_traverse_chain():
    g = _chain_graph()
    assert g.traverse("Paris", "located_in") == [
        ("Paris", "located_in", "France"),
        ("France", "located_in", "Europe"),
    ]


def test_traverse_depth_limits_steps():
    g = _chain_graph()
    assert g.traverse("Paris", "located_in", depth=1) == [
        ("Paris", "located_in", "France")]
    assert g.traverse("Paris", "located_in", depth=3) == [
        ("Paris", "located_in", "France"),
        ("France", "located_in", "Europe"),
        ("Europe", "located_in", "Earth"),
    ]


def test_traverse_unknown_subject_is_empty():
    assert _chain_graph().traverse("Nowhere", "located_in") == []


def test_traverse_wrong_relation_is_empty():
    assert _chain_graph().traverse("Paris", "capital_of") == []


def test_traverse_stops_at_cycle():
    g = KnowledgeGraph()
    g.add_fact("A", "next", "B", 1.0, "reference")
    g.add_fact("B", "next", "A", 1.0, "reference")
    steps = g.traverse("A", "next", depth=10)
    assert steps == [("A", "next", "B"), ("B", "next", "A")]


def test_traverse_any_relation_when_none():
    g = KnowledgeGraph()
    g.add_fact("Paris", "capital_of", "France", 1.0, "reference")
    g.add_fact("France", "located_in", "Europe", 1.0, "reference")
    assert g.traverse("Paris") == [
        ("Paris", "capital_of", "France"),
        ("France", "located_in", "Europe"),
    ]


# --- build_from_modules ------------------------------------------------------

def test_build_from_modules_reference_beats_learned():
    learned = SimpleNamespace(
        facts=[("Commodore 64", "introduced", "1977")],
        source="local", confidence=0.6)
    reference = SimpleNamespace(
        facts=[("Commodore 64", "introduced", "January 1982")],
        source="reference", confidence=1.0)
    g = KnowledgeGraph()
    counts = g.build_from_modules([learned, reference])
    assert counts == {"learned": 1, "blocked": 0, "superseded": 1}
    assert g.lookup("Commodore 64", "introduced") == "January 1982"
    # Reference wins even when ingested first (learned is then blocked).
    g2 = KnowledgeGraph()
    counts2 = g2.build_from_modules([reference, learned])
    assert counts2 == {"learned": 1, "blocked": 1, "superseded": 0}
    assert g2.lookup("Commodore 64", "introduced") == "January 1982"


def test_build_from_modules_dict_style_and_provenance():
    g = KnowledgeGraph()
    g.build_from_modules([{
        "facts": [("Paris", "located_in", "France")],
        "source": "geography", "confidence": 1.0,
    }])
    assert g.all_active() == [
        ("Paris", "located_in", "France", 1.0, "geography")]


def test_build_from_modules_two_modules_no_conflict():
    geo = SimpleNamespace(
        facts=[("Paris", "located_in", "France"),
               ("France", "located_in", "Europe")],
        source="geography", confidence=1.0)
    hist = SimpleNamespace(
        facts=[("Commodore 64", "introduced", "January 1982")],
        source="history", confidence=1.0)
    g = KnowledgeGraph()
    g.build_from_modules([geo, hist])
    assert len(g.all_active()) == 3
    assert g.traverse("Paris", "located_in") == [
        ("Paris", "located_in", "France"),
        ("France", "located_in", "Europe"),
    ]


# --- possessive vs module facts -------------------------------------------
# A conflicting fact stated possessively ("France's capital is ...")
# used to extract NO fact, so it slipped past the graph's conflict
# gate and overwrote the countries module's Paris as a raw pair.

def test_module_fact_beats_possessive_conflict():
    from pathlib import Path

    from ceti.modules import load_modules
    cassettes = (Path(__file__).resolve().parents[1]
                 / "data" / "cassettes")
    modules = load_modules(cassettes)
    assert modules, "repo cassettes must load"
    g = KnowledgeGraph()
    g.build_from_modules(modules)
    assert g.lookup("France", "capital") == "Paris"
    assert g.learn("What is France's capital?",
                   "France's capital is Berlin.", score=0.9) == "blocked"
    assert g.lookup("France", "capital") == "Paris"
    assert "Paris" in (g.answer("What is France's capital?") or "")
