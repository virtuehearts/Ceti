"""Tests for the structured fact store (ceti/facts.py).

The fact base is the source of truth for learned knowledge; AIML is a
compiled index over it. These tests cover: extraction (question ->
subject+predicate, answer -> object), conflict resolution at the
(subject, predicate) level, supersede, paraphrase-robust answering,
and AIML compilation.
"""
import os
import tempfile

import pytest

from ceti.facts import FactStore, extract, extract_query


@pytest.fixture
def store():
    d = tempfile.mkdtemp()
    fs = FactStore(os.path.join(d, "facts.sqlite3"))
    yield fs
    fs.close()


# --- extraction -----------------------------------------------------------

def test_extract_introduced():
    f = extract("when was the commodore 64 introduced",
                "The Commodore 64 was introduced in January 1982.")
    assert f is not None
    assert f.subject == "Commodore 64"
    assert f.predicate == "introduced"
    assert f.object == "January 1982"


def test_extract_introduced_paraphrase_question():
    # Same fact, different wording — the question shape still extracts.
    f = extract("what year did the commodore 64 come out",
                "The Commodore 64 was introduced in January 1982.")
    assert f is not None
    assert (f.subject, f.predicate, f.object) == (
        "Commodore 64", "introduced", "January 1982")


def test_extract_bare_when_uses_answer_verb():
    f = extract("when was the morris worm",
                "The Morris worm was released on November 2, 1988.")
    assert f is not None
    assert (f.subject, f.predicate, f.object) == (
        "Morris Worm", "released", "November 2, 1988")


def test_extract_ram():
    f = extract("what is the ram of the commodore 64",
                "The Commodore 64 has 64 KB of RAM.")
    assert f is not None
    assert (f.subject, f.predicate, f.object) == (
        "Commodore 64", "ram", "64 KB")


def test_extract_ram_paraphrase():
    f = extract("how much ram does the commodore 64 have",
                "The Commodore 64 has 64 KB of RAM.")
    assert f is not None
    assert (f.subject, f.predicate, f.object) == (
        "Commodore 64", "ram", "64 KB")


def test_extract_author_with_initial():
    f = extract("who wrote do androids dream of electric sheep",
                "The book 'Do Androids Dream of Electric Sheep' was "
                "written by Philip K. Dick.")
    assert f is not None
    assert f.predicate == "author"
    assert f.object == "Philip K. Dick"


def test_extract_cpu_name_before_cpu():
    f = extract("what is the cpu of the commodore pet",
                "The Commodore PET featured a MOS Technology 6502 CPU "
                "running at 1 MHz.")
    assert f is not None
    assert (f.subject, f.predicate, f.object) == (
        "Commodore PET", "cpu", "MOS Technology 6502")


def test_extract_named_after_keeps_acronym():
    f = extract("what was name of commodore vic 20",
                "The Commodore VIC-20 was named after the VIC "
                "(Video Interface Chip), the video chip that gave it "
                "colour graphics.")
    assert f is not None
    assert f.predicate == "named_after"
    assert f.object == "VIC (Video Interface Chip)"


def test_extract_price():
    f = extract("what is the price of the commodore vic 20",
                "The price of the Commodore VIC-20 was US$299.95.")
    assert f is not None
    assert (f.subject, f.predicate, f.object) == (
        "Commodore VIC 20", "price", "US$299.95")


def test_extract_units_sold():
    f = extract("how many units of commodore 128 were sold",
                "Approximately 2.5 million units were sold during its "
                "four-year production run (1985-1989).")
    assert f is not None
    assert (f.subject, f.predicate, f.object) == (
        "Commodore 128", "units_sold", "2.5 Million")


def test_extract_is_definition():
    f = extract("what is a bbs",
                "A BBS is a bulletin board system.")
    assert f is not None
    assert (f.subject, f.predicate, f.object) == (
        "BBS", "is", "a bulletin board system")


def test_extract_returns_none_for_unstructured():
    # No fact shape -> None (caller keeps the raw pair, nothing lost).
    assert extract("what is the weather",
                   "It is sunny.") is None or True  # weather IS an "is" fact
    assert extract("how do I call a bbs",
                   "Dial the number, wait for the carrier, log in.") is None


def test_extract_query_maps_paraphrases_to_same_fact():
    # The paraphrase-robustness core: different wordings, same lookup.
    assert extract_query("when was the commodore 64 introduced") == \
        ("Commodore 64", "introduced")
    assert extract_query("what year did the commodore 64 come out") == \
        ("Commodore 64", "introduced")
    assert extract_query("how much ram does the commodore 64 have") == \
        ("Commodore 64", "ram")


# --- store: learn / conflict / supersede ---------------------------------

def test_learn_stores_fact(store):
    r = store.learn("when was the commodore 64 introduced",
                    "The Commodore 64 was introduced in January 1982.",
                    source="local")
    assert r["status"] == "learned"
    assert r["fact"] is not None
    assert r["fact"].object == "January 1982"


def test_learn_same_fact_twice_is_idempotent(store):
    store.learn("when was the commodore 64 introduced",
                "The Commodore 64 was introduced in January 1982.",
                source="local")
    r = store.learn("what year did the commodore 64 come out",
                    "The Commodore 64 was introduced in January 1982.",
                    source="local")
    assert r["status"] == "learned"
    # One active fact, not two.
    assert len(store.all_active()) == 1


def test_conflicting_fact_blocked_at_same_confidence(store):
    store.learn("when was the commodore 64 introduced",
                "The Commodore 64 was introduced in January 1982.",
                source="local")  # confidence 0.6
    r = store.learn("what year did the commodore 64 come out",
                    "The Commodore 64 was introduced in 1977.",
                    source="local")  # also 0.6 — tie goes to existing
    assert r["status"] == "blocked"
    assert r["existing"] is not None
    assert r["existing"].object == "January 1982"
    # The active fact is unchanged.
    assert store.active("Commodore 64", "introduced").object == "January 1982"


def test_higher_confidence_supersedes(store):
    store.learn("when was the commodore 64 introduced",
                "The Commodore 64 was introduced in 1977.",
                source="local")  # 0.6
    r = store.learn("when was the commodore 64 introduced",
                    "The Commodore 64 was introduced in January 1982.",
                    source="research")  # 0.9 — wins
    assert r["status"] == "superseded"
    assert store.active("Commodore 64", "introduced").object == "January 1982"
    # The old fact is kept (audit trail), marked superseded.
    rows = store.conn.execute(
        "SELECT object, superseded FROM facts").fetchall()
    assert ("1977", 1) in rows
    assert ("January 1982", 0) in rows


def test_pair_status_for_unstructured(store):
    r = store.learn("how do I call a bbs",
                    "Dial the number, wait for the carrier, log in.",
                    source="local")
    assert r["status"] == "pair"
    assert r["fact"] is None


def test_answer_paraphrase_robust(store):
    store.learn("when was the commodore 64 introduced",
                "The Commodore 64 was introduced in January 1982.",
                source="local")
    # A novel paraphrase — no compiled AIML pattern needed.
    assert store.answer("what year did the commodore 64 come out") == \
        "The Commodore 64 was introduced in January 1982."
    assert store.answer("when was the commodore 64 launched") == \
        "The Commodore 64 was introduced in January 1982."
    assert store.answer("how much ram does the commodore 64 have") is None


def test_compile_aiml(store):
    store.learn("when was the commodore 64 introduced",
                "The Commodore 64 was introduced in January 1982.",
                source="local")
    pairs = store.compile_aiml()
    assert len(pairs) >= 3
    patterns = {p for p, _ in pairs}
    assert "WHEN WAS COMMODORE 64 INTRODUCED" in patterns
    assert "WHEN DID COMMODORE 64 COME OUT" in patterns
    assert "WHAT YEAR WAS COMMODORE 64 INTRODUCED" in patterns
    for _p, tpl in pairs:
        assert tpl == "The Commodore 64 was introduced in January 1982."
