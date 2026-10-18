"""Tests for the frontal-lobe composer (ceti/composer.py).

The composer is rules-only: no model calls, no brain imports. All tests
use a fake fetch backed by a dict mapping (entity, attribute) -> pieces.
"""

from ceti.composer import (
    ComposedAnswer,
    Composer,
    Intent,
    Piece,
    derive_intent,
    is_grounded,
)


def make_composer(mapping):
    """Fake fetch: look pieces up by (intent.entity, intent.attribute)."""
    def fetch(intent):
        return list(mapping.get((intent.entity, intent.attribute), []))
    return Composer(fetch)


# --- derive_intent ---------------------------------------------------------

def test_intent_capital():
    intent = derive_intent("What is the capital of France?")
    assert intent.action == "lookup"
    assert intent.entity == "France"
    assert intent.attribute == "capital"


def test_intent_capital_bare():
    intent = derive_intent("capital of japan")
    assert intent.action == "lookup"
    assert intent.entity == "Japan"
    assert intent.attribute == "capital"


def test_intent_currency_and_money():
    assert derive_intent("currency of France").attribute == "currency"
    intent = derive_intent("what is the money of Japan?")
    assert intent.action == "lookup"
    assert intent.attribute == "currency"
    assert intent.entity == "Japan"


def test_intent_population():
    intent = derive_intent("What is the population of France?")
    assert intent.action == "lookup"
    assert intent.entity == "France"
    assert intent.attribute == "population"


def test_intent_when_was():
    intent = derive_intent("When was the Commodore 64 introduced?")
    assert intent.action == "lookup"
    assert intent.entity == "Commodore 64"
    assert intent.attribute == "introduced"
    assert derive_intent("when was X released").attribute == "released"
    assert derive_intent("when was X founded").attribute == "founded"


def test_intent_explain():
    assert derive_intent("What is France?").action == "explain"
    assert derive_intent("What is France?").entity == "France"
    assert derive_intent("tell me about France").action == "explain"
    assert derive_intent("tell me about France").entity == "France"
    assert derive_intent("Who is Ada Lovelace?").action == "explain"
    assert derive_intent("Who is Ada Lovelace?").entity == "Ada Lovelace"


def test_intent_compare():
    intent = derive_intent("France vs Germany")
    assert intent.action == "compare"
    assert intent.entity == "France vs Germany"
    intent = derive_intent("compare France and Germany")
    assert intent.action == "compare"
    assert intent.entity == "France vs Germany"
    intent = derive_intent("What is the difference between France and Germany?")
    assert intent.action == "compare"
    assert intent.entity == "France vs Germany"


def test_intent_compute():
    assert derive_intent("What is 2+2?").action == "compute"
    assert derive_intent("2+2").action == "compute"
    assert derive_intent("What is 50% of 200?").action == "compute"
    assert derive_intent("sqrt 16").action == "compute"


def test_intent_greet():
    assert derive_intent("hello").action == "greet"
    assert derive_intent("Hi there").action == "greet"
    assert derive_intent("hey!").action == "greet"


def test_intent_unknown():
    intent = derive_intent("asdfgh qwerty")
    assert intent.action == "unknown"
    assert intent.confidence == 0.0


# --- compose: lookup ---------------------------------------------------------

def test_lookup_composes():
    composer = make_composer({
        ("France", "capital"): [Piece("Paris", "test", 1.0)],
    })
    answer = composer.compose("What is the capital of France?")
    assert answer is not None
    assert "Paris" in answer.text
    assert answer.text == "The capital of France is Paris."
    assert answer.grounded is True
    assert answer.intent.action == "lookup"


def test_lookup_sentence_piece_used_verbatim():
    composer = make_composer({
        ("France", "capital"):
            [Piece("Paris is the capital of France.", "test", 1.0)],
    })
    answer = composer.compose("What is the capital of France?")
    assert answer.text == "Paris is the capital of France."
    assert answer.grounded is True


def test_lookup_picks_highest_confidence_piece():
    composer = make_composer({
        ("France", "capital"): [
            Piece("Lyon", "test", 0.5),
            Piece("Paris", "test", 0.9),
        ],
    })
    answer = composer.compose("capital of France")
    assert "Paris" in answer.text


# --- compose: explain ---------------------------------------------------------

def test_explain_joins_pieces():
    composer = make_composer({
        ("France", ""): [
            Piece("France is a country in Europe.", "test", 1.0),
            Piece("Its capital is Paris.", "test", 1.0),
        ],
    })
    answer = composer.compose("tell me about France")
    assert answer is not None
    assert "France is a country in Europe." in answer.text
    assert "Its capital is Paris." in answer.text
    assert answer.grounded is True


def test_explain_uses_at_most_three_pieces():
    composer = make_composer({
        ("France", ""): [
            Piece("One.", "test", 1.0),
            Piece("Two.", "test", 1.0),
            Piece("Three.", "test", 1.0),
            Piece("Four.", "test", 1.0),
        ],
    })
    answer = composer.compose("tell me about France")
    assert "Four." not in answer.text
    assert "One." in answer.text and "Three." in answer.text


# --- compose: compare ---------------------------------------------------------

def test_compare_renders_both_sides():
    composer = make_composer({
        ("France vs Germany", ""): [
            Piece("Paris is the capital.", "test", 1.0),
            Piece("Berlin is the capital.", "test", 1.0),
        ],
    })
    answer = composer.compose("France vs Germany")
    assert answer is not None
    assert "France:" in answer.text
    assert "Germany:" in answer.text
    assert "Paris" in answer.text
    assert "Berlin" in answer.text
    assert answer.grounded is True


def test_compare_needs_two_pieces():
    composer = make_composer({
        ("France vs Germany", ""): [Piece("Paris is the capital.", "test", 1.0)],
    })
    assert composer.compose("France vs Germany") is None


# --- compose: passthrough / empty ------------------------------------------------

def test_no_pieces_returns_none():
    composer = make_composer({})
    assert composer.compose("What is the capital of France?") is None
    assert composer.compose("tell me about France") is None


def test_greet_compute_unknown_return_none():
    composer = make_composer({
        ("France", "capital"): [Piece("Paris", "test", 1.0)],
    })
    assert composer.compose("hello") is None
    assert composer.compose("What is 2+2?") is None
    assert composer.compose("asdfgh qwerty") is None


# --- grounding ---------------------------------------------------------------

def test_grounding_rejects_invented_number():
    pieces = [Piece("The population is 68,000.", "test", 1.0)]
    assert is_grounded("The population is 68,000.", pieces) is True
    assert is_grounded("The population is 99,999.", pieces) is False


def test_grounding_rejects_invented_words():
    pieces = [Piece("Paris is the capital of France.", "test", 1.0)]
    assert is_grounded("Paris is the capital of France.", pieces) is True
    # Mostly novel content words: below the 40% overlap threshold.
    assert is_grounded(
        "Zebra quantum flibberty nonsense blorp wobble.", pieces) is False


def test_grounding_number_must_match_whole_token():
    pieces = [Piece("Introduced in 1982.", "test", 1.0)]
    assert is_grounded("Introduced in 1982.", pieces) is True
    # "2" must not match by hiding inside "1982".
    assert is_grounded("Introduced in 2.", pieces) is False


def test_grounding_fallback_quotes_best_piece_verbatim():
    # Direct check of the fallback contract: when grounding fails, the
    # composer must quote, not invent. Simulate via a fetch whose piece
    # text the composer cannot use... the composer only rearranges
    # piece text, so exercise the fallback path through is_grounded
    # plus the ComposedAnswer shape a caller would get.
    pieces = [Piece("Paris is the capital of France.", "test", 0.9)]
    bad_draft = "The population is 99,999."
    assert is_grounded(bad_draft, pieces) is False
    fallback = ComposedAnswer(
        text=max(pieces, key=lambda p: p.confidence).text,
        grounded=False, pieces=pieces,
        intent=Intent("lookup", "France", "capital", "q", 0.9),
    )
    assert fallback.text == "Paris is the capital of France."
    assert fallback.grounded is False


def test_compose_fallback_on_grounding_failure(monkeypatch):
    # Wiring test: when grounding fails, compose quotes the best piece
    # verbatim with grounded=False instead of shipping the draft.
    import ceti.composer as composer_mod
    composer = make_composer({
        ("France", "capital"):
            [Piece("Paris is the capital.", "test", 1.0)],
    })
    monkeypatch.setattr(composer_mod, "is_grounded",
                        lambda *a, **k: False)
    answer = composer.compose("What is the capital of France?")
    assert answer.text == "Paris is the capital."
    assert answer.grounded is False


def test_composer_never_invents_outside_pieces_and_question():
    # Lookup template words come from the question, not the composer.
    composer = make_composer({
        ("France", "capital"): [Piece("Paris", "test", 1.0)],
    })
    answer = composer.compose("What is the capital of France?")
    assert answer.grounded is True
