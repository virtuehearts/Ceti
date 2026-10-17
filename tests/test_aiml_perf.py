"""Tests for the AIML engine optimization (ceti/aiml.py).

The wildcard index (first literal word) and the match cache must
preserve exact behaviour: exact > wildcard, specificity order, srai,
<random> still varies per call, learn() invalidates the cache.
"""
import os
import tempfile

from ceti.aiml import AimlBrain


def _brain():
    d = tempfile.mkdtemp()
    seed = os.path.join(d, "seed.aiml")
    with open(seed, "w") as f:
        f.write("""<aiml>
<category><pattern>WHAT IS A BBS</pattern><template>A BBS is a bulletin board system.</template></category>
<category><pattern>WHAT IS *</pattern><template>What is <star/>? I do not know.</template></category>
<category><pattern>HELLO *</pattern><template><random><li>Hi there.</li><li>Hello!</li></random></template></category>
<category><pattern>HI</pattern><template><srai>HELLO THERE</srai></template></category>
<category><pattern>* IS GREAT</pattern><template>Yes, it is great.</template></category>
</aiml>""")
    return AimlBrain(seed_path=seed,
                     learned_path=os.path.join(d, "learned.aiml"))


def test_exact_beats_wildcard():
    b = _brain()
    assert b.respond("what is a bbs") == "A BBS is a bulletin board system."


def test_wildcard_first_word_index():
    b = _brain()
    assert b.respond("what is a modem") == "What is MODEM? I do not know."


def test_wildcard_leading_star_bucket():
    b = _brain()
    assert b.respond("python is great") == "Yes, it is great."


def test_srai_still_works():
    b = _brain()
    assert b.respond("hi") in ("Hi there.", "Hello!")


def test_random_still_varies_despite_match_cache():
    b = _brain()
    seen = {b.respond("hello world") for _ in range(30)}
    assert seen == {"Hi there.", "Hello!"}


def test_learn_invalidates_match_cache():
    b = _brain()
    assert b.respond("what is a modem") == "What is MODEM? I do not know."
    b.learn("what is a modem", "A modem modulates a digital signal.")
    assert b.respond("what is a modem") == \
        "A modem modulates a digital signal."
    # The cached wildcard match must not leak into other inputs.
    assert b.respond("what is a bbs") == "A BBS is a bulletin board system."


def test_learn_compiled_registers_without_file():
    b = _brain()
    b.learn_compiled("WHEN WAS COMMODORE 64 INTRODUCED",
                     "The Commodore 64 was introduced in January 1982.")
    assert b.respond("when was the commodore 64 introduced") == \
        "The Commodore 64 was introduced in January 1982."
    # The learned file is untouched (compiled patterns are in-memory).
    with open(b.learned_path) as f:
        assert "COMMODORE 64" not in f.read()


def test_learn_compiled_override_replaces_template():
    b = _brain()
    b.learn_compiled("WHEN WAS COMMODORE 64 INTRODUCED",
                     "The Commodore 64 was introduced in 1977.")
    b.learn_compiled("WHEN WAS COMMODORE 64 INTRODUCED",
                     "The Commodore 64 was introduced in January 1982.")
    assert b.respond("when was the commodore 64 introduced") == \
        "The Commodore 64 was introduced in January 1982."
