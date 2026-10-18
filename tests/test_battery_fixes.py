"""Regression tests for bugs found by the 0.3.2 release test battery.

BUG-1 (serious): the memory tier's fuzzy knowledge fallback answered a
fact-shaped question with a DIFFERENT entity's fact — after learning
one capital, every "What is the capital of <unknown>?" returned the
learned capital (word-overlap 0.83 clears the floor). The fix rejects
a fuzzy hit whose stored question extracts to the same predicate but
a different subject.

BUG-2: "What currency does Japan use?" / "What money does France use?"
returned nothing even though the module holds the fact — the fact
layer's subject extraction lacked the composer's "does X use" shape,
and MONEY mapped to no predicate.

BUG-3: one malformed cassette file made boot refuse to start (CORE
FAILURE) even though the AIML loader and the module loader both skip
corrupt files. Boot now skips unreadable cassettes loudly; corrupt
core files (seed/learned) still fail boot.

BUG-4: dictionary/thesaurus swallowed a fetch failure and reported
"no entry found" for words that plainly exist (transient API miss).
The error now says the API was unreachable when it was.
"""
import os
import socket
import tempfile
import unittest
from unittest import mock

from ceti.brain import Brain
from ceti.config import BrainConfig
from ceti.facts import extract_query


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _make_brain(tmp):
    cfg = BrainConfig(data_dir=tmp, memory_backend="sqlite")
    with open(os.path.join(tmp, "seed.aiml"), "w") as f:
        f.write('<?xml version="1.0"?><aiml version="1.0">'
                '<category><pattern>HELLO</pattern>'
                '<template>Hi.</template></category></aiml>')
    return Brain(config=cfg, local_backend=None, cloud_backends=[])


class TestCrossEntityFuzzyGuard(unittest.TestCase):
    def test_unknown_entity_capital_fails_loud(self):
        tmp = tempfile.mkdtemp()
        b = _make_brain(tmp)
        try:
            ok = b._learn("What is the capital of Testland?",
                          "The capital of Testland is Testville.",
                          0.9, source="local")
            self.assertTrue(ok)
            # Same entity still answers.
            ans = b.ask("What is the capital of Testland?")
            self.assertIn("Testville", ans.text)
            # Unknown entities must fail loud — never Testland's fact.
            for q in ("What is the capital of Zzxq?",
                      "What is the capital of Iceland?",
                      "What is the capital of Qwertyuiop?"):
                ans = b.ask(q)
                self.assertEqual(ans.tier, "failed", (q, ans.text))
                self.assertNotIn("Testville", ans.text)
                self.assertIn("couldn't find", ans.text.lower())
        finally:
            b.close()

    def test_streaming_rejects_cross_entity_hit(self):
        tmp = tempfile.mkdtemp()
        b = _make_brain(tmp)
        try:
            b._learn("What is the capital of Testland?",
                     "The capital of Testland is Testville.",
                     0.9, source="local")
            blocks = [t for tag, t in b.stream_tokens(
                "What is the capital of Zzxq?")
                if tag in ("block", "chunk")]
            self.assertNotIn("Testville", " ".join(blocks))
        finally:
            b.close()

    def test_legitimate_fuzzy_answer_survives(self):
        # A raw pair with no extractable fact still serves a
        # paraphrase through the fuzzy tier — the guard only
        # rejects same-predicate/different-entity hits.
        tmp = tempfile.mkdtemp()
        b = _make_brain(tmp)
        try:
            ok = b._learn(
                "What is the airspeed velocity of an unladen swallow?",
                "African or European?", 0.9, source="local")
            self.assertTrue(ok)
            ans = b.ask("What is the airspeed of an unladen swallow?")
            self.assertIn("African or European", ans.text)
        finally:
            b.close()


class TestCurrencyDoesUseShape(unittest.TestCase):
    def test_extract_query_currency_shapes(self):
        self.assertEqual(extract_query("What currency does Japan use?"),
                         ("Japan", "currency"))
        self.assertEqual(extract_query("What money does France use?"),
                         ("France", "currency"))
        self.assertEqual(extract_query("What is the money of Japan?"),
                         ("Japan", "currency"))
        self.assertEqual(extract_query("What is the currency of Japan?"),
                         ("Japan", "currency"))

    def test_module_fact_serves_does_use_shape(self):
        tmp = tempfile.mkdtemp()
        geo = os.path.join(tmp, "cassettes", "geography")
        os.makedirs(geo)
        with open(os.path.join(geo, "countries.aiml"), "w") as f:
            f.write(
                '<?xml version="1.0"?>\n'
                '<module domain="geography" name="countries" version="1" '
                'source="test" confidence="1.0">\n'
                '  <fact subject="Japan" predicate="currency" '
                'object="Japanese yen"/>\n'
                "</module>\n")
        b = _make_brain(tmp)
        try:
            ans = b.ask("What currency does Japan use?")
            self.assertEqual(ans.tier, "memory", ans.text)
            self.assertIn("Japanese yen", ans.text)
            ans2 = b.ask("What money does Japan use?")
            self.assertIn("Japanese yen", ans2.text)
        finally:
            b.close()


class TestBootSkipsBrokenCassette(unittest.TestCase):
    def _data_dir(self, broken_cassette=True, corrupt_seed=False):
        from ceti.boot import boot  # local import: boot reads env-free cfg
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "persona.md"), "w") as f:
            f.write("# Persona\nTest voice.\n")
        with open(os.path.join(d, "seed.aiml"), "w") as f:
            if corrupt_seed:
                f.write("<aiml><category><pattern>oops")
            else:
                f.write('<?xml version="1.0"?><aiml version="1.0">'
                        '<category><pattern>HELLO</pattern>'
                        '<template>Hi.</template></category></aiml>')
        with open(os.path.join(d, "learned.aiml"), "w") as f:
            f.write('<?xml version="1.0"?><aiml version="1.0"></aiml>')
        os.makedirs(os.path.join(d, "cassettes"), exist_ok=True)
        if broken_cassette:
            with open(os.path.join(d, "cassettes", "broken.aiml"),
                      "w") as f:
                f.write("<aiml><category><pattern>BROKEN")
        cfg = BrainConfig(
            data_dir=d,
            embed_url="http://127.0.0.1:%d" % _free_port(),
            ollama_url="http://127.0.0.1:%d" % _free_port())
        return d, boot(cfg)

    def test_broken_cassette_skipped_loudly(self):
        _d, rep = self._data_dir(broken_cassette=True)
        self.assertTrue(rep.ok, rep.text)
        by_name = {c.name: c for c in rep.checks}
        self.assertEqual(by_name["custom local AIML engine"].status, "ok")
        self.assertIn("broken.aiml",
                      by_name["custom local AIML engine"].detail)

    def test_corrupt_seed_still_fails_boot(self):
        _d, rep = self._data_dir(broken_cassette=False, corrupt_seed=True)
        self.assertFalse(rep.ok)
        by_name = {c.name: c for c in rep.checks}
        self.assertEqual(by_name["custom local AIML engine"].status,
                         "fail")


class TestC2OfflineHonesty(unittest.TestCase):
    def test_thesaurus_offline_unknown_says_unreachable(self):
        from ceti import tools
        with mock.patch("ceti.tools.urllib.request.urlopen",
                        side_effect=OSError("offline")):
            r = tools.execute("thesaurus", word="happy")
        self.assertFalse(r.success)
        self.assertIn("unreachable", r.error)
        self.assertIn("no entry found", r.error)

    def test_dictionary_offline_unknown_says_unreachable(self):
        from ceti import tools
        with mock.patch("ceti.tools.urllib.request.urlopen",
                        side_effect=OSError("offline")):
            r = tools.execute("dictionary", word="zzzq")
        self.assertFalse(r.success)
        self.assertIn("unreachable", r.error)
        self.assertIn("no definition", r.error)


if __name__ == "__main__":
    unittest.main()
