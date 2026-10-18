"""Tests for the CETI Modules loader (ceti/modules.py).

Modules are AIML extended with a <module> header, explicit <fact>
tags, provenance, and domains as directories. Tests use tmp dirs
only — no network, no real data dir.
"""
import gzip
import os

from ceti.modules import (
    Module,
    infer_domain,
    load_module,
    load_modules,
    module_facts,
)

HEADER_MODULE = """\
<module domain="geography" name="countries" version="1" source="restcountries" confidence="1.0">
  <fact subject="France" predicate="capital" object="Paris"/>
  <category><pattern>WHAT IS THE CAPITAL OF FRANCE</pattern><template>Paris.</template></category>
</module>
"""

LEGACY_AIML = """\
<aiml version="1.0">
  <category><pattern>HELLO</pattern><template>Hi there.</template></category>
  <category><pattern>WHAT IS UP</pattern><template>Not much.</template></category>
</aiml>
"""


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


# --- header parsing -------------------------------------------------------

def test_header_parsing(tmp_path):
    path = _write(str(tmp_path / "countries.aiml"), HEADER_MODULE)
    mod = load_module(path)
    assert mod is not None
    assert mod.domain == "geography"
    assert mod.name == "countries"
    assert mod.version == "1"
    assert mod.source == "restcountries"
    assert mod.confidence == 1.0
    assert mod.legacy is False
    assert mod.path == path


def test_fact_extraction(tmp_path):
    path = _write(str(tmp_path / "countries.aiml"), HEADER_MODULE)
    mod = load_module(path)
    assert mod.facts == [("France", "capital", "Paris")]


def test_category_patterns(tmp_path):
    path = _write(str(tmp_path / "countries.aiml"), HEADER_MODULE)
    mod = load_module(path)
    assert mod.patterns == [("WHAT IS THE CAPITAL OF FRANCE", "Paris.")]


def test_module_facts_carries_provenance(tmp_path):
    path = _write(str(tmp_path / "countries.aiml"), HEADER_MODULE)
    mod = load_module(path)
    assert module_facts(mod) == [
        {
            "subject": "France",
            "predicate": "capital",
            "object": "Paris",
            "source": "restcountries",
            "confidence": 1.0,
        }
    ]


def test_header_defaults(tmp_path):
    # Missing optional attributes fall back, never crash.
    path = _write(str(tmp_path / "bare.aiml"),
                  "<module><category><pattern>HI</pattern>"
                  "<template>Yo.</template></category></module>")
    mod = load_module(path)
    assert mod is not None
    assert mod.name == "bare"
    assert mod.version == "1"
    assert mod.confidence == 1.0
    assert mod.legacy is False


def test_fact_tags_with_missing_attrs_skipped(tmp_path):
    path = _write(str(tmp_path / "partial.aiml"),
                  '<module domain="geography">'
                  '<fact subject="France" predicate="capital"/>'
                  '<fact subject="France" predicate="capital" object="Paris"/>'
                  "</module>")
    mod = load_module(path)
    assert mod.facts == [("France", "capital", "Paris")]


# --- legacy fallback ------------------------------------------------------

def test_legacy_fallback(tmp_path):
    path = _write(str(tmp_path / "greetings.aiml"), LEGACY_AIML)
    mod = load_module(path)
    assert mod is not None
    assert mod.legacy is True
    assert mod.patterns == [("HELLO", "Hi there."), ("WHAT IS UP", "Not much.")]
    assert mod.confidence == 0.6
    assert mod.source == "legacy"
    assert mod.domain == "general"
    assert mod.name == "greetings"


def test_domain_inference_keywords():
    assert infer_domain("fidonet-nodes.aiml") == "bbs"
    assert infer_domain("sysop-guide.aiml") == "bbs"
    assert infer_domain("phrack-issues.aiml") == "bbs"
    assert infer_domain("commodore-64.aiml") == "computing"
    assert infer_domain("c64-games.aiml") == "computing"
    assert infer_domain("ansi-art.aiml") == "demoscene"
    assert infer_domain("hayes-commands.aiml") == "telecom"
    assert infer_domain("modem-setup.aiml") == "telecom"
    assert infer_domain("random-notes.aiml") == "general"


def test_legacy_domain_inferred_from_filename(tmp_path):
    path = _write(str(tmp_path / "fidonet-nodes.aiml"), LEGACY_AIML)
    mod = load_module(path)
    assert mod.legacy is True
    assert mod.domain == "bbs"


# --- gzip -----------------------------------------------------------------

def test_gzip_round_trip(tmp_path):
    path = str(tmp_path / "countries.aiml.gz")
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(HEADER_MODULE)
    mod = load_module(path)
    assert mod is not None
    assert mod.domain == "geography"
    assert mod.facts == [("France", "capital", "Paris")]
    assert mod.patterns == [("WHAT IS THE CAPITAL OF FRANCE", "Paris.")]


def test_gzip_legacy(tmp_path):
    path = str(tmp_path / "fidonet-nodes.aiml.gz")
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(LEGACY_AIML)
    mod = load_module(path)
    assert mod is not None
    assert mod.legacy is True
    assert mod.domain == "bbs"
    assert len(mod.patterns) == 2


# --- load_modules ---------------------------------------------------------

def test_load_modules_flat_and_subdir(tmp_path):
    _write(str(tmp_path / "flat.aiml"), LEGACY_AIML)
    _write(str(tmp_path / "geography" / "countries.aiml"), HEADER_MODULE)
    _write(str(tmp_path / "bbs" / "fidonet.aiml"), LEGACY_AIML)
    _write(str(tmp_path / "notes.txt"), "not a module")

    mods = load_modules(str(tmp_path))
    by_path = {os.path.basename(m.path): m for m in mods}
    assert set(by_path) == {"flat.aiml", "countries.aiml", "fidonet.aiml"}
    assert by_path["flat.aiml"].legacy is True
    assert by_path["countries.aiml"].legacy is False


def test_domain_from_subdir_name(tmp_path):
    # Precedence: explicit header domain wins; otherwise the subdir
    # name wins over filename inference. HEADER_MODULE carries
    # domain="geography" in its header, so it keeps "geography"
    # even inside a "custom" directory; the legacy file (no header)
    # takes the subdir name "history" over its fidonet inference.
    _write(str(tmp_path / "history" / "fidonet-nodes.aiml"), LEGACY_AIML)
    _write(str(tmp_path / "custom" / "countries.aiml"), HEADER_MODULE)
    mods = load_modules(str(tmp_path))
    by_name = {m.name: m for m in mods}
    assert by_name["fidonet-nodes"].domain == "history"
    assert by_name["countries"].domain == "geography"


def test_load_modules_skips_malformed(tmp_path):
    _write(str(tmp_path / "good.aiml"), LEGACY_AIML)
    _write(str(tmp_path / "bad.aiml"), "<aiml><category><pattern>oops")
    _write(str(tmp_path / "alsobad.aiml"), "this is not xml at all {{{")
    mods = load_modules(str(tmp_path))
    assert [m.name for m in mods] == ["good"]


def test_load_modules_missing_dir_returns_empty(tmp_path):
    assert load_modules(str(tmp_path / "no-such-dir")) == []


def test_load_modules_gz_in_subdir(tmp_path):
    path = str(tmp_path / "geography" / "countries.aiml.gz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(HEADER_MODULE)
    mods = load_modules(str(tmp_path))
    assert len(mods) == 1
    assert mods[0].domain == "geography"
    assert mods[0].facts == [("France", "capital", "Paris")]


# --- fail-closed ----------------------------------------------------------

def test_malformed_returns_none(tmp_path):
    path = _write(str(tmp_path / "bad.aiml"), "<aiml><category><pattern>oops")
    assert load_module(path) is None


def test_unreadable_returns_none(tmp_path):
    assert load_module(str(tmp_path / "does-not-exist.aiml")) is None


def test_corrupt_gzip_returns_none(tmp_path):
    path = _write(str(tmp_path / "bad.aiml.gz"), "not gzip bytes at all")
    assert load_module(path) is None


def test_empty_file_returns_none(tmp_path):
    # An empty file is a parse failure, so it fails closed like
    # malformed XML.
    path = _write(str(tmp_path / "empty.aiml"), "")
    assert load_module(path) is None


def test_module_dataclass_defaults():
    mod = Module(domain="general", name="x", version="1", source="s",
                 confidence=0.5)
    assert mod.facts == []
    assert mod.patterns == []
    assert mod.path == ""
    assert mod.legacy is False


def test_header_domain_beats_subdir(tmp_path):
    # Explicit <module domain="..."> wins over the directory name.
    sub = tmp_path / "other"
    sub.mkdir()
    path = _write(str(sub / "m.aiml"),
                  '<module domain="geography" name="m" version="1"'
                  ' source="s" confidence="1.0"></module>')
    assert load_module(path).domain == "geography"
    assert load_modules(str(tmp_path))[0].domain == "geography"


def test_subdir_beats_filename_inference(tmp_path):
    # No explicit header domain: the subdirectory wins over the
    # filename-inference keywords (fidonet -> bbs by inference).
    sub = tmp_path / "geography"
    sub.mkdir()
    path = _write(str(sub / "fidonet.aiml"),
                  "<aiml version=\"1.0\"></aiml>")
    assert load_modules(str(tmp_path))[0].domain == "geography"
