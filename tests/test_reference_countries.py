"""Validation tests for the geography/countries reference module.

Parses data/cassettes/geography/countries.aiml with stdlib xml.etree
ONLY -- deliberately does NOT import ceti.modules, which is built in
parallel and may not exist yet.

The module is PARTIAL coverage (restcountries.com v3.1 was deprecated;
v5 requires an API key), so coverage asserts the actual 30-country file,
not the ~195-country full target.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "cassettes"
    / "geography"
    / "countries.aiml"
)

EXPECTED_PREDICATES = {
    "capital",
    "currency",
    "currency_code",
    "region",
    "population",
    "languages",
    "iso_alpha2",
    "iso_alpha3",
}

# Spot-check table: (capital, currency, region) verified against the
# search-verified sources used to build the module (Wikipedia country
# infoboxes, ISO 3166-1 alpha-3 list, Statvoo/country.io).
SPOT_CHECKS = {
    "France": ("Paris", "Euro", "Europe"),
    "Japan": ("Tokyo", "Japanese yen", "Asia"),
    "Brazil": ("Brasilia", "Brazilian real", "Americas"),
    "Canada": ("Ottawa", "Canadian dollar", "Americas"),
    "Egypt": ("Cairo", "Egyptian pound", "Africa"),
    "India": ("New Delhi", "Indian rupee", "Asia"),
    "Australia": ("Canberra", "Australian dollar", "Oceania"),
    "Germany": ("Berlin", "Euro", "Europe"),
    "Kenya": ("Nairobi", "Kenyan shilling", "Africa"),
    "Mexico": ("Mexico City", "Mexican peso", "Americas"),
}


def _load():
    assert MODULE_PATH.is_file(), f"missing module file: {MODULE_PATH}"
    tree = ET.parse(MODULE_PATH)  # raises on invalid XML
    return tree.getroot()


def _facts_by_subject(root):
    by_subject = {}
    for fact in root.findall("fact"):
        subject = fact.get("subject")
        by_subject.setdefault(subject, {})[fact.get("predicate")] = fact.get("object")
    return by_subject


def test_file_exists_and_valid_xml():
    root = _load()
    assert root.tag == "module"


def test_module_header():
    root = _load()
    assert root.get("domain") == "geography"
    assert root.get("name") == "countries"
    assert root.get("version") == "1-partial"
    assert root.get("confidence") == "1.0"
    assert "PARTIAL" in (root.get("source") or "")


def test_coverage_partial():
    root = _load()
    subjects = {f.get("subject") for f in root.findall("fact")}
    # Partial file: 30 countries verified via search, not the ~195 full set.
    assert len(subjects) == 30
    assert len(subjects) >= 20


def test_every_country_has_all_predicates():
    root = _load()
    by_subject = _facts_by_subject(root)
    assert by_subject, "no facts found"
    for subject, preds in by_subject.items():
        assert set(preds) == EXPECTED_PREDICATES, (
            f"{subject}: missing/extra predicates: "
            f"missing={EXPECTED_PREDICATES - set(preds)} "
            f"extra={set(preds) - EXPECTED_PREDICATES}"
        )


def test_every_fact_nonempty():
    root = _load()
    facts = root.findall("fact")
    assert facts, "no <fact> elements found"
    for fact in facts:
        assert (fact.get("subject") or "").strip(), f"empty subject: {fact.attrib}"
        assert (fact.get("predicate") or "").strip(), f"empty predicate: {fact.attrib}"
        assert (fact.get("object") or "").strip(), f"empty object: {fact.attrib}"


def test_population_is_integer_string():
    root = _load()
    by_subject = _facts_by_subject(root)
    for subject, preds in by_subject.items():
        pop = preds["population"]
        assert pop.isdigit(), f"{subject}: population not an integer string: {pop!r}"
        assert int(pop) > 0, f"{subject}: population not positive: {pop!r}"


def test_iso_code_shapes():
    root = _load()
    by_subject = _facts_by_subject(root)
    for subject, preds in by_subject.items():
        assert len(preds["iso_alpha2"]) == 2, f"{subject}: bad iso_alpha2"
        assert preds["iso_alpha2"].isalpha() and preds["iso_alpha2"].isupper()
        assert len(preds["iso_alpha3"]) == 3, f"{subject}: bad iso_alpha3"
        assert preds["iso_alpha3"].isalpha() and preds["iso_alpha3"].isupper()
        assert len(preds["currency_code"]) == 3, f"{subject}: bad currency_code"
        assert preds["currency_code"].isalpha() and preds["currency_code"].isupper()


def test_spot_check_capitals():
    root = _load()
    by_subject = _facts_by_subject(root)
    assert by_subject["France"]["capital"] == "Paris"
    assert by_subject["Japan"]["capital"] == "Tokyo"
    assert by_subject["Brazil"]["capital"] == "Brasilia"
    assert by_subject["Canada"]["capital"] == "Ottawa"
    assert by_subject["Egypt"]["capital"] == "Cairo"


def test_spot_check_currencies():
    root = _load()
    by_subject = _facts_by_subject(root)
    assert by_subject["France"]["currency"] == "Euro"
    assert by_subject["Japan"]["currency"] == "Japanese yen"


def test_spot_check_ten_countries():
    root = _load()
    by_subject = _facts_by_subject(root)
    for country, (capital, currency, region) in SPOT_CHECKS.items():
        assert country in by_subject, f"missing spot-check country: {country}"
        assert by_subject[country]["capital"] == capital, country
        assert by_subject[country]["currency"] == currency, country
        assert by_subject[country]["region"] == region, country
