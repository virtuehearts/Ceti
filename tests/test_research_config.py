"""Research config tests: allowlist, interests, nightly budget."""
import os
import tempfile
import unittest

from ceti.research_config import (ResearchConfig, host_allowed,
                                  normalize_site)


class TestNormalizeSite(unittest.TestCase):
    def test_bare_domain(self):
        self.assertEqual(normalize_site("example.com"), "example.com")

    def test_strips_scheme_path_port(self):
        self.assertEqual(
            normalize_site("https://Sub.Example.COM:443/wiki/X"),
            "sub.example.com")

    def test_rejects_wildcard_empty_spaces(self):
        for bad in ("", "*.example.com", "exa mple.com", "   "):
            with self.assertRaises(ValueError):
                normalize_site(bad)


class TestHostAllowed(unittest.TestCase):
    def test_exact_and_subdomain(self):
        self.assertTrue(host_allowed("https://en.wikipedia.org/wiki/X",
                                     ["en.wikipedia.org"]))
        self.assertTrue(host_allowed("https://a.en.wikipedia.org/",
                                     ["en.wikipedia.org"]))

    def test_suffix_spoof_rejected(self):
        self.assertFalse(host_allowed("https://en.wikipedia.org.evil.com/",
                                      ["en.wikipedia.org"]))
        self.assertFalse(host_allowed("https://evil.com/",
                                      ["en.wikipedia.org"]))


class TestResearchConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = ResearchConfig(self.tmp)

    def test_defaults(self):
        self.assertEqual(self.cfg.sites, ["en.wikipedia.org"])
        self.assertEqual(self.cfg.interests, [])
        self.assertEqual(self.cfg.nightly,
                         {"max_topics": 5, "max_qa": 40,
                          "max_minutes": 120})
        self.assertTrue(self.cfg.wiki_only())

    def test_add_remove_site_persists(self):
        self.cfg.add_site("https://Example.com/")
        self.assertEqual(self.cfg.sites,
                         ["en.wikipedia.org", "example.com"])
        self.assertFalse(self.cfg.wiki_only())
        again = ResearchConfig(self.tmp)
        self.assertEqual(again.sites, self.cfg.sites)
        self.cfg.remove_site("example.com")
        self.assertEqual(self.cfg.sites, ["en.wikipedia.org"])

    def test_add_site_dedupes(self):
        self.cfg.add_site("example.com")
        self.cfg.add_site("example.com")
        self.assertEqual(self.cfg.sites.count("example.com"), 1)

    def test_remove_last_site_refused(self):
        with self.assertRaises(ValueError):
            self.cfg.remove_site("en.wikipedia.org")
        self.assertEqual(self.cfg.sites, ["en.wikipedia.org"])

    def test_remove_unknown_site(self):
        with self.assertRaises(ValueError):
            self.cfg.remove_site("nope.com")

    def test_interests(self):
        self.cfg.add_interest("  retro   computing ")
        self.assertEqual(self.cfg.interests, ["retro computing"])
        self.cfg.add_interest("retro computing")
        self.assertEqual(len(self.cfg.interests), 1)
        self.cfg.remove_interest("retro computing")
        self.assertEqual(self.cfg.interests, [])
        with self.assertRaises(ValueError):
            self.cfg.remove_interest("retro computing")
        with self.assertRaises(ValueError):
            self.cfg.add_interest("   ")

    def test_budget_validation(self):
        nb = self.cfg.set_budget(max_topics=2, max_qa=10)
        self.assertEqual(nb["max_topics"], 2)
        self.assertEqual(nb["max_qa"], 10)
        self.assertEqual(nb["max_minutes"], 120)
        for bad in (0, -1, "3", 2.5):
            with self.assertRaises(ValueError):
                self.cfg.set_budget(max_topics=bad)
        with self.assertRaises(ValueError):
            self.cfg.set_budget(nope=1)

    def test_corrupt_file_falls_back_to_defaults(self):
        with open(os.path.join(self.tmp, "research.json"), "w") as f:
            f.write("{not json")
        cfg = ResearchConfig(self.tmp)
        self.assertEqual(cfg.sites, ["en.wikipedia.org"])


if __name__ == "__main__":
    unittest.main()
