"""research_config — what CETI is allowed to research, and how much.

One JSON file in the data dir: ``data/research.json``.

    {
      "sites": ["en.wikipedia.org"],       // content is fetched ONLY
                                            // from these hosts (exact
                                            // host or any subdomain)
      "interests": ["retro computing"],     // evergreen topics: when the
                                            // topics queue runs dry the
                                            // researcher cycles these
      "nightly": {"max_topics": 5,          // per-night budget, enforced
                  "max_qa": 40,             // by run_nightly() and the
                  "max_minutes": 120}        // nightly driver
    }

The allowlist is the rule that replaced the old Wikipedia-only switch:
Jero adds a site explicitly, and only then may the fetcher touch it.
No site is ever fetched because a search engine happened to return it.
The research stack enforces the list with its --allow-sites flag; the
Researcher passes it on every run (or --wiki-only for the Wikipedia
default, same behaviour, cheaper). An empty list is refused — the
config must always name at least one site, fail-closed.
"""
from __future__ import annotations

import json
import os
import urllib.parse

DEFAULT_SITES = ["en.wikipedia.org"]
DEFAULT_NIGHTLY = {"max_topics": 5, "max_qa": 40, "max_minutes": 120}
FILENAME = "research.json"


def normalize_site(raw: str) -> str:
    """'https://en.wikipedia.org/wiki/X' -> 'en.wikipedia.org'.

    Strips scheme, path, query, fragment, port, trailing dot and
    whitespace. Rejects wildcards, empties, and anything with spaces.
    """
    s = (raw or "").strip().lower()
    if not s or "*" in s or any(ch.isspace() for ch in s):
        raise ValueError("bad site: %r" % raw)
    if "://" in s:
        s = s.split("://", 1)[1]
    s = s.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    if "@" in s:  # userinfo tricks: user@host
        s = s.rsplit("@", 1)[1]
    if ":" in s:  # strip :port (IPv6 literals keep their brackets out)
        s = s.split(":", 1)[0]
    s = s.strip(".")
    if not s or any(ch.isspace() for ch in s):
        raise ValueError("bad site: %r" % raw)
    return s


def host_allowed(url: str, sites: list[str]) -> bool:
    """True if url's host is listed, or a subdomain of a listed site."""
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    for site in sites:
        if host == site or host.endswith("." + site):
            return True
    return False


class ResearchConfig:
    def __init__(self, data_dir: str):
        self.path = os.path.join(os.path.abspath(data_dir), FILENAME)
        self.sites: list[str] = list(DEFAULT_SITES)
        self.interests: list[str] = []
        self.nightly: dict = dict(DEFAULT_NIGHTLY)
        self.load()

    # ---- persistence -------------------------------------------------
    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError):
            return  # corrupt file: fall back to defaults, don't crash boot
        if isinstance(doc.get("sites"), list) and doc["sites"]:
            try:
                self.sites = [normalize_site(s) for s in doc["sites"]]
            except ValueError:
                pass
        if isinstance(doc.get("interests"), list):
            self.interests = [str(i).strip() for i in doc["interests"]
                              if str(i).strip()]
        if isinstance(doc.get("nightly"), dict):
            for k in ("max_topics", "max_qa", "max_minutes"):
                if isinstance(doc["nightly"].get(k), int) \
                        and doc["nightly"][k] > 0:
                    self.nightly[k] = doc["nightly"][k]

    def save(self) -> None:
        doc = {"sites": self.sites, "interests": self.interests,
               "nightly": self.nightly}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)

    # ---- sites --------------------------------------------------------
    def add_site(self, raw: str) -> str:
        site = normalize_site(raw)
        if site not in self.sites:
            self.sites.append(site)
            self.save()
        return site

    def remove_site(self, raw: str) -> str:
        site = normalize_site(raw)
        if site not in self.sites:
            raise ValueError("not in allowlist: %s" % site)
        if len(self.sites) == 1:
            raise ValueError("refusing to empty the allowlist — "
                             "at least one site must stay")
        self.sites.remove(site)
        self.save()
        return site

    def wiki_only(self) -> bool:
        """Fast path: the allowlist is exactly Wikipedia."""
        return self.sites == DEFAULT_SITES

    # ---- interests ----------------------------------------------------
    def add_interest(self, text: str) -> str:
        item = " ".join((text or "").split())
        if not item:
            raise ValueError("empty interest")
        if item not in self.interests:
            self.interests.append(item)
            self.save()
        return item

    def remove_interest(self, text: str) -> str:
        item = " ".join((text or "").split())
        if item not in self.interests:
            raise ValueError("not listed: %s" % item)
        self.interests.remove(item)
        self.save()
        return item

    # ---- nightly budget ----------------------------------------------
    def set_budget(self, **kwargs) -> dict:
        for k, v in kwargs.items():
            if k not in ("max_topics", "max_qa", "max_minutes"):
                raise ValueError("unknown budget key: %s" % k)
            if not isinstance(v, int) or v <= 0:
                raise ValueError("%s must be a positive integer" % k)
            self.nightly[k] = v
        self.save()
        return dict(self.nightly)
