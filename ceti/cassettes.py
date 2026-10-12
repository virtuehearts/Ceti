"""Cassettes — swappable AIML topic datasets.

Each cassette is one AIML file in data/cassettes/, e.g.
`bbs-history.aiml`, `dial-up-modems.aiml`. Like tapes: the brain loads
a set into the rack, the idle researcher records new ones, and a
cassette can be ejected (unloaded) without touching anything else.

A cassette file is just AIML 1.0 with <category> entries — the same
shape as learned.aiml, so AimlBrain(learned_path=...) doubles as the
recorder.
"""
from __future__ import annotations

import os
import re

_SLUG_RX = re.compile(r"[^a-z0-9]+")


def slugify(topic: str) -> str:
    return _SLUG_RX.sub("-", topic.lower()).strip("-")[:60] or "topic"


class CassetteRack:
    def __init__(self, cassette_dir: str):
        self.dir = os.path.abspath(cassette_dir)
        os.makedirs(self.dir, exist_ok=True)

    def list(self) -> list[str]:
        """Names of all cassettes on the shelf (without .aiml)."""
        names = []
        for fn in sorted(os.listdir(self.dir)):
            if fn.endswith(".aiml"):
                names.append(fn[:-5])
        return names

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name + ".aiml")

    def create(self, topic: str) -> str:
        """Reserve the cassette file for a topic; return its path.
        Creates an empty valid AIML file if missing."""
        name = slugify(topic)
        p = self.path(name)
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8") as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                        '<!-- cassette: %s — recorded by the idle researcher'
                        ' -->\n<aiml version="1.0">\n</aiml>\n' % topic)
        return p

    def load_paths(self, names: list[str] | str = "all") -> list[str]:
        """Resolve cassette names to existing file paths."""
        if names == "all":
            names = self.list()
        paths = []
        for n in names:
            p = self.path(n)
            if os.path.exists(p):
                paths.append(p)
        return paths

    def eject(self, name: str) -> bool:
        """Remove a cassette file from the shelf."""
        p = self.path(name)
        if os.path.exists(p):
            os.remove(p)
            return True
        return False

    def stats(self) -> dict:
        out = {}
        for n in self.list():
            try:
                size = os.path.getsize(self.path(n))
            except OSError:
                size = 0
            out[n] = {"bytes": size}
        return out
