"""Skill bundles: markdown + code, hot-loadable like cassettes.

A skill is a directory containing a SKILL.md with frontmatter:

    ---
    name: web-fetch
    description: Fetch web pages to markdown.
    ---
    # Web fetch
    ...the guide...

Only a flat `key: value` frontmatter subset is supported — CETI is
stdlib-only, so there is no PyYAML. `name` and `description` are
required; anything else is kept as metadata.

The agent discovers skills through the `list_skills` / `read_skill`
tools, and the agent loop lists them in its system prompt. Extra
files in the skill directory (scripts, examples) are bundled along
and listed by read_skill.
"""

from __future__ import annotations

import os
import re

from . import tools as _tools

SKILL_FILENAME = "SKILL.md"


class SkillError(Exception):
    """A skill bundle is malformed."""


class Skill:
    """One loaded skill bundle."""

    def __init__(self, name: str, description: str, doc: str,
                 path: str, meta: dict | None = None):
        self.name = name
        self.description = description
        self.doc = doc            # markdown body, frontmatter stripped
        self.path = path          # skill directory
        self.meta = meta or {}

    def files(self) -> list[str]:
        """Other bundled files besides SKILL.md, relative to the dir."""
        out = []
        for root, _dirs, names in os.walk(self.path):
            for n in sorted(names):
                if n == SKILL_FILENAME and root == self.path:
                    continue
                full = os.path.join(root, n)
                out.append(os.path.relpath(full, self.path))
        return sorted(out)

    def full_text(self) -> str:
        parts = ["# %s — %s" % (self.name, self.description),
                 self.doc.strip()]
        files = self.files()
        if files:
            parts.append("Bundled files:\n"
                         + "\n".join("  " + f for f in files))
        return "\n\n".join(parts)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split SKILL.md into (metadata, body). Raises SkillError."""
    m = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n(.*)$", text, re.DOTALL)
    if not m:
        raise SkillError("missing frontmatter block (--- ... ---)")
    meta: dict[str, str] = {}
    for i, line in enumerate(m.group(1).split("\n"), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise SkillError("bad frontmatter line %d: %r" % (i, line))
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if (len(value) >= 2 and value[0] == value[-1]
                and value[0] in "\"'"):
            value = value[1:-1]
        meta[key] = value
    return meta, m.group(2)


def load_skill_file(skill_md: str) -> Skill:
    """Parse one SKILL.md file into a Skill. Raises SkillError."""
    try:
        with open(skill_md, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise SkillError("cannot read %s: %s" % (skill_md, e))
    meta, body = _parse_frontmatter(text)
    name = meta.get("name", "").strip()
    description = meta.get("description", "").strip()
    if not name:
        raise SkillError("%s: frontmatter needs a `name`" % skill_md)
    if not description:
        raise SkillError("%s: frontmatter needs a `description`"
                         % skill_md)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise SkillError("%s: bad skill name %r (lowercase, digits, "
                         "-/_)" % (skill_md, name))
    meta = {k: v for k, v in meta.items()
            if k not in ("name", "description")}
    return Skill(name=name, description=description, doc=body,
                 path=os.path.dirname(os.path.abspath(skill_md)),
                 meta=meta)


class SkillLoader:
    """Scans a skills directory; hot-reloadable."""

    def __init__(self, skills_dir: str | None = None):
        self.dir = os.path.abspath(
            skills_dir or os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "..", "skills"))
        self.skills: dict[str, Skill] = {}
        self.errors: list[str] = []

    def load_all(self) -> dict[str, Skill]:
        """(Re)scan the directory. Bad bundles are skipped loudly —
        recorded in self.errors, never silent."""
        found: dict[str, Skill] = {}
        errors: list[str] = []
        if os.path.isdir(self.dir):
            for entry in sorted(os.listdir(self.dir)):
                sub = os.path.join(self.dir, entry)
                if not os.path.isdir(sub):
                    continue
                md = os.path.join(sub, SKILL_FILENAME)
                if not os.path.isfile(md):
                    continue
                try:
                    skill = load_skill_file(md)
                except SkillError as e:
                    errors.append(str(e))
                    continue
                if skill.name in found:
                    errors.append("duplicate skill name %r (%s)"
                                  % (skill.name, md))
                    continue
                found[skill.name] = skill
        self.skills = found
        self.errors = errors
        return self.skills

    def reload(self) -> dict[str, Skill]:
        return self.load_all()

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def prompt_section(self) -> str:
        """One-liner list for the agent's system prompt."""
        if not self.skills:
            return ""
        lines = ["Skill bundles (call read_skill with the name to load "
                 "the full guide):"]
        for name in sorted(self.skills):
            s = self.skills[name]
            lines.append("- %s: %s" % (name, s.description))
        return "\n".join(lines)

    # ---- tools ---------------------------------------------------------
    def _tool_list_skills(self) -> str:
        if not self.skills:
            return "(no skills loaded from %s)" % self.dir
        return "\n".join(
            "  %s — %s" % (n, self.skills[n].description)
            for n in sorted(self.skills))

    def _tool_read_skill(self, name: str) -> str:
        skill = self.get(name)
        if skill is None:
            raise _tools.ToolError(
                "no such skill: %r (see list_skills)" % name)
        return skill.full_text()

    def register_tools(self) -> None:
        _tools.register_tool(_tools.Tool(
            name="list_skills",
            description="List the loaded skill bundles: one-line "
                        "guides the agent can read with read_skill.",
            parameters=_tools._schema({}, []),
            side_effects="read", sandboxed=False,
            func=self._tool_list_skills))
        _tools.register_tool(_tools.Tool(
            name="read_skill",
            description="Load a skill bundle's full guide by name.",
            parameters=_tools._schema(
                {"name": {"type": "string",
                          "description": "skill name from list_skills"}},
                ["name"]),
            side_effects="read", sandboxed=False,
            func=self._tool_read_skill))


_LOADER: SkillLoader | None = None


def get_loader() -> SkillLoader:
    """Process-wide loader over the repo skills/ directory."""
    global _LOADER
    if _LOADER is None:
        _LOADER = SkillLoader()
        _LOADER.load_all()
    return _LOADER


def ensure_tools_registered() -> SkillLoader:
    """Load skills and register list_skills/read_skill (idempotent)."""
    loader = get_loader()
    names = set(_tools.all_tools())
    if "list_skills" not in names or "read_skill" not in names:
        loader.register_tools()
    return loader
