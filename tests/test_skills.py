import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ceti import skills, tools
from ceti.skills import SkillError, SkillLoader, load_skill_file


def _write_skill(base, dirname, body):
    d = os.path.join(base, dirname)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "SKILL.md"), "w") as f:
        f.write(body)
    return d


GOOD = """---
name: demo-skill
description: A demo skill for tests.
version: 3
---
# Demo

Do the thing.
"""


class TestSkillParsing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ceti-skills-")

    def _md(self, body):
        d = _write_skill(self.tmp, "demo", body)
        return os.path.join(d, "SKILL.md")

    def test_valid(self):
        s = load_skill_file(self._md(GOOD))
        self.assertEqual(s.name, "demo-skill")
        self.assertEqual(s.description, "A demo skill for tests.")
        self.assertIn("Do the thing.", s.doc)
        self.assertEqual(s.meta.get("version"), "3")

    def test_missing_frontmatter(self):
        with self.assertRaises(SkillError):
            load_skill_file(self._md("# no frontmatter\n"))

    def test_missing_name(self):
        with self.assertRaises(SkillError):
            load_skill_file(self._md(
                "---\ndescription: x\n---\nbody\n"))

    def test_missing_description(self):
        with self.assertRaises(SkillError):
            load_skill_file(self._md("---\nname: x\n---\nbody\n"))

    def test_bad_name(self):
        with self.assertRaises(SkillError):
            load_skill_file(self._md(
                "---\nname: Bad Name!\ndescription: x\n---\nbody\n"))

    def test_quoted_values(self):
        s = load_skill_file(self._md(
            '---\nname: q\ndescription: "quoted desc"\n---\nbody\n'))
        self.assertEqual(s.description, "quoted desc")


class TestSkillLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ceti-skills-")
        _write_skill(self.tmp, "aaa", GOOD)
        _write_skill(self.tmp, "bbb", GOOD.replace(
            "demo-skill", "other-skill").replace(
            "A demo skill for tests.", "Another one."))
        os.makedirs(os.path.join(self.tmp, "empty-dir"))
        with open(os.path.join(self.tmp, "stray.txt"), "w") as f:
            f.write("not a skill")
        self.loader = SkillLoader(self.tmp)

    def test_load_all(self):
        found = self.loader.load_all()
        self.assertEqual(set(found), {"demo-skill", "other-skill"})
        self.assertEqual(self.loader.errors, [])

    def test_missing_dir(self):
        loader = SkillLoader(os.path.join(self.tmp, "nope"))
        self.assertEqual(loader.load_all(), {})

    def test_duplicate_name(self):
        _write_skill(self.tmp, "ccc", GOOD)  # same frontmatter name
        found = self.loader.load_all()
        self.assertEqual(len(found), 2)  # first wins
        self.assertTrue(any("duplicate" in e
                            for e in self.loader.errors))

    def test_bad_bundle_skipped_loudly(self):
        _write_skill(self.tmp, "bad", "# no frontmatter\n")
        found = self.loader.load_all()
        self.assertEqual(len(found), 2)
        self.assertEqual(len(self.loader.errors), 1)

    def test_reload_picks_up_new(self):
        self.loader.load_all()
        _write_skill(self.tmp, "ddd", GOOD.replace(
            "demo-skill", "third-skill").replace(
            "A demo skill for tests.", "Third."))
        found = self.loader.reload()
        self.assertIn("third-skill", found)

    def test_prompt_section(self):
        self.loader.load_all()
        section = self.loader.prompt_section()
        self.assertIn("demo-skill", section)
        self.assertIn("read_skill", section)

    def test_prompt_section_empty(self):
        loader = SkillLoader(os.path.join(self.tmp, "nope"))
        loader.load_all()
        self.assertEqual(loader.prompt_section(), "")

    def test_skill_files(self):
        d = os.path.join(self.tmp, "aaa")
        with open(os.path.join(d, "helper.py"), "w") as f:
            f.write("# helper")
        self.loader.load_all()
        s = self.loader.get("demo-skill")
        self.assertIn("helper.py", s.files())
        self.assertIn("helper.py", s.full_text())


class TestSkillTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ceti-skills-")
        _write_skill(self.tmp, "aaa", GOOD)
        self.loader = SkillLoader(self.tmp)
        self.loader.load_all()
        self.loader.register_tools()

    def tearDown(self):
        # restore the repo loader's tools in the global registry
        skills.ensure_tools_registered()

    def test_list_skills(self):
        r = tools.execute("list_skills")
        self.assertTrue(r.success)
        self.assertIn("demo-skill", r.content)

    def test_read_skill(self):
        r = tools.execute("read_skill", name="demo-skill")
        self.assertTrue(r.success)
        self.assertIn("Do the thing.", r.content)

    def test_read_skill_unknown(self):
        r = tools.execute("read_skill", name="nope")
        self.assertFalse(r.success)
        self.assertIn("no such skill", r.error)

    def test_schemas(self):
        names = {s["function"]["name"]
                 for s in tools.openai_schemas()}
        self.assertTrue({"list_skills", "read_skill"} <= names)


class TestAgentSessionSkills(unittest.TestCase):
    def test_system_prompt_lists_skills(self):
        from ceti.agent_loop import AgentSession
        from ceti.config import BrainConfig

        class FakeBackend:
            name = "fake"
            api_key = "k"

            def complete_with_tools(self, messages, tools_, **kw):
                return {"content": "done", "tool_calls": []}

        tmp = tempfile.mkdtemp(prefix="ceti-skills-")
        _write_skill(tmp, "aaa", GOOD)
        loader = SkillLoader(tmp)
        loader.load_all()
        cfg = BrainConfig(
            data_dir=os.path.join(tempfile.mkdtemp(), "data"))
        sess = AgentSession(cfg, FakeBackend(), skill_loader=loader)
        try:
            system = sess.messages[0]["content"]
        finally:
            sess.close()
        self.assertIn("demo-skill", system)
        self.assertIn("read_skill", system)


if __name__ == "__main__":
    unittest.main()
