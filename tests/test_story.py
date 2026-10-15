import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ceti.config import BrainConfig
from ceti.story import StoryError, StoryWriter, slugify, word_count


class FakeBackend:
    """Scripted oracle: pops one canned completion per call."""
    name = "fake-oracle"

    def __init__(self, script, api_key="k"):
        self.script = list(script)
        self.api_key = api_key
        self.calls = 0

    def complete(self, system, user, max_tokens=512, temperature=0.8):
        self.calls += 1
        if not self.script:
            raise AssertionError("backend called more than scripted")
        return self.script.pop(0)


BIBLE = ("# The Clockwork Orchard\n\n1. Mara finds a brass key.\n"
         "2. The orchard ticks.\n3. The keeper wakes.\n")


class TestStory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ceti-story-")
        self.cfg = BrainConfig(data_dir=self.tmp)

    def _writer(self, script):
        return StoryWriter(self.cfg, FakeBackend(script), library=None)

    def test_slugify(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")
        self.assertTrue(slugify(""))

    def test_new_story(self):
        w = self._writer([BIBLE])
        sid = w.new_story("a clockwork orchard mystery")
        self.assertTrue(sid.startswith("a-clockwork-orchard-mystery-"))
        d = os.path.join(self.tmp, "stories", sid)
        self.assertTrue(os.path.isfile(os.path.join(d, "bible.md")))
        self.assertTrue(os.path.isfile(os.path.join(d, "summary.md")))
        self.assertIn(sid, w.list_stories())

    def test_new_story_empty_premise(self):
        w = self._writer([])
        with self.assertRaises(StoryError):
            w.new_story("   ")

    def test_next_block(self):
        chapter = "Mara turned the brass key. " * 120  # ~600 words
        summary = "Chapter 1: Mara found the key."
        w = self._writer([BIBLE, chapter, summary])
        sid = w.new_story("premise")
        r = w.next_block(sid, pages=3)
        self.assertEqual(r["block"], 1)
        self.assertGreater(r["words"], 400)
        self.assertTrue(os.path.isfile(r["path"]))
        # summary rolled forward
        with open(os.path.join(self.tmp, "stories", sid,
                               "summary.md")) as f:
            self.assertIn("Chapter 1", f.read())

    def test_short_block_expands(self):
        # chapter too short -> one expansion call
        w = self._writer([BIBLE, "Too short.", "More prose. " * 200,
                          "Chapter 1: things."])
        sid = w.new_story("premise")
        r = w.next_block(sid, pages=3)  # target 600 words
        self.assertGreater(r["words"], 400)
        # bible + chapter + expansion + summary = 4 calls
        self.assertEqual(w.backend.calls, 4)

    def test_beats_drive_chapters(self):
        w = self._writer([BIBLE])
        sid = w.new_story("premise")
        self.assertIn("brass key", w._beats(sid)[0])

    def test_status_and_compile(self):
        ch1 = "First. " * 150
        ch2 = "Second. " * 150
        w = self._writer([BIBLE, ch1, "s1", ch2, "s2"])
        sid = w.new_story("premise")
        w.next_block(sid, pages=1)
        w.next_block(sid, pages=1)
        st = w.status(sid)
        self.assertEqual(st["blocks"], 2)
        self.assertEqual(st["words"], 300)
        r = w.compile(sid)
        self.assertEqual(r["chapters"], 2)
        self.assertTrue(os.path.isfile(r["md"]))
        self.assertTrue(os.path.isfile(r["txt"]))
        with open(r["md"]) as f:
            content = f.read()
        self.assertIn("Chapter 1", content)
        self.assertIn("Chapter 2", content)

    def test_compile_no_blocks(self):
        w = self._writer([BIBLE])
        sid = w.new_story("premise")
        with self.assertRaises(StoryError):
            w.compile(sid)

    def test_no_key_refuses(self):
        with self.assertRaises(StoryError):
            StoryWriter(self.cfg, FakeBackend([], api_key=None))

    def test_word_count(self):
        self.assertEqual(word_count("a b c"), 3)


if __name__ == "__main__":
    unittest.main()
