"""TUI colour tests: role classification + attrs reach the screen.

The palette is render-time only — the transcript stays plain text,
so saved sessions are unaffected. Tests drive render() with a
recording fake screen; init_palette() is never called in tests,
so attrs are injected directly into the module palette.
"""
import os
import unittest

from ceti import tui as tui_mod
from ceti.tui import TUI, attr_for
from tests.test_tui import FakeBrain


class RecScreen:
    def __init__(self):
        self.calls = []

    def getch(self):
        return -1

    def getmaxyx(self):
        return (24, 80)

    def addstr(self, *args):
        self.calls.append(args)

    def erase(self):
        pass

    def refresh(self):
        pass

    def curs_set(self, n):
        pass


class TestAttrFor(unittest.TestCase):
    def setUp(self):
        self._saved_attrs = dict(tui_mod._ATTRS)
        self._saved_init = tui_mod._PALETTE_INIT
        tui_mod._ATTRS.update({"header": 1, "user": 2, "ceti": 3,
                               "tag": 4, "system": 5, "alert": 6,
                               "popup_sel": 7})

    def tearDown(self):
        tui_mod._ATTRS.clear()
        tui_mod._ATTRS.update(self._saved_attrs)
        tui_mod._PALETTE_INIT = self._saved_init

    def test_roles(self):
        self.assertEqual(attr_for("you> hello"), 2)
        self.assertEqual(attr_for("[local] (499 t/s)"), 4)
        self.assertEqual(attr_for("error: boom"), 6)
        self.assertEqual(attr_for("(cancelled)"), 6)
        self.assertEqual(attr_for("  1. [FP] subject"), 5)
        self.assertEqual(attr_for("(new chat — abc)"), 5)
        self.assertEqual(attr_for("plain answer text"), 3)

    def test_uninitialised_palette_is_zero(self):
        tui_mod._ATTRS.clear()
        self.assertEqual(attr_for("you> hello"), 0)
        self.assertEqual(attr_for("plain"), 0)


class TestInitPalette(unittest.TestCase):
    def setUp(self):
        self._saved_attrs = dict(tui_mod._ATTRS)
        self._saved_init = tui_mod._PALETTE_INIT
        self._saved_no_color = os.environ.get("NO_COLOR")

    def tearDown(self):
        tui_mod._ATTRS.clear()
        tui_mod._ATTRS.update(self._saved_attrs)
        tui_mod._PALETTE_INIT = self._saved_init
        if self._saved_no_color is None:
            os.environ.pop("NO_COLOR", None)
        else:
            os.environ["NO_COLOR"] = self._saved_no_color

    def test_no_color_leaves_palette_empty(self):
        tui_mod._ATTRS.clear()
        tui_mod._PALETTE_INIT = False
        os.environ["NO_COLOR"] = ""
        tui_mod.init_palette()
        self.assertEqual(tui_mod._ATTRS, {})

    def test_safe_outside_curses(self):
        tui_mod._ATTRS.clear()
        tui_mod._PALETTE_INIT = False
        os.environ.pop("NO_COLOR", None)
        tui_mod.init_palette()  # must not raise without initscr()
        self.assertTrue(tui_mod._PALETTE_INIT)


class TestRenderAttrs(unittest.TestCase):
    def setUp(self):
        self._saved_attrs = dict(tui_mod._ATTRS)
        tui_mod._ATTRS.update({"header": 11, "user": 22, "ceti": 33,
                               "tag": 44, "system": 55, "alert": 66,
                               "popup_sel": 77})

    def tearDown(self):
        tui_mod._ATTRS.clear()
        tui_mod._ATTRS.update(self._saved_attrs)

    def test_render_passes_attrs(self):
        app = TUI(FakeBrain())
        app.transcript.lines = ["you> hello",
                                "[local] (499 t/s)",
                                "the answer",
                                "error: boom"]
        scr = RecScreen()
        app.render(scr)
        by_text = {}
        for call in scr.calls:
            if len(call) >= 3 and isinstance(call[2], str):
                by_text[call[2].strip()] = call[3] if len(call) > 3 \
                    else 0
        self.assertEqual(by_text.get("you> hello"), 22)
        self.assertEqual(by_text.get("[local] (499 t/s)"), 44)
        self.assertEqual(by_text.get("the answer"), 33)
        self.assertEqual(by_text.get("error: boom"), 66)
        # header carries the header attr
        header_calls = [c for c in scr.calls
                        if len(c) >= 3 and "CETI //" in str(c[2])]
        self.assertTrue(header_calls)
        self.assertEqual(header_calls[0][3], 11)

    def test_strict_3arg_screen_degrades_to_plain(self):
        """A screen whose addstr takes exactly (y, x, s) must still
        render with a forced palette — colour is best-effort."""
        class StrictScreen:
            def __init__(self):
                self.rows = {}

            def getmaxyx(self):
                return (24, 80)

            def addstr(self, y, x, s):
                self.rows[y] = s

            def erase(self):
                pass

            def refresh(self):
                pass

            def curs_set(self, n):
                pass

        app = TUI(FakeBrain())
        app.transcript.lines = ["you> hello", "the answer"]
        scr = StrictScreen()
        app.render(scr)
        self.assertIn("you> hello", scr.rows[1])


if __name__ == "__main__":
    unittest.main()
