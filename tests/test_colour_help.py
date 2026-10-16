"""Colour + help tests: plain text off-tty, ASCII-only banner."""
import os
import unittest

from ceti import colour
from ceti.help import HELP_TEXT


class TestColour(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k)
                       for k in ("NO_COLOR", "CETI_COLOR", "TERM")}
        for k in ("NO_COLOR", "CETI_COLOR"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_plain_when_not_tty(self):
        # pytest captures stdout: not a tty -> plain text
        self.assertFalse(colour.enabled())
        self.assertEqual(colour.paint("hi", colour.GREEN), "hi")
        self.assertEqual(colour.clear_code(), "")

    def test_no_color_overrides_force(self):
        os.environ["CETI_COLOR"] = "1"
        os.environ["NO_COLOR"] = ""
        self.assertFalse(colour.enabled())

    def test_force_off(self):
        os.environ["CETI_COLOR"] = "0"
        self.assertFalse(colour.enabled())

    def test_force_on(self):
        os.environ["CETI_COLOR"] = "1"
        self.assertTrue(colour.enabled())
        self.assertEqual(colour.paint("hi", colour.GREEN),
                         colour.GREEN + "hi" + colour.RESET)
        self.assertEqual(colour.clear_code(), "\033[2J\033[H")

    def test_banner_is_ascii(self):
        self.assertTrue(colour.BANNER.strip())
        self.assertTrue(colour.BANNER.isascii())
        self.assertIn("CETI", colour.BANNER.upper().replace(" ", ""))


class TestHelp(unittest.TestCase):
    def test_help_lists_commands(self):
        for cmd in ("/help", "/tui", "/quit", "/research", "/task",
                    "/board", "/mail", "/chat", "/benchmark"):
            self.assertIn(cmd, HELP_TEXT)


if __name__ == "__main__":
    unittest.main()
