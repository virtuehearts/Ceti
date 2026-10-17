import tempfile
import threading
import unittest

from ceti import tui as tui_mod
from ceti import tui_sessions as ts
from ceti.tui import (History, LineEditor, Transcript, TUI,
                      complete_command)


class FakeAnswer:
    def __init__(self, text, tag="[local] (499 t/s)"):
        self.text = text
        self.display_tag = tag


class FakeBrain:
    def __init__(self, data_dir=None):
        class Ident:
            fingerprint = "CETI-TEST-0000-0000"
            node_id = "jn1test"
        self.identity = Ident()
        self.cfg = type("Cfg", (),
                        {"data_dir": data_dir or tempfile.mkdtemp()})()


class FakeScreen:
    """A curses window stand-in: keys from a queue, -1 when empty."""

    def __init__(self, keys):
        self.keys = list(keys)
        self.calls = 0

    def getch(self):
        self.calls += 1
        if self.keys:
            return self.keys.pop(0)
        return -1

    def getmaxyx(self):
        return (24, 80)

    def addstr(self, *args):
        pass

    def erase(self):
        pass

    def refresh(self):
        pass

    def curs_set(self, n):
        pass


def type_keys(text):
    return [ord(c) for c in text]


class TestComplete(unittest.TestCase):
    def test_prefix_filters(self):
        self.assertIn("/gopher", complete_command("/g"))
        self.assertNotIn("/board", complete_command("/g"))
        self.assertEqual(complete_command("/quit"), ["/quit"])
        self.assertEqual(complete_command("/zzz"), [])


class TestHistory(unittest.TestCase):
    def test_prev_next_cycle(self):
        h = History()
        h.add("first")
        h.add("second")
        self.assertEqual(h.prev(""), "second")
        self.assertEqual(h.prev("second"), "first")
        self.assertEqual(h.next("first"), "second")
        self.assertEqual(h.next("second"), "")

    def test_empty_history(self):
        h = History()
        self.assertEqual(h.prev("typed"), "typed")
        self.assertEqual(h.next("typed"), "typed")


class TestLineEditor(unittest.TestCase):
    def test_insert_and_backspace(self):
        e = LineEditor()
        for c in "abc":
            e.insert(c)
        self.assertEqual(e.text, "abc")
        e.backspace()
        self.assertEqual(e.text, "ab")

    def test_newline_and_join(self):
        e = LineEditor("hello world")
        for _ in range(5):
            e.move_left()
        e.newline()
        self.assertEqual(e.lines, ["hello ", "world"])
        e.backspace()
        self.assertEqual(e.text, "hello world")

    def test_cursor_moves(self):
        e = LineEditor("ab\ncd")
        e.move_up()
        self.assertEqual((e.row, e.col), (0, 2))
        e.move_down()
        self.assertEqual((e.row, e.col), (1, 2))
        e.move_home()
        self.assertEqual(e.col, 0)
        e.move_end()
        self.assertEqual(e.col, 2)

    def test_delete_joins_lines(self):
        e = LineEditor("ab\ncd")
        e.move_up()
        e.move_end()
        e.delete()
        self.assertEqual(e.text, "abcd")


class TestTranscript(unittest.TestCase):
    def test_render_window(self):
        t = Transcript()
        for i in range(10):
            t.add(f"line {i}")
        self.assertEqual(t.render(3, 80),
                         ["line 7", "line 8", "line 9"])
        t.scroll_up(2)
        self.assertEqual(t.render(3, 80),
                         ["line 5", "line 6", "line 7"])
        t.scroll_down(2)
        self.assertEqual(t.render(3, 80),
                         ["line 7", "line 8", "line 9"])


class TestTUILoop(unittest.TestCase):
    def _drive(self, tui, keys, timeout=5):
        screen = FakeScreen(keys)
        th = threading.Thread(target=tui.run, args=(screen,),
                              daemon=True)
        th.start()
        th.join(timeout)
        self.assertFalse(th.is_alive(), "TUI loop did not exit")
        return screen

    def test_send_and_quit(self):
        brain = FakeBrain()
        answers = []

        def ask(q):
            answers.append(q)
            return FakeAnswer("hello back")

        t = TUI(brain, ask_fn=ask)
        # BBS: launch lands on the Main Menu — pick door 1 (AI)
        # first, then ask. -1 keys are poll ticks: give the answer
        # thread time to finish and be collected before /quit exits.
        keys = (type_keys("1") + [10]
                + type_keys("hello") + [10] + [-1] * 5
                + type_keys("/quit") + [10])
        self._drive(t, keys)
        self.assertEqual(answers, ["hello"])
        joined = "\n".join(t.transcript.lines)
        self.assertIn("you> hello", joined)
        self.assertIn("hello back", joined)

    def test_slash_completion(self):
        t = TUI(FakeBrain())
        t.handle_key(ord("/"))
        t.handle_key(ord("g"))
        self.assertIn("/gopher", t.completions)
        t.handle_key(9)  # Tab
        self.assertTrue(t.editor.text.startswith("/gopher"))

    def test_history_keys(self):
        t = TUI(FakeBrain())
        t.history.add("earlier question")
        t.handle_key(259)  # Up
        self.assertEqual(t.editor.text, "earlier question")
        t.handle_key(258)  # Down
        self.assertEqual(t.editor.text, "")

    def test_ctrl_c_cancels(self):
        t = TUI(FakeBrain())
        t.handle_key(3)
        self.assertIn("(cancelled)",
                      "\n".join(t.transcript.lines))

    def test_alt_enter_newline(self):
        t = TUI(FakeBrain())
        for c in "ab":
            t.handle_key(ord(c))
        t.handle_key(27)  # Alt
        t.handle_key(10)  # Enter -> newline, not send
        self.assertEqual(t.editor.lines, ["ab", ""])
        self.assertTrue(t.running)


class TestWrapAndSafePut(unittest.TestCase):
    def test_wrap_long_lines_never_exceed_width_minus_one(self):
        # Windows overflow: no rendered line may touch the last column.
        t = Transcript()
        t.add("short")
        t.add("word " * 30)  # 150 chars, must wrap at width 20
        t.add("a" * 100)  # one long token, must break
        out = t.render(50, 20)
        self.assertTrue(all(len(ln) <= 19 for ln in out),
                        out)
        self.assertTrue(any("short" in ln for ln in out))

    def test_wrap_short_lines_unchanged(self):
        t = Transcript()
        t.add("hello")
        self.assertEqual(t.render(5, 80), ["hello"])

    def test_wrap_tabs_and_cr(self):
        t = Transcript()
        t.add("a\tb\r c")
        out = t.render(5, 80)
        self.assertEqual(out, ["a    b c"])

    def test_render_never_writes_last_column(self):
        # FakeScreen records every addstr; safe put() must trim to
        # width-x-1 so no call touches the last column.
        class Screen:
            def __init__(self):
                self.calls = []

            def getmaxyx(self):
                return (24, 80)

            def addstr(self, y, x, s, *a):
                self.calls.append((y, x, s))

            def erase(self):
                pass

            def refresh(self):
                pass

            def curs_set(self, n):
                pass

        t = TUI(FakeBrain(), ask_fn=lambda q: FakeAnswer("x"))
        t.transcript.add("word " * 40)
        screen = Screen()
        t.render(screen)
        for y, x, s in screen.calls:
            self.assertLess(x + len(s), 80,
                            f"addstr at ({y},{x}) len {len(s)} touches last col")
            self.assertLessEqual(y, 23)


class TestBBSMenu(unittest.TestCase):
    def test_launch_lands_on_main_menu(self):
        t = TUI(FakeBrain())
        self.assertEqual(t.menu, "main")

    def test_menu_pick_ai_door(self):
        t = TUI(FakeBrain())
        t.handle_key(ord("1"))
        t.handle_key(10)
        self.assertIsNone(t.menu)

    def test_menu_pick_submenu_and_back(self):
        t = TUI(FakeBrain())
        t.handle_key(ord("6"))  # System
        t.handle_key(10)
        self.assertEqual(t.menu, "system")
        t.handle_key(ord("0"))  # Back
        t.handle_key(10)
        self.assertEqual(t.menu, "main")

    def test_menu_goodbye_quits(self):
        t = TUI(FakeBrain())
        t.handle_key(ord("0"))
        t.handle_key(10)
        self.assertFalse(t.running)

    def test_menu_unknown_stays(self):
        t = TUI(FakeBrain())
        for c in "hello":
            t.handle_key(ord(c))
        t.handle_key(10)
        self.assertEqual(t.menu, "main")
        self.assertIn("unknown choice",
                      "\n".join(t.transcript.lines))

    def test_menu_slash_ai(self):
        t = TUI(FakeBrain())
        for c in "/ai":
            t.handle_key(ord(c))
        t.handle_key(10)
        self.assertIsNone(t.menu)

    def test_menu_slash_menu_returns(self):
        t = TUI(FakeBrain())
        t.menu = None
        t._run_command("/menu")
        self.assertEqual(t.menu, "main")

    def test_menu_shows_in_render(self):
        class Screen:
            def __init__(self):
                self.rows = {}

            def getmaxyx(self):
                return (24, 80)

            def addstr(self, y, x, s, *a):
                self.rows[y] = self.rows.get(y, "") + s

            def erase(self):
                pass

            def refresh(self):
                pass

            def curs_set(self, n):
                pass

        t = TUI(FakeBrain())
        screen = Screen()
        t.render(screen)
        joined = "\n".join(screen.rows.values())
        self.assertIn("MAIN MENU", joined)
        self.assertIn("[1] AI", joined)

    def test_menu_resize_key_is_noop(self):
        t = TUI(FakeBrain())
        t.handle_key(410)  # KEY_RESIZE
        self.assertTrue(t.running)
        self.assertEqual(t.menu, "main")


class TestTUISessions(unittest.TestCase):
    def _ask(self, q):
        return FakeAnswer("answer to " + q)

    def test_new_chat_saves_old(self):
        brain = FakeBrain()
        t = TUI(brain, ask_fn=self._ask)
        old_id = t.session.id
        t._submit("first question")
        t._submit("/new")
        self.assertNotEqual(t.session.id, old_id)
        saved = ts.load_session(brain.cfg.data_dir, old_id)
        self.assertIsNotNone(saved)
        self.assertIn("you> first question", saved.lines)
        self.assertIn("(new chat", "\n".join(t.transcript.lines))

    def test_sessions_and_resume(self):
        brain = FakeBrain()
        t1 = TUI(brain, ask_fn=self._ask)
        t1._submit("first question")
        sid = t1.session.id
        t2 = TUI(brain, ask_fn=self._ask)
        t2._submit("/sessions")
        self.assertIn("first question",
                      "\n".join(t2.transcript.lines))
        t2._submit(f"/resume {sid}")
        self.assertIn("you> first question",
                      "\n".join(t2.transcript.lines))
        self.assertIn("(resumed", "\n".join(t2.transcript.lines))

    def test_empty_chat_leaves_no_file(self):
        brain = FakeBrain()
        TUI(brain, ask_fn=self._ask)
        self.assertEqual(ts.list_sessions(brain.cfg.data_dir), [])

    def test_delete(self):
        brain = FakeBrain()
        t = TUI(brain, ask_fn=self._ask)
        t._submit("to delete")
        sid = t.session.id
        t._submit(f"/delete {sid}")
        self.assertIsNone(ts.load_session(brain.cfg.data_dir, sid))
        self.assertIn("(deleted", "\n".join(t.transcript.lines))

    def test_help(self):
        t = TUI(FakeBrain())
        t._submit("/help")
        joined = "\n".join(t.transcript.lines)
        self.assertIn("/new", joined)
        self.assertIn("Enter send", joined)

    def test_header_shows_session_title(self):
        class Screen:
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

        t = TUI(FakeBrain(), ask_fn=self._ask)
        t._submit("hello world")
        screen = Screen()
        t.render(screen)
        self.assertIn("hello world", screen.rows[0])


if __name__ == "__main__":
    unittest.main()
