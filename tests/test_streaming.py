"""Streaming tests: SSE parsing in complete_stream, the
Brain.stream_tokens ladder + finish_stream, and the TUI pump
(chunks, simulated block reveal, cancel, redirect)."""
import http.server
import json
import tempfile
import threading
import time
import unittest

from ceti import tui as tui_mod
from ceti.backends import BackendError, ChatBackend, OllamaBackend
from ceti.brain import Answer, Brain
from ceti.config import BrainConfig
from ceti.tui import TUI


# ---------------- backend SSE ----------------

class SSEHandler(http.server.BaseHTTPRequestHandler):
    mode = "ok"

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        if self.mode == "error":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for payload in ('{"choices":[{"delta":{"content":"Hel"}}]}',
                        '{"choices":[{"delta":{"content":"lo"}}]}'):
            self.wfile.write(f"data: {payload}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.25)  # quiet gap: the client must heartbeat
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def _serve(mode):
    SSEHandler.mode = mode
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SSEHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class TestCompleteStream(unittest.TestCase):
    def test_sse_reassembly_and_heartbeat(self):
        srv = _serve("ok")
        try:
            b = OllamaBackend(
                base_url=f"http://127.0.0.1:{srv.server_address[1]}",
                model="test")
            events = list(b.complete_stream("sys", "user"))
            self.assertIn("Hel", events)
            self.assertIn("lo", events)
            self.assertIn(None, events)  # heartbeat during the gap
            self.assertEqual("".join(e for e in events if e), "Hello")
        finally:
            srv.shutdown()

    def test_http_error(self):
        srv = _serve("error")
        try:
            b = OllamaBackend(
                base_url=f"http://127.0.0.1:{srv.server_address[1]}",
                model="test")
            with self.assertRaises(BackendError):
                list(b.complete_stream("sys", "user"))
        finally:
            srv.shutdown()


# ---------------- brain stream ladder ----------------

class FakeStreamLocal(ChatBackend):
    name = "fake-stream-local"

    def __init__(self, fragments=("Hel", "lo"), fail=None):
        self.fragments = fragments
        self.fail = fail  # None | "pre" | "mid"

    def complete(self, system, user, max_tokens=512, temperature=0.3):
        return "".join(self.fragments)

    def complete_stream(self, system, user, max_tokens=512,
                        temperature=0.3):
        if self.fail == "pre":
            raise BackendError("local down")
        yield None
        # the memory-tier handler gets its own output: a paraphrase
        # of the knowledge (no numbers, so it grounds cleanly)
        frags = (("A BBS is a bulletin board system.",)
                 if "KNOWLEDGE:" in user else self.fragments)
        for i, frag in enumerate(frags):
            if self.fail == "mid" and i == 1:
                raise BackendError("mid-stream boom")
            yield frag


class FakeStreamCloud(ChatBackend):
    name = "fake-stream-cloud"

    def complete(self, system, user, max_tokens=512, temperature=0.3):
        return "cloud answer"

    def complete_stream(self, system, user, max_tokens=512,
                        temperature=0.3):
        yield "cloud "
        yield "answer"


def make_stream_brain(local=None, clouds=None, **kw):
    tmp = tempfile.mkdtemp()
    cfg = BrainConfig(data_dir=tmp, adjudicate_with_llm=False,
                      memory_backend="sqlite", **kw)
    return Brain(config=cfg, local_backend=local or FakeStreamLocal(),
                 cloud_backends=clouds if clouds is not None
                 else [FakeStreamCloud()])


def _drain(brain, query, **kw):
    """Collect stream events; return (events, finish args)."""
    events = list(brain.stream_tokens(query, **kw))
    args = None
    for kind, payload in events:
        if kind == "stream_end":
            args = payload
    return events, args


class TestBrainStream(unittest.TestCase):
    def test_calc_block(self):
        brain = make_stream_brain()
        events, args = _drain(brain, "2+2", force_tier="calc")
        self.assertEqual(events[0], ("tag", "[calc]"))
        self.assertEqual(events[1], ("block", "4"))
        ans = brain.finish_stream(args)
        self.assertEqual((ans.tier, ans.text), ("calc", "4"))
        self.assertIn("[calc]", ans.display_tag)

    def test_local_chunks(self):
        brain = make_stream_brain()
        events, args = _drain(brain, "tell me about bbs",
                              force_tier="local")
        kinds = [k for k, _ in events]
        # the fake yields a heartbeat first: tag follows it
        self.assertIn(("tag", "[local]"), events)
        self.assertIn("chunk", kinds)
        self.assertEqual(kinds[-1], "stream_end")
        ans = brain.finish_stream(args)
        self.assertEqual((ans.tier, ans.text), ("local", "Hello"))

    def test_memory_handler_grounded(self):
        brain = make_stream_brain()
        mem = type("M", (), {})()
        mem.search = lambda q, top_k=1: [
            (1, "BBS stands for bulletin board system.", None, 1.0)]
        brain.memory = mem
        events, args = _drain(brain, "what is a bbs",
                              force_tier="memory")
        # the handler's own paraphrase streams, not the raw
        # knowledge — the grounding check passed, so it stands
        self.assertIn(("block", "A BBS is a bulletin board system."),
                      events)
        ans = brain.finish_stream(args)
        self.assertEqual(ans.tier, "memory")

    def test_pre_chunk_failure_falls_through(self):
        brain = make_stream_brain(local=FakeStreamLocal(fail="pre"))
        events, args = _drain(brain, "tell me something new")
        self.assertEqual(events[0], ("tag", "[oracle]"))
        ans = brain.finish_stream(args)
        self.assertEqual((ans.tier, ans.text),
                         ("cloud", "cloud answer"))

    def test_mid_stream_error_is_terminal(self):
        brain = make_stream_brain(local=FakeStreamLocal(fail="mid"))
        events, args = _drain(brain, "tell me about bbs",
                              force_tier="local")
        kinds = [k for k, _ in events]
        self.assertIn("error", kinds)
        self.assertNotIn("stream_end", kinds)
        self.assertIsNone(args)


# ---------------- TUI streaming ----------------

class FakeStreamBrain:
    def __init__(self):
        class Ident:
            fingerprint = "CETI-TEST-0000-0000"
            node_id = "jn1test"
        self.identity = Ident()
        self.cfg = type("Cfg", (),
                        {"data_dir": tempfile.mkdtemp()})()
        self.finished = []

    def stream_tokens(self, q):
        yield ("tag", "[local]")
        if q == "chunky":
            yield ("chunk", "Hel")
            yield ("chunk", "lo")
        else:
            yield ("block", "word " * 100)
        yield ("stream_end", {"query": q, "stripped_q": q,
                              "text": "Hello" if q == "chunky"
                              else "word " * 100,
                              "tier": "local", "intent": "",
                              "learn": False, "t0": time.monotonic(),
                              "trace": {}, "pii_found": []})

    def finish_stream(self, args):
        ans = Answer(text=args["text"], tier=args["tier"],
                     latency_ms=5, intent="", pii_found=[])
        self.finished.append(ans)
        return ans


def _pump_until_done(t, limit=50):
    # the real run loop spaces pumps ~100ms apart (getch timeout);
    # simulate that frame time so the reveal advances
    for _ in range(limit):
        if not t._streaming():
            return
        t._stream_last_pump = time.monotonic() - 0.1
        t._pump_stream()


class TestTUIStreaming(unittest.TestCase):
    def test_chunk_flow(self):
        t = TUI(FakeStreamBrain())
        t._submit("chunky")
        _pump_until_done(t)
        joined = "\n".join(t.transcript.lines)
        self.assertIn("Hello", joined)
        self.assertIn("[local]", joined)
        self.assertFalse(t._streaming())

    def test_block_reveal_pacing(self):
        t = TUI(FakeStreamBrain())
        t._submit("blocky")
        t._pump_stream()  # tag
        t._pump_stream()  # block pulled, nothing revealed yet
        self.assertEqual(t._stream_buf, "")
        self.assertTrue(t._stream_reveal)
        t._stream_last_pump = time.monotonic() - 0.05  # 50ms pass
        t._pump_stream()
        self.assertTrue(t._stream_buf)  # a slice revealed
        self.assertTrue(t._stream_reveal)  # not all of it yet
        _pump_until_done(t)
        self.assertIn("word word", "\n".join(t.transcript.lines))

    def test_cancel_keeps_partial(self):
        brain = FakeStreamBrain()
        t = TUI(brain)
        t._submit("chunky")
        t._pump_stream()  # tag
        t._pump_stream()  # first chunk
        self.assertIn("Hel", t._stream_buf)
        t.handle_key(3)  # Ctrl-C
        joined = "\n".join(t.transcript.lines)
        self.assertIn("Hel", joined)
        self.assertIn("(cancelled)", joined)
        self.assertFalse(t._streaming())
        self.assertEqual(brain.finished, [])  # cancelled: not learned

    def test_redirect_keeps_partial(self):
        t = TUI(FakeStreamBrain())
        t._submit("chunky")
        t._pump_stream()
        t._pump_stream()
        t._submit("blocky")  # Enter while streaming: redirect
        joined = "\n".join(t.transcript.lines)
        self.assertIn("Hel", joined)
        self.assertIn("you> blocky", joined)


if __name__ == "__main__":
    unittest.main()
