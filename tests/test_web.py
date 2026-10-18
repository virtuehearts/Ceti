"""Web front end tests: the ceti.web HTTP API, ceti/web_ui.html, and the
/web REPL command.

HTTP contract under test (all localhost, stdlib http.client):
  GET  /                    -> 200 text/html (the UI)
  POST /api/ask             -> SSE stream: tag/chunk/done events;
                               done carries text/display_tag/tier/trace/
                               session_id. Bad JSON -> 400, missing
                               question -> 400.
  GET/POST /api/sessions, GET/DELETE /api/sessions/<id>
  GET/POST /api/settings    -> key fields are "set"/"not set" only, never
                               the value; POST writes data/.env mode 0600
  POST /api/tts             -> audio bytes via the PocketTTS bridge
                               (fake bridge here); bridge down -> 502 JSON
  GET  /api/node            -> fingerprint/node_id, no key material
  GET  /api/health          -> booleans only
  unknown path              -> 404 JSON

SEAMS REQUIRED OF ceti/web.py (beyond the HTTP contract above):
  * ceti.web.make_server(host, port, data_dir) -> a bound
    http.server.HTTPServer (port may be 0 for an ephemeral port; read the
    bound port from server.server_address[1]). The server builds its own
    Brain on the scratch data_dir given, never the real
    ~/workspace/ceti/data. The server object carries .brain.
  * The /api/ask answer path is server.brain.stream_tokens(question)
    (yields (kind, payload) tuples: "tag"/"chunk"/"block"/"heartbeat"/
    "error"/"stream_end") followed by server.brain.finish_stream(args)
    -> Answer(text, display_tag, tier, trace). The ask tests mock these
    two methods, so no Ollama/cloud call ever happens.
  * ceti.cli.handle_web_command(arg) for the REPL tests ("/web" starts,
    "/web stop" stops; stop with nothing running must not crash; the
    handler keeps state in cli._web_state).
  * GET / serves the file ceti/web_ui.html (next to ceti/web.py).

If ceti.web / the handler / web_ui.html are absent (parallel builders),
the affected tests SKIP with an explicit reason instead of failing.

JS syntax check for web_ui.html uses `node --check` (node is on PATH);
falls back to a balanced-braces scan otherwise.

No network beyond 127.0.0.1. No cloud keys (OPENROUTER_API_KEY /
NVIDIA_API_KEY are scrubbed for every server test). Every server test
uses a scratch data dir that is removed afterwards.
"""
import contextlib
import http.client
import io
import json
import os
import re
import shutil
import socket
import stat
import struct
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock
from urllib.parse import urlsplit

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# The task names ceti/web_ui.html; the file currently lives at the repo
# root (frontend builder's working copy, 2026-10-18). Accept either, and
# note which one was used.
WEB_UI_CANDIDATES = (
    os.path.join(REPO_ROOT, "ceti", "web_ui.html"),
    os.path.join(REPO_ROOT, "web_ui.html"),
)

FAKE_KEY_A = "sk-or-fake-AAAAAAAAAAAAAAAA"
FAKE_KEY_B = "sk-fake-BBBBBBBBBBBBBBBB"

FINGERPRINT_RE = re.compile(
    r"^[A-Z0-9]{1,12}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}$")


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

def _require_web():
    """Import ceti.web and check the make_server seam; SkipTest when the
    backend builder has not landed it yet."""
    try:
        from ceti import web as webmod
    except ImportError as e:
        raise unittest.SkipTest(
            "ceti.web not importable (backend not built yet): %s" % e)
    if not hasattr(webmod, "make_server"):
        raise unittest.SkipTest(
            "ceti.web.make_server(host, port, data_dir) missing "
            "(backend not built yet)")
    return webmod


def _find_web_ui():
    for path in WEB_UI_CANDIDATES:
        if os.path.isfile(path):
            return path
    raise unittest.SkipTest(
        "web_ui.html not found at %s (frontend not built yet)"
        % " or ".join(WEB_UI_CANDIDATES))


@contextlib.contextmanager
def _started(tmpdir):
    """Start the real ceti.web server on an ephemeral port against a
    scratch data dir. Yields (base_url, webmod, server).

    os.environ is scrubbed for the duration (CETI_DATA_DIR -> tmpdir,
    cloud keys removed); mock.patch.dict restores it on exit, so keys
    the server loads from data/.env never leak into other tests."""
    webmod = _require_web()
    with mock.patch.dict(os.environ, {"CETI_DATA_DIR": tmpdir}):
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.pop("NVIDIA_API_KEY", None)
        srv = webmod.make_server("127.0.0.1", 0, tmpdir)
        try:
            thread = threading.Thread(target=srv.serve_forever,
                                      daemon=True)
            thread.start()
            port = srv.server_address[1]
            assert port != 0, "server did not bind an ephemeral port"
            yield "http://127.0.0.1:%d" % port, webmod, srv
        finally:
            srv.shutdown()
            srv.server_close()


@contextlib.contextmanager
def _serve():
    """_started with a fresh scratch dir that is removed afterwards."""
    tmp = tempfile.mkdtemp(prefix="ceti-web-test-")
    try:
        with _started(tmp) as (base, webmod, srv):
            yield base, tmp, webmod, srv
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _request(base, method, path, body=None, raw_body=None,
             headers=None, timeout=15):
    """One HTTP request; returns (status, headers_dict, body_bytes)."""
    parts = urlsplit(base)
    conn = http.client.HTTPConnection(parts.hostname, parts.port,
                                      timeout=timeout)
    try:
        data = raw_body
        hdrs = dict(headers or {})
        if data is None and body is not None:
            data = (body if isinstance(body, bytes)
                    else json.dumps(body).encode("utf-8"))
            hdrs.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=data, headers=hdrs)
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read()
    finally:
        conn.close()


def _parse_sse(payload: bytes):
    """Parse an SSE byte stream into a list of JSON event payloads.

    Accepts the event name either as {"event": ...} inside the data JSON
    or as an SSE `event:` field line on the frame.
    """
    events = []
    text = payload.decode("utf-8", "replace")
    for frame in text.split("\n\n"):
        ev_name = None
        data_lines = []
        for line in frame.splitlines():
            if line.startswith("event:"):
                ev_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if not data_lines:
            continue
        raw = "\n".join(data_lines).strip()
        if raw in ("", "[DONE]"):
            continue
        try:
            obj = json.loads(raw)
        except ValueError:
            obj = {"_raw": raw}
        if isinstance(obj, dict) and ev_name and "event" not in obj:
            obj = dict(obj, event=ev_name)
        events.append(obj)
    return events


def _fake_wav(data_len: int = 3200) -> bytes:
    byte_rate = 32000
    header = b"RIFF" + struct.pack("<I", 36 + data_len) + b"WAVE"
    fmt = (b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000,
                                 byte_rate, 2, 16))
    return (header + fmt + b"data" + struct.pack("<I", data_len)
            + b"\x00" * data_len)


FAKE_AUDIO = _fake_wav()


def _bridge_handler(seen: list):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            try:
                seen.append(json.loads(body.decode("utf-8")))
            except Exception:
                seen.append({})
            if self.path != "/v1/audio/speech":
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"detail": "not found"}')
                return
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(FAKE_AUDIO)))
            self.end_headers()
            self.wfile.write(FAKE_AUDIO)

    return Handler


@contextlib.contextmanager
def _fake_bridge():
    """Stub PocketTTS bridge on an ephemeral port (cf. test_tts.py)."""
    seen: list = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0),
                              _bridge_handler(seen))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d" % srv.server_address[1], seen
    finally:
        srv.shutdown()
        srv.server_close()


def _unused_url() -> str:
    """A localhost URL with nothing listening (bridge down)."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return "http://127.0.0.1:%d" % port


FAKE_ANSWER = {"text": "four", "display_tag": "calc",
               "tier": "calc", "trace": ["calc-tier"]}


@contextlib.contextmanager
def _fake_brain_answer(server):
    """Deterministic /api/ask answers with no Ollama/cloud: mock the two
    brain methods the web handler calls (stream_tokens yields the SSE
    ladder, finish_stream returns the final Answer)."""
    ans = mock.Mock()
    ans.text = FAKE_ANSWER["text"]
    ans.display_tag = FAKE_ANSWER["display_tag"]
    ans.tier = FAKE_ANSWER["tier"]
    ans.trace = list(FAKE_ANSWER["trace"])
    events = iter([("tag", "calc"),
                   ("chunk", "four"),
                   ("stream_end", {"answer": ans})])

    def fake_stream(question, *args, **kwargs):
        assert question == "what is 2+2", question
        return events

    with mock.patch.object(server.brain, "stream_tokens",
                           side_effect=fake_stream), \
         mock.patch.object(server.brain, "finish_stream",
                           return_value=ans):
        yield ans


def _session_list(data):
    """GET /api/sessions returns a bare list of summaries."""
    if isinstance(data, list):
        return data
    return data.get("sessions", [])


def _served_index(base):
    """GET / body, or SkipTest when ceti/web_ui.html is not in place
    yet (the backend 404s loudly in that case). Any other failure is
    a real failure."""
    status, headers, body = _request(base, "GET", "/")
    if status == 404 and b"not been built yet" in body:
        root_copy = WEB_UI_CANDIDATES[1]
        hint = (" (a copy exists at repo-root web_ui.html — move it "
                "next to ceti/web.py)" if os.path.isfile(root_copy)
                else "")
        raise unittest.SkipTest(
            "GET / 404s: ceti/web_ui.html not in place yet%s" % hint)
    return status, headers, body


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestIndexPage(unittest.TestCase):
    def test_root_serves_html_ui(self):
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = _served_index(base)
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers.get("Content-Type", ""))
        text = body.decode("utf-8", "replace")
        self.assertIn("<title>", text)

    def test_root_contains_ui_markers(self):
        """The served page is the ChatGPT-like UI: sidebar, settings,
        streaming ask path, save-chat."""
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = _served_index(base)
        text = body.decode("utf-8", "replace").lower()
        for marker in ("sidebar", "settings", "/api/ask"):
            self.assertIn(marker, text,
                          "GET / missing UI marker %r" % marker)
        self.assertTrue("eventsource" in text or "getreader" in text,
                        "GET / has no SSE handling marker")

    def test_root_matches_web_ui_file(self):
        """GET / serves the same UI as the web_ui.html source file."""
        path = _find_web_ui()
        with open(path, encoding="utf-8") as f:
            file_html = f.read()
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = _served_index(base)
        served = body.decode("utf-8", "replace")
        m = re.search(r"<title>(.*?)</title>", file_html, re.S | re.I)
        if m:
            self.assertIn(m.group(1).strip(), served,
                          "served page title differs from web_ui.html")


# ---------------------------------------------------------------------------
# POST /api/ask (SSE)
# ---------------------------------------------------------------------------

class TestAskSSE(unittest.TestCase):
    def _ask(self, base, srv, payload, raw=None):
        with _fake_brain_answer(srv):
            if raw is not None:
                return _request(base, "POST", "/api/ask",
                                raw_body=raw,
                                headers={"Content-Type":
                                         "application/json"})
            return _request(base, "POST", "/api/ask", payload)

    def _done(self, events):
        return next(e for e in events
                    if isinstance(e, dict) and e.get("event") == "done")

    def test_ask_streams_tag_chunk_done(self):
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = self._ask(
                base, srv, {"question": "what is 2+2"})
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream",
                      headers.get("Content-Type", ""))
        events = _parse_sse(body)
        kinds = [e.get("event") for e in events
                 if isinstance(e, dict)]
        self.assertIn("chunk", kinds, "no chunk event in %r" % kinds)
        self.assertIn("done", kinds, "no done event in %r" % kinds)
        self.assertIn("tag", kinds, "no tag event in %r" % kinds)

    def test_done_carries_answer_shape(self):
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = self._ask(
                base, srv, {"question": "what is 2+2"})
        done = self._done(_parse_sse(body))
        for key in ("text", "display_tag", "tier", "trace",
                    "session_id"):
            self.assertIn(key, done,
                          "done event missing %r: %r" % (key, done))
        self.assertEqual(done["text"], "four")
        self.assertEqual(done["display_tag"], "calc")
        self.assertEqual(done["tier"], "calc")
        self.assertEqual(done["trace"], ["calc-tier"])
        self.assertTrue(done["session_id"],
                        "done event has empty session_id")

    def test_chunks_reassemble_to_done_text(self):
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = self._ask(
                base, srv, {"question": "what is 2+2"})
        events = _parse_sse(body)
        chunks = [e for e in events
                  if isinstance(e, dict) and e.get("event") == "chunk"]
        self.assertTrue(chunks, "no chunk events to reassemble")
        text = "".join(e.get("text", e.get("chunk", "")) for e in chunks)
        done = self._done(events)
        self.assertEqual(text, done["text"],
                         "chunks %r do not reassemble to done text %r"
                         % (text, done["text"]))

    def test_tag_event_names_display_tag(self):
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = self._ask(
                base, srv, {"question": "what is 2+2"})
        events = _parse_sse(body)
        tags = [e for e in events
                if isinstance(e, dict) and e.get("event") == "tag"]
        self.assertTrue(tags, "no tag event")
        self.assertIn("calc", json.dumps(tags),
                      "tag event does not name the display_tag")

    def test_ask_with_session_id_echoes_it(self):
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = self._ask(
                base, srv, {"question": "what is 2+2",
                            "session_id": "sess-abc"})
        done = self._done(_parse_sse(body))
        self.assertEqual(done["session_id"], "sess-abc")

    def test_ask_without_session_id_mints_one(self):
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = self._ask(
                base, srv, {"question": "what is 2+2"})
            done = self._done(_parse_sse(body))
            sid = done["session_id"]
            self.assertTrue(isinstance(sid, str) and sid.strip())
            # the minted session is retrievable and holds the exchange
            status, headers, body = _request(
                base, "GET", "/api/sessions/%s" % sid)
            self.assertEqual(status, 200)
            data = json.loads(body.decode("utf-8"))
            self.assertIn("messages", data)
            blob = json.dumps(data)
            self.assertIn("what is 2+2", blob)
            self.assertIn("four", blob)

    def test_ask_bad_json_is_400(self):
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = self._ask(
                base, srv, None, raw=b"{not json")
        self.assertEqual(status, 400)

    def test_ask_missing_question_is_400(self):
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = self._ask(
                base, srv, {"nope": 1})
        self.assertEqual(status, 400)

    def test_ask_empty_question_is_400(self):
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = self._ask(
                base, srv, {"question": "   "})
        self.assertEqual(status, 400)

    def test_ask_wrong_method_rejected(self):
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = _request(base, "GET", "/api/ask")
        self.assertGreaterEqual(status, 400,
                                "GET /api/ask should not succeed")


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------

class TestSessions(unittest.TestCase):
    def test_list_starts_empty(self):
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = _request(base, "GET",
                                             "/api/sessions")
        self.assertEqual(status, 200)
        self.assertIn("application/json",
                      headers.get("Content-Type", ""))
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(_session_list(data), [])

    def test_create_returns_id(self):
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = _request(base, "POST",
                                             "/api/sessions", {})
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertTrue(data.get("id"),
                        "create did not return an id: %r" % data)

    def test_create_then_list_then_get(self):
        with _serve() as (base, tmp, webmod, srv):
            _, _, body = _request(base, "POST", "/api/sessions", {})
            sid = json.loads(body.decode("utf-8"))["id"]
            _, _, body = _request(base, "GET", "/api/sessions")
            ids = [s.get("id") for s in _session_list(
                json.loads(body.decode("utf-8")))]
            self.assertIn(sid, ids)
            status, _, body = _request(base, "GET",
                                       "/api/sessions/%s" % sid)
            self.assertEqual(status, 200)
            data = json.loads(body.decode("utf-8"))
            self.assertIn("messages", data)
            self.assertIsInstance(data["messages"], list)

    def test_get_unknown_is_404(self):
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = _request(
                base, "GET", "/api/sessions/nope-not-here")
        self.assertEqual(status, 404)

    def test_delete_unknown_is_404(self):
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = _request(
                base, "DELETE", "/api/sessions/nope-not-here")
        self.assertEqual(status, 404)

    def test_delete_removes_session(self):
        with _serve() as (base, tmp, webmod, srv):
            _, _, body = _request(base, "POST", "/api/sessions", {})
            sid = json.loads(body.decode("utf-8"))["id"]
            status, _, _ = _request(base, "DELETE",
                                    "/api/sessions/%s" % sid)
            self.assertEqual(status, 200)
            status, _, _ = _request(base, "GET",
                                    "/api/sessions/%s" % sid)
            self.assertEqual(status, 404)

    def test_ask_appends_to_session_messages(self):
        with _serve() as (base, tmp, webmod, srv):
            with _fake_brain_answer(srv):
                status, _, body = _request(
                    base, "POST", "/api/ask",
                    {"question": "what is 2+2",
                     "session_id": "chat-1"})
            self.assertEqual(status, 200)
            status, _, body = _request(base, "GET",
                                       "/api/sessions/chat-1")
            self.assertEqual(status, 200)
            data = json.loads(body.decode("utf-8"))
            msgs = data.get("messages", [])
            self.assertGreaterEqual(len(msgs), 2,
                                    "expected Q+A messages: %r" % msgs)
            roles = [m.get("role") for m in msgs]
            self.assertIn("user", roles)
            self.assertIn("assistant", roles)
            blob = json.dumps(data)
            self.assertIn("what is 2+2", blob)
            self.assertIn("four", blob)

    def test_sessions_isolated_between_data_dirs(self):
        tmp_a = tempfile.mkdtemp(prefix="ceti-web-a-")
        tmp_b = tempfile.mkdtemp(prefix="ceti-web-b-")
        try:
            with _started(tmp_a) as (base_a, _, _), \
                    _started(tmp_b) as (base_b, _, _):
                _, _, body = _request(base_a, "POST",
                                      "/api/sessions", {})
                sid = json.loads(body.decode("utf-8"))["id"]
                status, _, _ = _request(
                    base_b, "GET", "/api/sessions/%s" % sid)
                self.assertEqual(
                    status, 404,
                    "session leaked across data dirs")
        finally:
            shutil.rmtree(tmp_a, ignore_errors=True)
            shutil.rmtree(tmp_b, ignore_errors=True)


# ---------------------------------------------------------------------------
# GET/POST /api/settings — keys are NEVER disclosed
# ---------------------------------------------------------------------------

class TestSettings(unittest.TestCase):
    @staticmethod
    def _keys(data):
        """The redacted key fields: name -> 'set'/'not set', found
        wherever the snapshot nests them (top level or under 'keys')."""
        found = {}

        def walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str) and v in ("set", "not set"):
                        found[k] = v
                    else:
                        walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    walk(v)

        walk(data)
        assert found, \
            "GET /api/settings has no set/not-set key fields: %r" % (data,)
        return found

    @staticmethod
    def _voice(data):
        """The tts voice in the snapshot, flat or nested."""
        if isinstance(data.get("tts"), dict):
            return data["tts"].get("voice")
        return data.get("tts_voice")

    def test_seeded_key_shows_set_never_value(self):
        """Seed a fake key into data/.env (0600) first; the API must say
        'set' and never echo the value anywhere in the body."""
        tmp = tempfile.mkdtemp(prefix="ceti-web-settings-")
        try:
            env_path = os.path.join(tmp, ".env")
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("OPENROUTER_API_KEY=%s\n" % FAKE_KEY_A)
            os.chmod(env_path, 0o600)
            with _started(tmp) as (base, webmod, srv):
                status, headers, body = _request(
                    base, "GET", "/api/settings")
            self.assertEqual(status, 200)
            raw = body.decode("utf-8", "replace")
            self.assertNotIn(FAKE_KEY_A, raw,
                             "GET /api/settings leaked the key value")
            keys = self._keys(json.loads(raw))
            self.assertEqual(keys.get("openrouter_key"), "set")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_unset_key_shows_not_set(self):
        with _serve() as (base, tmp, webmod, srv):
            status, _, body = _request(base, "GET", "/api/settings")
        keys = self._keys(json.loads(body.decode("utf-8")))
        self.assertIn("openrouter_key", keys)
        self.assertIn("nvidia_key", keys)
        for name, value in keys.items():
            self.assertIn(value, ("set", "not set"),
                          "key field %r not redacted: %r" % (name, value))
        self.assertIn("not set", list(keys.values()))

    def test_settings_body_never_contains_key_like_values(self):
        """No 'set'/'not set' dance elsewhere: scan the whole snapshot
        for anything shaped like a pasted key."""
        with _serve() as (base, tmp, webmod, srv):
            status, _, body = _request(base, "GET", "/api/settings")
        raw = body.decode("utf-8", "replace")
        for fake in (FAKE_KEY_A, FAKE_KEY_B):
            self.assertNotIn(fake, raw)

    def test_post_key_writes_0600_env_and_stays_redacted(self):
        with _serve() as (base, tmp, webmod, srv):
            _, _, body = _request(base, "GET", "/api/settings")
            keys = self._keys(json.loads(body.decode("utf-8")))
            unset = [k for k, v in keys.items() if v == "not set"]
            if not unset:
                self.skipTest("no unset key field to exercise")
            target = unset[0]
            status, _, body = _request(base, "POST", "/api/settings",
                                       {target: FAKE_KEY_B})
            self.assertEqual(status, 200)
            resp = json.loads(body.decode("utf-8"))
            self.assertTrue(resp.get("ok"), resp)
            self.assertIn(target, resp.get("updated", []))
            env_path = os.path.join(tmp, ".env")
            self.assertTrue(os.path.isfile(env_path),
                            "POST /api/settings did not create data/.env")
            self.assertEqual(stat.S_IMODE(os.stat(env_path).st_mode),
                             0o600,
                             "data/.env is not mode 0600")
            with open(env_path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn(FAKE_KEY_B, content,
                          "key was not persisted to data/.env")
            # follow-up GET: "set", never the value
            _, _, body = _request(base, "GET", "/api/settings")
            raw = body.decode("utf-8", "replace")
            self.assertNotIn(FAKE_KEY_B, raw,
                             "GET /api/settings leaked the posted key")
            keys = self._keys(json.loads(raw))
            self.assertEqual(keys[target], "set")

    def test_post_empty_key_clears_it(self):
        with _serve() as (base, tmp, webmod, srv):
            status, _, _ = _request(base, "POST", "/api/settings",
                                     {"nvidia_key": FAKE_KEY_B})
            self.assertEqual(status, 200)
            self.assertTrue(os.path.isfile(os.path.join(tmp, ".env")))
            status, _, _ = _request(base, "POST", "/api/settings",
                                     {"nvidia_key": ""})
            self.assertEqual(status, 200)
            self.assertFalse(
                os.path.isfile(os.path.join(tmp, ".env")),
                "clearing the last key should remove data/.env")
            _, _, body = _request(base, "GET", "/api/settings")
            keys = self._keys(json.loads(body.decode("utf-8")))
            self.assertEqual(keys["nvidia_key"], "not set")

    def test_post_non_key_setting_sticks(self):
        """tts_voice is a non-key setting: POST applies it live, shows
        it on the next GET, and persists it to data/web.json."""
        with _serve() as (base, tmp, webmod, srv):
            status, _, body = _request(base, "POST", "/api/settings",
                                       {"tts_voice": "mya-01"})
            self.assertEqual(status, 200)
            _, _, body = _request(base, "GET", "/api/settings")
            data = json.loads(body.decode("utf-8"))
            self.assertEqual(self._voice(data), "mya-01",
                             "non-key setting did not stick")
            web_json = os.path.join(tmp, "web.json")
            self.assertTrue(os.path.isfile(web_json),
                            "non-key setting not persisted to web.json")
            with open(web_json, encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved.get("tts_voice"), "mya-01")

    def test_post_unknown_setting_is_400(self):
        with _serve() as (base, tmp, webmod, srv):
            status, _, body = _request(base, "POST", "/api/settings",
                                       {"nope_not_a_setting": 1})
        self.assertEqual(status, 400)
        data = json.loads(body.decode("utf-8"))
        self.assertTrue(data.get("error"))

    def test_post_wrong_type_setting_is_400(self):
        with _serve() as (base, tmp, webmod, srv):
            status, _, _ = _request(base, "POST", "/api/settings",
                                     {"handler_mode": "yes"})
        self.assertEqual(status, 400)

    def test_post_settings_bad_json_is_400(self):
        with _serve() as (base, tmp, webmod, srv):
            status, _, _ = _request(base, "POST", "/api/settings",
                                    raw_body=b"{not json",
                                    headers={"Content-Type":
                                             "application/json"})
        self.assertEqual(status, 400)


# ---------------------------------------------------------------------------
# POST /api/tts
# ---------------------------------------------------------------------------

class TestTTS(unittest.TestCase):
    def test_tts_returns_audio_from_fake_bridge(self):
        with _serve() as (base, tmp, webmod, srv):
            with _fake_bridge() as (bridge_url, seen):
                with mock.patch.dict(os.environ,
                                     {"CETI_TTS_URL": bridge_url}):
                    status, headers, body = _request(
                        base, "POST", "/api/tts",
                        {"text": "hello world"})
        self.assertEqual(status, 200)
        self.assertTrue(
            headers.get("Content-Type", "").startswith("audio/"),
            "not audio: %r" % (headers.get("Content-Type"),))
        self.assertEqual(body, FAKE_AUDIO)
        self.assertEqual(seen[0]["input"], "hello world")

    def test_tts_default_voice_is_mya03(self):
        with _serve() as (base, tmp, webmod, srv):
            with _fake_bridge() as (bridge_url, seen):
                with mock.patch.dict(os.environ,
                                     {"CETI_TTS_URL": bridge_url}):
                    status, _, _ = _request(
                        base, "POST", "/api/tts", {"text": "hi"})
        self.assertEqual(status, 200)
        self.assertEqual(seen[0]["voice"], "mya-03")

    def test_tts_voice_override_passes_through(self):
        with _serve() as (base, tmp, webmod, srv):
            with _fake_bridge() as (bridge_url, seen):
                with mock.patch.dict(os.environ,
                                     {"CETI_TTS_URL": bridge_url}):
                    status, _, _ = _request(
                        base, "POST", "/api/tts",
                        {"text": "hi", "voice": "mya-01"})
        self.assertEqual(status, 200)
        self.assertEqual(seen[0]["voice"], "mya-01")

    def test_tts_bridge_down_is_502_json_never_fake_audio(self):
        with _serve() as (base, tmp, webmod, srv):
            with mock.patch.dict(os.environ,
                                 {"CETI_TTS_URL": _unused_url()}):
                status, headers, body = _request(
                    base, "POST", "/api/tts", {"text": "hi"})
        self.assertEqual(status, 502,
                         "bridge down must be 502, got %d" % status)
        self.assertIn("application/json",
                      headers.get("Content-Type", ""))
        data = json.loads(body.decode("utf-8"))
        self.assertTrue(data.get("error"),
                        "502 body has no error field: %r" % (data,))
        self.assertNotEqual(body, FAKE_AUDIO)

    def test_tts_empty_text_is_400(self):
        with _serve() as (base, tmp, webmod, srv):
            with _fake_bridge() as (bridge_url, seen):
                with mock.patch.dict(os.environ,
                                     {"CETI_TTS_URL": bridge_url}):
                    status, _, _ = _request(
                        base, "POST", "/api/tts", {"text": "   "})
        self.assertEqual(status, 400)
        self.assertEqual(seen, [], "bridge called for empty text")

    def test_tts_uses_synthesize_bytes_seam(self):
        """The web layer speaks through ceti.tts.synthesize_bytes, so a
        monkeypatched bridge proves the wiring without HTTP."""
        from ceti import tts as tts_mod
        with _serve() as (base, tmp, webmod, srv):
            with mock.patch.object(
                    tts_mod, "synthesize_bytes",
                    return_value=(b"BYTES", "audio/wav")) as m:
                status, headers, body = _request(
                    base, "POST", "/api/tts", {"text": "hi"})
        self.assertTrue(m.called,
                        "web layer did not call ceti.tts.synthesize_bytes")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"BYTES")

    def test_tts_bad_json_is_400(self):
        with _serve() as (base, tmp, webmod, srv):
            status, _, _ = _request(base, "POST", "/api/tts",
                                    raw_body=b"{not json",
                                    headers={"Content-Type":
                                             "application/json"})
        self.assertEqual(status, 400)


# ---------------------------------------------------------------------------
# GET /api/node, GET /api/health
# ---------------------------------------------------------------------------

class TestNodeHealth(unittest.TestCase):
    def test_node_has_fingerprint_and_node_id(self):
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = _request(base, "GET", "/api/node")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        fp = data.get("fingerprint")
        nid = data.get("node_id", data.get("nodeId"))
        self.assertTrue(fp, "no fingerprint in %r" % (data,))
        self.assertTrue(nid, "no node_id in %r" % (data,))
        self.assertRegex(fp, FINGERPRINT_RE,
                         "fingerprint has wrong shape: %r" % fp)
        self.assertTrue(nid.startswith("jn1"),
                        "node_id has wrong shape: %r" % nid)

    def test_node_exposes_no_key_material(self):
        with _serve() as (base, tmp, webmod, srv):
            status, _, body = _request(base, "GET", "/api/node")
        raw = body.decode("utf-8", "replace")
        data = json.loads(raw)
        for key in data:
            self.assertNotIn("private", key.lower(),
                             "private key field exposed: %r" % key)
            self.assertNotIn("secret", key.lower(),
                             "secret field exposed: %r" % key)
        # the scratch identity's actual secret must not appear
        from ceti import signing
        ident = signing.ensure_identity(tmp, quiet=True)
        secret_hex = ident.identity_dict()["secret"]
        self.assertNotIn(secret_hex, raw,
                         "/api/node leaked the identity secret")
        self.assertNotIn(secret_hex.upper(), raw)

    def test_health_is_booleans_only(self):
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = _request(base, "GET",
                                             "/api/health")
        self.assertEqual(status, 200)
        self.assertIn("application/json",
                      headers.get("Content-Type", ""))
        data = json.loads(body.decode("utf-8"))
        self.assertIsInstance(data, dict)
        self.assertTrue(data, "empty /api/health payload")
        for key, value in data.items():
            self.assertIsInstance(
                value, bool,
                "/api/health[%r] is not a boolean: %r" % (key, value))


# ---------------------------------------------------------------------------
# error responses
# ---------------------------------------------------------------------------

class TestErrorResponses(unittest.TestCase):
    def test_unknown_path_is_404_json(self):
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = _request(
                base, "GET", "/api/does-not-exist")
        self.assertEqual(status, 404)
        self.assertIn("application/json",
                      headers.get("Content-Type", ""))
        data = json.loads(body.decode("utf-8"))  # must parse
        self.assertTrue(data.get("error"),
                        "404 body has no error field: %r" % (data,))

    def test_unknown_api_post_is_404_json(self):
        with _serve() as (base, tmp, webmod, srv):
            status, headers, body = _request(
                base, "POST", "/api/nope", {"a": 1})
        self.assertEqual(status, 404)
        self.assertIn("application/json",
                      headers.get("Content-Type", ""))

    def test_malformed_json_is_400(self):
        with _serve() as (base, tmp, webmod, srv):
            for path in ("/api/ask", "/api/settings", "/api/tts"):
                status, _, _ = _request(
                    base, "POST", path, raw_body=b"\xff\xfe{",
                    headers={"Content-Type": "application/json"})
                self.assertEqual(
                    status, 400,
                    "malformed JSON to %s gave %d, want 400"
                    % (path, status))


# ---------------------------------------------------------------------------
# REPL: /web command wiring in ceti/cli.py
# ---------------------------------------------------------------------------

class TestReplWebCommand(unittest.TestCase):
    def _need_handler(self):
        from ceti import cli
        if not hasattr(cli, "handle_web_command"):
            self.skipTest("ceti.cli.handle_web_command missing "
                          "(backend not built yet)")
        return cli

    def test_handler_exists(self):
        self._need_handler()

    def test_web_starts_server_thread_and_prints_url(self):
        cli = self._need_handler()
        from ceti import web as webmod
        fake = mock.Mock()
        threads_made = []
        real_thread = threading.Thread

        def spy_thread(*args, **kwargs):
            t = real_thread(*args, **kwargs)
            threads_made.append(t)
            return t

        buf = io.StringIO()
        with mock.patch.object(webmod, "make_server",
                               return_value=fake) as make_server:
            with mock.patch("threading.Thread",
                            side_effect=spy_thread):
                with contextlib.redirect_stdout(buf):
                    cli.handle_web_command("")
        self.assertTrue(make_server.called,
                        "handle_web_command did not start the server")
        # default host/port from ceti.web
        make_server.assert_called_with(webmod.DEFAULT_HOST,
                                       webmod.DEFAULT_PORT)
        self.assertTrue(threads_made,
                        "handle_web_command did not start a thread")
        for t in threads_made:
            t.join(timeout=5)
        out = buf.getvalue()
        # the handler prints the requested URL (default port 8899)
        self.assertIn(str(webmod.DEFAULT_PORT), out,
                      "no port in /web output: %r" % out)
        self.assertIn("http", out.lower(),
                      "no URL in /web output: %r" % out)
        # leave no fake server registered for later tests
        cli._web_state.update(server=None, thread=None,
                              host=None, port=None)

    def test_web_stop_with_nothing_running_does_not_crash(self):
        cli = self._need_handler()
        # the true nothing-running path: clear the handler's state
        cli._web_state.update(server=None, thread=None,
                              host=None, port=None)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.handle_web_command("stop")  # must not raise
        self.assertIn("not running", buf.getvalue().lower(),
                      "unexpected /web stop output: %r"
                      % buf.getvalue())
        # stopping twice in a row is also fine
        with contextlib.redirect_stdout(io.StringIO()):
            cli.handle_web_command("stop")


# ---------------------------------------------------------------------------
# frontend file: ceti/web_ui.html (or repo-root web_ui.html)
# ---------------------------------------------------------------------------

def _inline_scripts(html: str):
    """Inline <script> blocks (no src=): the app's JS."""
    return [m.group(1) for m in
            re.finditer(r"<script(?![^>]*\bsrc\b)[^>]*>(.*?)</script>",
                        html, re.S | re.I)]


def _assert_balanced(testcase, src: str, label: str):
    """Fallback syntax sanity scan when node is unavailable: balanced
    (), [], {} outside strings/comments."""
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    i, n = 0, len(src)
    in_str = None
    while i < n:
        ch = src[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == in_str:
                in_str = None
        elif ch in ("'", '"', "`"):
            in_str = ch
        elif ch == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        elif ch == "/" and i + 1 < n and src[i + 1] == "*":
            end = src.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        elif ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            testcase.assertTrue(
                stack, "%s: unbalanced %r" % (label, ch))
            testcase.assertEqual(
                stack.pop(), pairs[ch],
                "%s: mismatched %r" % (label, ch))
        i += 1
    testcase.assertFalse(stack, "%s: unclosed %r" % (label, stack))


class TestFrontendFile(unittest.TestCase):
    js_check_method = None  # filled in by test_inline_scripts_parse

    def _html(self):
        path = _find_web_ui()
        with open(path, encoding="utf-8") as f:
            return f.read(), path

    def test_file_is_html(self):
        html, path = self._html()
        self.assertIn("<!DOCTYPE html>", html[:200],
                      "%s does not look like HTML" % path)

    def test_sidebar_marker(self):
        html, _ = self._html()
        low = html.lower()
        self.assertIn("sidebar", low)
        self.assertIn('id="sessions"', html)

    def test_settings_modal_marker(self):
        html, _ = self._html()
        self.assertIn('id="settingsBtn"', html)
        self.assertIn('id="modal"', html)
        self.assertIn('id="saveSettings"', html)

    def test_waveform_canvas_marker(self):
        html, _ = self._html()
        low = html.lower()
        self.assertIn("canvas", low, "no canvas for the waveform")
        self.assertIn("waveform", low, "no waveform marker")

    def test_sse_handling_marker(self):
        html, _ = self._html()
        # the UI streams /api/ask with fetch + a frame parser
        # (parseSSE), or with EventSource
        self.assertTrue("EventSource" in html
                        or ("getReader" in html
                            and "parseSSE" in html),
                        "no SSE handling (EventSource or fetch+parseSSE)")

    def test_save_chat_marker(self):
        html, _ = self._html()
        low = html.lower()
        self.assertTrue("savebtn" in low or "save-chat" in low
                        or "savechat" in low,
                        "no save-chat control")

    def test_api_endpoints_referenced(self):
        html, _ = self._html()
        for ep in ("/api/ask", "/api/sessions", "/api/settings",
                   "/api/tts", "/api/health", "/api/node"):
            self.assertIn(ep, html,
                          "frontend never calls %s" % ep)

    def test_inline_scripts_parse(self):
        html, path = self._html()
        scripts = _inline_scripts(html)
        self.assertTrue(scripts,
                        "%s has no inline <script> blocks" % path)
        if shutil.which("node"):
            TestFrontendFile.js_check_method = "node --check"
            for i, src in enumerate(scripts):
                with tempfile.NamedTemporaryFile(
                        "w", suffix=".js", delete=False,
                        encoding="utf-8") as f:
                    f.write(src)
                    js_path = f.name
                try:
                    r = subprocess.run(
                        ["node", "--check", js_path],
                        capture_output=True, text=True, timeout=30)
                finally:
                    os.unlink(js_path)
                self.assertEqual(
                    r.returncode, 0,
                    "inline script #%d in %s failed node --check:\n%s"
                    % (i, path, r.stderr))
        else:
            TestFrontendFile.js_check_method = \
                "balanced-braces scan (node not on PATH)"
            for i, src in enumerate(scripts):
                _assert_balanced(
                    self, src, "inline script #%d in %s" % (i, path))


if __name__ == "__main__":
    unittest.main()
