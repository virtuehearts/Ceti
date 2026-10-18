"""TTS tool + /tts REPL command, against a fake PocketTTS bridge.

No live bridge needed: a stub HTTP server answers
POST /v1/audio/speech with canned WAV bytes (or errors), so the
suite proves the real HTTP path — request shape, voice default,
fail-loud behaviour — without the model.
"""
import contextlib
import io
import json
import os
import struct
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from ceti import cli, tools, tts


def _fake_wav(data_len: int = 3200) -> bytes:
    """Minimal valid WAV: PCM 16kHz mono 16-bit, data_len zero bytes."""
    byte_rate = 32000
    header = b"RIFF" + struct.pack("<I", 36 + data_len) + b"WAVE"
    fmt = (b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000,
                                 byte_rate, 2, 16))
    return header + fmt + b"data" + struct.pack("<I", data_len) + b"\x00" * data_len


FAKE_AUDIO = _fake_wav(3200)  # 0.1s of silence-shaped WAV


def _make_handler(mode: str, seen: list):
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
                self._send(404, b'{"detail": "not found"}', "application/json")
                return
            if mode == "not_found":
                self._send(404, b'{"detail": "voice not found"}',
                           "application/json")
            elif mode == "empty":
                self._send(200, b"", "audio/wav")
            elif mode == "not_audio":
                self._send(200, b"not audio at all", "text/html")
            elif mode == "server_error":
                self._send(500, b'{"detail": "boom"}', "application/json")
            else:
                self._send(200, FAKE_AUDIO, "audio/wav")

        def _send(self, code, payload, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


class _FakeBridge:
    """A stub PocketTTS bridge on an ephemeral port."""

    def __init__(self, mode: str = "ok"):
        self.seen: list = []
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), _make_handler(mode, self.seen))
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        return False

    @property
    def url(self) -> str:
        return "http://127.0.0.1:%d" % self.port


def _unused_url() -> str:
    """A localhost URL with nothing listening (bridge down)."""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return "http://127.0.0.1:%d" % port


class TestSynthesize(unittest.TestCase):
    def test_posts_openai_shape_with_default_voice(self):
        with _FakeBridge() as bridge:
            data, ctype = tts.synthesize_bytes(
                "hello world", base_url=bridge.url)
        self.assertEqual(data, FAKE_AUDIO)
        self.assertEqual(ctype, "audio/wav")
        req = bridge.seen[0]
        self.assertEqual(req["input"], "hello world")
        self.assertEqual(req["voice"], "mya-03")  # default voice
        self.assertEqual(req["response_format"], "wav")
        self.assertEqual(req["model"], "tts-1")

    def test_voice_override_sent(self):
        with _FakeBridge() as bridge:
            tts.synthesize_bytes("hi", voice="mya-01",
                                 base_url=bridge.url)
        self.assertEqual(bridge.seen[0]["voice"], "mya-01")

    def test_env_default_voice(self):
        with mock.patch.dict(os.environ, {"CETI_TTS_VOICE": "mya-02"}):
            with _FakeBridge() as bridge:
                tts.synthesize_bytes("hi", base_url=bridge.url)
        self.assertEqual(bridge.seen[0]["voice"], "mya-02")

    def test_empty_text_refused(self):
        with self.assertRaises(tts.TTSError):
            tts.synthesize_bytes("   ")

    def test_bridge_down_fails_loud(self):
        with self.assertRaises(tts.TTSError) as ctx:
            tts.synthesize_bytes("hi", base_url=_unused_url())
        self.assertIn("unreachable", str(ctx.exception))

    def test_unknown_voice_404_fails_loud(self):
        with _FakeBridge("not_found") as bridge:
            with self.assertRaises(tts.TTSError) as ctx:
                tts.synthesize_bytes("hi", voice="nope",
                                     base_url=bridge.url)
        self.assertIn("404", str(ctx.exception))

    def test_server_error_fails_loud(self):
        with _FakeBridge("server_error") as bridge:
            with self.assertRaises(tts.TTSError) as ctx:
                tts.synthesize_bytes("hi", base_url=bridge.url)
        self.assertIn("500", str(ctx.exception))

    def test_empty_audio_refused_never_silence_as_speech(self):
        with _FakeBridge("empty") as bridge:
            with self.assertRaises(tts.TTSError) as ctx:
                tts.synthesize_bytes("hi", base_url=bridge.url)
        self.assertIn("EMPTY", str(ctx.exception))

    def test_non_audio_refused(self):
        with _FakeBridge("not_audio") as bridge:
            with self.assertRaises(tts.TTSError) as ctx:
                tts.synthesize_bytes("hi", base_url=bridge.url)
        self.assertIn("not audio", str(ctx.exception))

    def test_synthesize_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _FakeBridge() as bridge:
                res = tts.synthesize_to_file(
                    "hi", base_url=bridge.url, out_dir=tmp)
            self.assertTrue(os.path.isfile(res["path"]))
            with open(res["path"], "rb") as f:
                self.assertEqual(f.read(), FAKE_AUDIO)
            self.assertEqual(res["bytes"], len(FAKE_AUDIO))
            self.assertEqual(res["voice"], "mya-03")
            self.assertAlmostEqual(res["duration_s"], 0.1, places=3)
            self.assertIn("spoke", tts.describe(res))

    def test_filename_jailed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _FakeBridge() as bridge:
                res = tts.synthesize_to_file(
                    "hi", base_url=bridge.url, out_dir=tmp,
                    filename="../../evil.wav")
            self.assertEqual(os.path.dirname(res["path"]), tmp)
            self.assertEqual(os.path.basename(res["path"]), "evil.wav")


class TestWavDuration(unittest.TestCase):
    def test_valid_header(self):
        self.assertAlmostEqual(tts.wav_duration_seconds(FAKE_AUDIO),
                               0.1, places=3)

    def test_garbage_returns_none(self):
        self.assertIsNone(tts.wav_duration_seconds(b"nope"))
        self.assertIsNone(tts.wav_duration_seconds(b""))


class TestParseCommand(unittest.TestCase):
    def test_plain(self):
        text, voice = tts.parse_command("hello world")
        self.assertEqual(text, "hello world")
        self.assertIsNone(voice)

    def test_voice_flag(self):
        text, voice = tts.parse_command("--voice mya-01 hello")
        self.assertEqual((text, voice), ("hello", "mya-01"))

    def test_voice_equals(self):
        text, voice = tts.parse_command("hello --voice=mya-02")
        self.assertEqual((text, voice), ("hello", "mya-02"))

    def test_empty_raises(self):
        with self.assertRaises(tts.TTSError):
            tts.parse_command("   ")

    def test_voice_without_name_raises(self):
        with self.assertRaises(tts.TTSError):
            tts.parse_command("--voice")


class TestTTSTool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._old = os.environ.get("CETI_WORK_ROOT")
        os.environ["CETI_WORK_ROOT"] = self.tmp

    def tearDown(self):
        if self._old is None:
            del os.environ["CETI_WORK_ROOT"]
        else:
            os.environ["CETI_WORK_ROOT"] = self._old

    def test_registered(self):
        self.assertIn("tts", tools.all_tools())

    def test_tool_saves_wav_in_work_root(self):
        with _FakeBridge() as bridge:
            with mock.patch.dict(os.environ, {"CETI_TTS_URL": bridge.url}):
                r = tools.execute("tts", text="hello world")
        self.assertTrue(r.success, r.error)
        self.assertIn("spoke", r.content)
        self.assertIn("mya-03", r.content)
        outdir = os.path.join(self.tmp, "tts")
        files = os.listdir(outdir)
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith(".wav"))
        with open(os.path.join(outdir, files[0]), "rb") as f:
            self.assertEqual(f.read(), FAKE_AUDIO)

    def test_tool_bridge_down_fails_loud_not_success(self):
        with mock.patch.dict(os.environ,
                             {"CETI_TTS_URL": _unused_url()}):
            r = tools.execute("tts", text="hello")
        self.assertFalse(r.success)
        self.assertIn("TTS FAILED", r.error)
        self.assertIn("unreachable", r.error)

    def test_tool_empty_text_fails(self):
        r = tools.execute("tts", text="   ")
        self.assertFalse(r.success)


class TestReplCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._old = os.environ.get("CETI_WORK_ROOT")
        os.environ["CETI_WORK_ROOT"] = self.tmp

    def tearDown(self):
        if self._old is None:
            del os.environ["CETI_WORK_ROOT"]
        else:
            os.environ["CETI_WORK_ROOT"] = self._old

    def _run(self, arg, url):
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"CETI_TTS_URL": url}):
            with contextlib.redirect_stdout(buf):
                cli.handle_tts_command(arg)
        return buf.getvalue()

    def test_repl_speaks_and_reports(self):
        with _FakeBridge() as bridge:
            out = self._run("hello world", bridge.url)
        self.assertIn("spoke", out)
        self.assertIn("mya-03", out)
        self.assertIn(".wav", out)

    def test_repl_voice_flag(self):
        with _FakeBridge() as bridge:
            out = self._run("--voice mya-01 hello", bridge.url)
        self.assertIn("mya-01", out)
        self.assertEqual(bridge.seen[0]["voice"], "mya-01")

    def test_repl_bridge_down_loud(self):
        out = self._run("hello", _unused_url())
        self.assertIn("TTS FAILED", out)
        self.assertIn("unreachable", out)

    def test_repl_empty_loud_usage(self):
        with _FakeBridge() as bridge:
            out = self._run("   ", bridge.url)
        self.assertIn("TTS FAILED", out)
        self.assertEqual(bridge.seen, [])  # never called the bridge


if __name__ == "__main__":
    unittest.main()
