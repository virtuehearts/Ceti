"""TTS — speak through the PocketTTS bridge.

The bridge is the sovereign voice layer: a self-hosted FastAPI app
with an OpenAI-compatible POST /v1/audio/speech endpoint, normally
at http://127.0.0.1:8000 with auth disabled for local dev
(ENABLE_AUTH=false). The working voice is mya-03 (cloned 2026-09-24).

Fail LOUD, always: if the bridge is down, returns an error, returns
empty audio, or returns something that isn't audio, this module
raises TTSError. It never fakes audio and never returns silence as
if it were speech. The tool layer turns TTSError into
ToolResult(success=False); the REPL prints it as a loud failure.

Env overrides (read fresh on every call, so tests can point at a
fake bridge without restarting anything):
  CETI_TTS_URL    base URL of the bridge (default http://127.0.0.1:8000)
  CETI_TTS_VOICE  default voice (default mya-03)
"""
from __future__ import annotations

import json
import os
import struct
import time
import urllib.error
import urllib.request
import uuid

DEFAULT_VOICE = "mya-03"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


class TTSError(Exception):
    """The bridge could not produce real speech. Loud by design."""


def bridge_url() -> str:
    return (os.environ.get("CETI_TTS_URL") or DEFAULT_BASE_URL).rstrip("/")


def default_voice() -> str:
    return (os.environ.get("CETI_TTS_VOICE") or DEFAULT_VOICE).strip().lower()


def parse_command(arg: str) -> tuple[str, str | None]:
    """Parse the /tts REPL argument: [/tts] [--voice NAME] <text>.

    Returns (text, voice-or-None). Raises TTSError on empty text or
    a --voice flag with no name, so the REPL can print usage loudly.
    """
    tokens = (arg or "").split()
    voice: str | None = None
    text_parts: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--voice":
            if i + 1 >= len(tokens):
                raise TTSError("usage: /tts [--voice NAME] <text> "
                               "(--voice needs a name)")
            voice = tokens[i + 1].strip().lower()
            i += 2
        elif tok.startswith("--voice="):
            voice = tok[len("--voice="):].strip().lower()
            if not voice:
                raise TTSError("usage: /tts [--voice NAME] <text> "
                               "(--voice= needs a name)")
            i += 1
        else:
            text_parts.append(tok)
            i += 1
    text = " ".join(text_parts).strip()
    if not text:
        raise TTSError("usage: /tts [--voice NAME] <text> (no text given)")
    return text, voice


def wav_duration_seconds(data: bytes) -> float | None:
    """Best-effort duration from a WAV header; None if unparsable."""
    try:
        if len(data) < 44 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
            return None
        byte_rate = struct.unpack("<I", data[28:32])[0]
        # data chunk: usually at offset 36 ("data" + size); scan for it.
        off = 12
        while off + 8 <= len(data):
            tag, size = data[off:off + 4], struct.unpack("<I", data[off + 4:off + 8])[0]
            if tag == b"data":
                if byte_rate == 0:
                    return None
                return size / byte_rate
            off += 8 + size + (size % 2)
        return None
    except (struct.error, IndexError):
        return None


def synthesize_bytes(text: str, voice: str | None = None,
                     base_url: str | None = None,
                     timeout_s: int = 60) -> tuple[bytes, str]:
    """POST text to the bridge; return (audio_bytes, content_type).

    Raises TTSError on every failure mode: empty text, bridge down,
    HTTP error, empty body, non-audio content type.
    """
    text = (text or "").strip()
    if not text:
        raise TTSError("TTS needs non-empty text")
    voice = (voice or default_voice()).strip().lower()
    if not voice:
        raise TTSError("TTS needs a voice (default: %s)" % DEFAULT_VOICE)
    url = (base_url or bridge_url()) + "/v1/audio/speech"
    payload = json.dumps({
        "model": "tts-1",
        "input": text,
        "voice": voice,
        "response_format": "wav",
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json",
                 "User-Agent": "CETI/0.3.3 (tts)"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            content_type = resp.headers.get_content_type() or ""
            data = resp.read()
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            detail = ""
        raise TTSError(
            "TTS bridge refused the request: HTTP %s %s — %s"
            % (e.code, e.reason, detail)) from e
    except urllib.error.URLError as e:
        raise TTSError(
            "TTS bridge unreachable at %s: %s — is the PocketTTS "
            "bridge running? (cd ~/workspace/pockettts-agent-bridge && "
            "ENABLE_AUTH=false ~/workspace/venvs/bridge/bin/python "
            "-m uvicorn app.main:app --host 127.0.0.1 --port 8000)"
            % (url, e.reason)) from e
    except (TimeoutError, OSError) as e:
        raise TTSError("TTS bridge request failed: %s" % e) from e
    if not data:
        raise TTSError(
            "TTS bridge returned EMPTY audio for voice %r — refusing "
            "to treat silence as speech" % voice)
    if not content_type.startswith("audio/"):
        raise TTSError(
            "TTS bridge returned content-type %r, not audio — "
            "refusing to treat it as speech" % content_type)
    return data, content_type


def synthesize_to_file(text: str, voice: str | None = None,
                       base_url: str | None = None,
                       out_dir: str | None = None,
                       filename: str | None = None,
                       timeout_s: int = 60) -> dict:
    """Speak text via the bridge and save the WAV to out_dir.

    Returns {"path", "bytes", "content_type", "voice", "duration_s"}.
    Raises TTSError on every failure (see synthesize_bytes).
    """
    data, content_type = synthesize_bytes(
        text, voice=voice, base_url=base_url, timeout_s=timeout_s)
    voice = (voice or default_voice()).strip().lower()
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), "tts")
    os.makedirs(out_dir, exist_ok=True)
    if filename:
        # Jail the name: bare filename only, no directories, no "..".
        filename = os.path.basename(filename.strip())
        if not filename or filename in (".", ".."):
            raise TTSError("TTS filename must be a plain file name")
        if not filename.lower().endswith(".wav"):
            filename += ".wav"
    else:
        filename = "tts-%s-%s.wav" % (
            time.strftime("%Y%m%d-%H%M%S", time.gmtime()),
            uuid.uuid4().hex[:8])
    path = os.path.join(out_dir, filename)
    with open(path, "wb") as f:
        f.write(data)
    return {"path": path, "bytes": len(data),
            "content_type": content_type, "voice": voice,
            "duration_s": wav_duration_seconds(data)}


def describe(result: dict) -> str:
    """One-line report for a synthesize_to_file result."""
    dur = result.get("duration_s")
    dur_txt = ("%.1fs" % dur) if dur is not None else "unknown duration"
    return ("spoke %d bytes as voice %r -> %s (%s, %s)"
            % (result["bytes"], result["voice"], result["path"],
               result["content_type"], dur_txt))
