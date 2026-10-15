"""LLM backends — OpenAI-compatible chat completions over stdlib urllib.

  OllamaBackend      local Qwen (or anything Ollama serves)
  OpenRouterBackend  cloud, key from OPENROUTER_API_KEY
  NvidiaBackend      cloud, key from NVIDIA_API_KEY

All three speak /v1/chat/completions. Keys are read from the
environment at call time and never logged or stored.
"""
from __future__ import annotations

import http.client
import json
import queue
import threading
import urllib.error
import urllib.parse
import urllib.request


class BackendError(Exception):
    """A backend call failed. retryable=False means retrying is futile
    (e.g. HTTP 4xx) — the retry helper re-raises immediately."""
    retryable = True


class BackendClientError(BackendError):
    """HTTP 4xx: the request itself is bad. Don't retry."""
    retryable = False


def _post_json(url: str, payload: dict, headers: dict,
               timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:400]
        if 400 <= e.code < 500:
            raise BackendClientError(f"HTTP {e.code} from {url}: {body}")
        raise BackendError(f"HTTP {e.code} from {url}: {body}")
    except urllib.error.URLError as e:
        raise BackendError(f"unreachable {url}: {e.reason}")
    except TimeoutError:
        raise BackendError(f"timeout after {timeout}s: {url}")
    except Exception as e:  # socket.timeout etc.
        raise BackendError(f"{type(e).__name__}: {e} ({url})")


def _to_wire(m: dict) -> dict:
    """Convert a CETI message to the OpenAI wire format.

    Internal assistant tool_calls are [{"id", "name", "arguments"}];
    the wire needs {"id", "type": "function", "function": {...}} with
    arguments as a JSON string. Tool results become role="tool"
    messages with tool_call_id.
    """
    role = m.get("role")
    if role == "assistant" and m.get("tool_calls"):
        return {
            "role": "assistant",
            "content": m.get("content") or None,
            "tool_calls": [
                {"id": tc.get("id"),
                 "type": "function",
                 "function": {
                     "name": tc.get("name"),
                     "arguments": json.dumps(tc.get("arguments") or {}),
                 }}
                for tc in m["tool_calls"]
            ],
        }
    if role == "tool":
        return {"role": "tool",
                "tool_call_id": m.get("tool_call_id"),
                "content": str(m.get("content") or "")}
    return {"role": role, "content": m.get("content") or ""}


class ChatBackend:
    """Base: complete(system, user) -> str."""
    name = "base"

    def complete(self, system: str, user: str,
                 max_tokens: int = 512,
                 temperature: float = 0.3) -> str:
        raise NotImplementedError


class _OpenAICompat(ChatBackend):
    def __init__(self, url: str, model: str, timeout: int,
                 api_key: str | None = None,
                 extra_headers: dict | None = None):
        self.url = url
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        self.extra_headers = extra_headers or {}

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        h.update(self.extra_headers)
        return h

    def complete(self, system: str, user: str,
                 max_tokens: int = 512,
                 temperature: float = 0.3) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        resp = _post_json(self.url, payload, self._headers(), self.timeout)
        try:
            text = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise BackendError(f"bad response shape: {e}")
        text = (text or "").strip()
        if not text:
            raise BackendError("empty completion")
        return text

    def complete_stream(self, system: str, user: str,
                        max_tokens: int = 512,
                        temperature: float = 0.3,
                        heartbeat: float = 0.1):
        """Stream a completion: yield content deltas as they arrive,
        None while waiting for the next chunk. Same payload as
        complete() but stream:true, read as SSE. The caller pumps
        this on the main thread — None means 'no chunk yet, check
        keys and pump again'. Raises BackendError on HTTP errors.
        The caller closes the generator to abort (Ctrl-C).

        Implementation: a daemon reader thread does the blocking
        readline and feeds a queue; the generator polls the queue
        with a timeout, so it never blocks on the socket. (A
        socket timeout on the buffered reader would poison it —
        CPython refuses further reads after one timeout — so the
        reader thread is the honest way to stay responsive.)"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        parts = urllib.parse.urlsplit(self.url)
        path = parts.path + ("?" + parts.query if parts.query else "")
        if parts.scheme == "https":
            conn = http.client.HTTPSConnection(
                parts.hostname, parts.port, timeout=self.timeout)
        else:
            conn = http.client.HTTPConnection(
                parts.hostname, parts.port, timeout=self.timeout)
        try:
            conn.request("POST", path,
                         body=json.dumps(payload).encode("utf-8"),
                         headers=self._headers())
            resp = conn.getresponse()
            if resp.status >= 400:
                body = resp.read().decode("utf-8", "replace")[:400]
                if 400 <= resp.status < 500:
                    raise BackendClientError(
                        f"HTTP {resp.status} from {self.url}: {body}")
                raise BackendError(
                    f"HTTP {resp.status} from {self.url}: {body}")
        except BackendError:
            try:
                conn.close()
            except Exception:
                pass
            raise
        except (http.client.HTTPException, OSError) as e:
            try:
                conn.close()
            except Exception:
                pass
            raise BackendError(f"{type(e).__name__}: {e} ({self.url})")

        q: queue.Queue = queue.Queue()
        stop = threading.Event()

        def reader() -> None:
            try:
                while not stop.is_set():
                    line = resp.readline()
                    if not line:
                        q.put(("done", None))
                        return
                    text = line.decode("utf-8", "replace").strip()
                    if not text.startswith("data:"):
                        continue
                    data = text[5:].strip()
                    if data == "[DONE]":
                        q.put(("done", None))
                        return
                    try:
                        obj = json.loads(data)
                    except ValueError:
                        continue
                    try:
                        delta = (obj["choices"][0]["delta"].get("content")
                                 or "")
                    except (KeyError, IndexError, TypeError):
                        continue
                    if delta:
                        q.put(("delta", delta))
            except Exception as e:
                if not stop.is_set():
                    q.put(("error", e))
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        threading.Thread(target=reader, daemon=True).start()
        try:
            while True:
                try:
                    kind, payload = q.get(timeout=heartbeat)
                except queue.Empty:
                    yield None
                    continue
                if kind == "delta":
                    yield payload
                elif kind == "done":
                    return
                else:  # error from the reader thread
                    raise BackendError(
                        f"{type(payload).__name__}: {payload} "
                        f"({self.url})")
        finally:
            # generator closed (Ctrl-C) or stream done: stop the
            # reader; closing the conn unblocks its readline
            stop.set()
            try:
                conn.close()
            except Exception:
                pass

    def complete_with_tools(self, messages: list[dict],
                            tools: list[dict],
                            max_tokens: int = 1024,
                            temperature: float = 0.3) -> dict:
        """Chat completion with OpenAI-style function calling.

        messages: [{"role": ..., "content": ...}, ...]. Assistant
        messages may carry tool_calls in CETI's internal shape
        ([{"id", "name", "arguments": dict}]); they are converted to
        the OpenAI wire format here.
        tools: schemas from tools.Tool.to_openai_schema().
        Returns {"content": str, "tool_calls": [{"id", "name",
        "arguments": dict}]}. Empty tool_calls means "task done".
        """
        payload = {
            "model": self.model,
            "messages": [_to_wire(m) for m in messages],
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        resp = _post_json(self.url, payload, self._headers(), self.timeout)
        try:
            msg = resp["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise BackendError(f"bad response shape: {e}")
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {}) or {}
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw)
                if not isinstance(args, dict):
                    args = {"_value": args}
            except (ValueError, TypeError):
                args = {"_raw": raw}
            calls.append({"id": tc.get("id") or "call-?",
                          "name": fn.get("name") or "?",
                          "arguments": args})
        return {"content": (msg.get("content") or "").strip(),
                "tool_calls": calls}


class OllamaBackend(_OpenAICompat):
    name = "local-qwen"

    def __init__(self, base_url: str = "http://127.0.0.1:11434",
                 model: str = "qwen2.5-coder:1.5b",
                 timeout: int = 120):
        super().__init__(f"{base_url.rstrip('/')}/v1/chat/completions",
                         model, timeout)


class OpenRouterBackend(_OpenAICompat):
    name = "openrouter"

    def __init__(self, api_key: str | None,
                 model: str = "qwen/qwen3-8b:free",
                 url: str = "https://openrouter.ai/api/v1/chat/completions",
                 timeout: int = 45):
        if not api_key:
            raise BackendError("OPENROUTER_API_KEY not set")
        super().__init__(url, model, timeout, api_key,
                         {"HTTP-Referer": "https://darknet.ca",
                          "X-Title": "brain-framework"})


class NvidiaBackend(_OpenAICompat):
    name = "nvidia"

    def __init__(self, api_key: str | None,
                 model: str = "qwen/qwen3-8b",
                 url: str = "https://integrate.api.nvidia.com/v1/chat/completions",
                 timeout: int = 45):
        if not api_key:
            raise BackendError("NVIDIA_API_KEY not set")
        super().__init__(url, model, timeout, api_key)
