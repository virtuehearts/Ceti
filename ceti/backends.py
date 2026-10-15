"""LLM backends — OpenAI-compatible chat completions over stdlib urllib.

  OllamaBackend      local Qwen (or anything Ollama serves)
  OpenRouterBackend  cloud, key from OPENROUTER_API_KEY
  NvidiaBackend      cloud, key from NVIDIA_API_KEY

All three speak /v1/chat/completions. Keys are read from the
environment at call time and never logged or stored.
"""
from __future__ import annotations

import json
import urllib.error
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
