import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ceti.backends import (_to_wire, BackendClientError, BackendError,
                            NvidiaBackend, OpenRouterBackend)
from ceti.retry import retry


class TestWireFormat(unittest.TestCase):
    def test_assistant_tool_calls(self):
        m = {"role": "assistant", "content": "doing it",
             "tool_calls": [{"id": "c1", "name": "read_file",
                             "arguments": {"path": "a.txt"}}]}
        w = _to_wire(m)
        self.assertEqual(w["role"], "assistant")
        tc = w["tool_calls"][0]
        self.assertEqual(tc["id"], "c1")
        self.assertEqual(tc["type"], "function")
        self.assertEqual(tc["function"]["name"], "read_file")
        # arguments must be a JSON STRING on the wire
        args = json.loads(tc["function"]["arguments"])
        self.assertEqual(args, {"path": "a.txt"})

    def test_tool_result(self):
        w = _to_wire({"role": "tool", "tool_call_id": "c1",
                      "name": "read_file", "content": "hello"})
        self.assertEqual(w, {"role": "tool", "tool_call_id": "c1",
                             "content": "hello"})

    def test_plain_passthrough(self):
        self.assertEqual(_to_wire({"role": "user", "content": "hi"}),
                         {"role": "user", "content": "hi"})

    def test_wire_roundtrip_through_backend(self):
        """complete_with_tools must send provider-valid messages."""
        backend = OpenRouterBackend("k", model="m", timeout=5)
        captured = {}

        def fake_post(url, payload, headers, timeout):
            captured["payload"] = payload
            return {"choices": [{"message": {
                "content": "done", "tool_calls": []}}]}

        msgs = [{"role": "user", "content": "go"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "c9", "name": "list_dir",
                                 "arguments": {"path": "."}}]},
                {"role": "tool", "tool_call_id": "c9",
                 "content": "a.txt"}]
        with mock.patch("ceti.backends._post_json", fake_post):
            resp = backend.complete_with_tools(msgs, [])
        sent = captured["payload"]["messages"]
        # assistant tool_calls on the wire
        atc = sent[1]["tool_calls"][0]
        self.assertEqual(atc["type"], "function")
        self.assertIn("function", atc)
        json.loads(atc["function"]["arguments"])  # valid JSON string
        # tool result shape
        self.assertEqual(sent[2]["role"], "tool")
        self.assertEqual(sent[2]["tool_call_id"], "c9")
        self.assertNotIn("name", sent[2])
        self.assertEqual(resp["content"], "done")


class TestNoRetryOn4xx(unittest.TestCase):
    def test_client_error_not_retried(self):
        calls = []

        def bad():
            calls.append(1)
            raise BackendClientError("HTTP 400: nope")

        with self.assertRaises(BackendClientError):
            retry(bad, attempts=3, delay=0,
                  exceptions=(BackendError,))
        self.assertEqual(len(calls), 1)

    def test_server_error_still_retried(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise BackendError("HTTP 500: oops")
            return "ok"

        self.assertEqual(retry(flaky, attempts=3, delay=0,
                               exceptions=(BackendError,)), "ok")
        self.assertEqual(len(calls), 3)


class TestProviderErrorPayload(unittest.TestCase):
    """A provider can answer HTTP 200 with an error payload instead
    of a completion (OpenRouter does this when its upstream times
    out). The backend must surface the provider's error, not "bad
    response shape: 'choices'"."""

    def _backend(self):
        return OpenRouterBackend("k", model="m", timeout=5)

    def test_504_error_payload_is_backend_error(self):
        def fake_post(url, payload, headers, timeout):
            return {"error": {"message": "The operation was aborted",
                              "code": 504,
                              "metadata": {"error_type": "timeout"}}}
        with mock.patch("ceti.backends._post_json", fake_post):
            with self.assertRaisesRegex(BackendError, "504.*aborted"):
                self._backend().complete("sys", "hi")

    def test_4xx_error_payload_is_client_error(self):
        def fake_post(url, payload, headers, timeout):
            return {"error": {"message": "model not found", "code": 404}}
        with mock.patch("ceti.backends._post_json", fake_post):
            with self.assertRaises(BackendClientError):
                self._backend().complete("sys", "hi")

    def test_error_payload_in_tools_call(self):
        def fake_post(url, payload, headers, timeout):
            return {"error": {"message": "upstream timeout", "code": 504}}
        with mock.patch("ceti.backends._post_json", fake_post):
            with self.assertRaises(BackendError):
                self._backend().complete_with_tools([], [])


class TestNvidiaModelId(unittest.TestCase):
    def test_free_suffix_stripped(self):
        # ":free" is an OpenRouter suffix; Nvidia 404s on it.
        b = NvidiaBackend("k", model="nvidia/nemotron-3.5-lightning:free")
        self.assertEqual(b.model, "nvidia/nemotron-3.5-lightning")

    def test_plain_model_untouched(self):
        b = NvidiaBackend("k", model="nvidia/nemotron-3.5-lightning")
        self.assertEqual(b.model, "nvidia/nemotron-3.5-lightning")


if __name__ == "__main__":
    unittest.main()
