"""Tests for ceti.trim — a faithful Python port of trim-1.

Parity and golden fixtures come from the upstream repository
(https://github.com/caveman-ai/trim-1, Apache-2.0): parity.bin replays rows
LightGBM scored at export time; golden.txt pins the full output on the
sample log. If the port drifts, these catch it.
"""

import os
import struct
import unittest

from ceti import trim

_HERE = os.path.dirname(__file__)
_TESTDATA = os.path.join(_HERE, "testdata_trim")
_LOG_QUERY = "Why did the checkout payment fail? Find the ledger error."


def _log_context():
    parts = ["$ kubectl logs deploy/checkout --since=1h\n"]
    for i in range(400):
        parts.append(
            "2026-10-11T03:%02d:%02dZ INFO checkout request %d served "
            "in %dms\n" % (i // 60 % 60, i % 60, 1000 + i, 10 + i % 17))
        if i == 271:
            parts.append(
                "2026-10-11T03:04:31Z ERROR payment client: ledger-db "
                "timeout after 30s (order 1271)\n")
            parts.append("    at LedgerClient.charge (ledger/client.go:88)\n")
    parts.append("exit status 1\n")
    return "".join(parts)


class TestTrim(unittest.TestCase):
    def test_prediction_parity(self):
        """Tree evaluation matches LightGBM's raw scores, bit for bit."""
        with open(os.path.join(_TESTDATA, "parity.bin"), "rb") as fh:
            data = fh.read()
        n, width = struct.unpack_from("<2I", data, 0)
        self.assertEqual(width, trim.N_FEATURES)
        self.assertEqual(len(data), 8 + n * width * 4 + n * 8)
        forest = trim._model()
        rows = data[8:8 + n * width * 4]
        raws = data[8 + n * width * 4:]
        for i in range(n):
            row = list(struct.unpack_from("<%df" % width, rows, i * width * 4))
            (want,) = struct.unpack_from("<d", raws, i * 8)
            self.assertEqual(forest.raw(row), want, "row %d" % i)

    def test_golden(self):
        """Full output on the sample log pins to the upstream golden file."""
        got = trim.compress(_log_context(), _LOG_QUERY).output
        with open(os.path.join(_TESTDATA, "golden.txt"),
                  encoding="utf-8") as fh:
            want = fh.read()
        self.assertEqual(got, want)

    def test_deterministic_verbatim_in_order(self):
        ctx = _log_context()
        a = trim.compress(ctx, _LOG_QUERY)
        b = trim.compress(ctx, _LOG_QUERY)
        self.assertEqual(a.output, b.output)
        self.assertEqual(a.kept_lines, b.kept_lines)
        self.assertFalse(a.unchanged)
        self.assertTrue(a.output.endswith("\n"))
        lines = ctx.removesuffix("\n").split("\n")
        nxt = kept = 0
        for out in a.output.removesuffix("\n").split("\n"):
            if out.startswith("\u2026 ") and out.endswith(" \u2026"):
                continue
            while nxt < len(lines) and lines[nxt] != out:
                nxt += 1
            self.assertLess(nxt, len(lines),
                            "output line %r is not a later line" % out)
            nxt += 1
            kept += 1
        self.assertEqual(kept, a.kept_lines)
        self.assertEqual(a.kept_lines + a.removed_lines, len(lines))
        self.assertIn("ledger-db timeout", a.output)

    def test_refusals(self):
        ctx = _log_context()
        cases = [
            ("unknown point",
             dict(context=ctx, query=_LOG_QUERY, point="turbo"),
             "unknown operating point"),
            ("small input",
             dict(context="one line\ntwo lines\n", query=_LOG_QUERY),
             "context below the minimum size"),
            ("cut too small",
             dict(context=ctx, query=_LOG_QUERY,
                  policy=trim.Policy(min_cut=1)),
             "cut below the minimum"),
            ("blank query",
             dict(context=ctx, query=" \n\t "),
             "no query"),
            ("over 8 MiB",
             dict(context=ctx * (trim.MAX_CONTEXT_BYTES // len(ctx) + 1),
                  query=_LOG_QUERY),
             "context over 8 MiB"),
            ("many lines",
             dict(context="x\n" * (trim.MAX_CONTEXT_LINES + 1),
                  query=_LOG_QUERY),
             "context over 200,000 lines"),
            ("marker line",
             dict(context=ctx.replace("exit status 1", "\u2026 3 lines \u2026",
                                      1),
                  query=_LOG_QUERY),
             "context has a line that reads as an elision marker"),
        ]
        for name, kwargs, reason in cases:
            with self.subTest(name=name):
                res = trim.compress(**kwargs)
                self.assertTrue(res.unchanged, name)
                self.assertEqual(res.output, kwargs["context"], name)
                self.assertEqual(res.reason, reason, name)

    def test_query_cap(self):
        """Only the query's last 8 KiB count: earlier text changes nothing."""
        ctx = _log_context()
        tail = "é" * trim.MAX_QUERY_BYTES + _LOG_QUERY
        _, a = trim.line_features(ctx, "ledger payment " + tail)
        _, b = trim.line_features(ctx, "checkout timeout " + tail)
        self.assertEqual(a, b)

    def test_final_newline_is_not_a_line(self):
        lines = "a\nb\nc\nd\n".split("\n")
        policy = trim.Policy(floor=1, top_k=1, context=0, headers=0, gap=0)
        for scores, want in [
                ([1, 0, 0, 0, 0], "a\n\u2026 3 lines \u2026\n"),
                ([0, 0, 0, 1, 0], "\u2026 3 lines \u2026\nd\n")]:
            out, kept = policy.apply(lines, scores)
            self.assertEqual(out, want)
            self.assertEqual(kept, 1)

    def test_model_loads(self):
        self.assertEqual(len(trim.FEATURE_NAMES), trim.N_FEATURES)
        forest = trim._model()
        self.assertGreaterEqual(len(forest.roots), 10)
        with open(trim._model_path(), "rb") as fh:
            data = fh.read()
        with self.assertRaises(ValueError):
            trim._parse_model(data[:-1])

    def test_model_info(self):
        info = trim.model_info()
        self.assertIn("712 trees", info)
        self.assertIn("67 features", info)

    def test_points(self):
        for name in ("conservative", "balanced", "aggressive"):
            res = trim.compress(_log_context(), _LOG_QUERY, point=name)
            self.assertFalse(res.unchanged, name)
            self.assertEqual(res.point, name)
            self.assertIn("ledger-db timeout", res.output, name)

    def test_blank_lines_score_zero(self):
        lines, scores = trim.score("alpha beta gamma\n\n" + "filler\n" * 300,
                                   "alpha beta gamma")
        self.assertEqual(scores[1], 0.0)

    def test_compress_keeps_error_line_in_tool_output(self):
        """Realistic shape: a long log-shaped tool result, one line the
        task needs. The model separates repetitive log lines from the
        error line it was asked for."""
        parts = []
        for i in range(300):
            parts.append(
                "2026-10-11T03:%02d:%02dZ INFO checkout request %d served "
                "in %dms" % (i // 60 % 60, i % 60, 1000 + i, 10 + i % 17))
        parts.append(
            "2026-10-11T03:04:31Z ERROR payment client: ledger-db "
            "timeout after 30s (order 1271)")
        parts.append("    at LedgerClient.charge (ledger/client.go:88)")
        ctx = "\n".join(parts) + "\n"
        res = trim.compress(ctx, "why did the payment fail? find the "
                            "ledger error")
        self.assertFalse(res.unchanged)
        self.assertIn("ledger-db timeout", res.output)
        self.assertLess(len(res.output), len(ctx) // 2)


if __name__ == "__main__":
    unittest.main()
