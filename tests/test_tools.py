import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ceti import tools
from ceti.tools import ToolError


class TestTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ceti-work-")
        self._old = os.environ.get("CETI_WORK_ROOT")
        os.environ["CETI_WORK_ROOT"] = self.tmp

    def tearDown(self):
        if self._old is None:
            os.environ.pop("CETI_WORK_ROOT", None)
        else:
            os.environ["CETI_WORK_ROOT"] = self._old

    def test_write_read_edit_list(self):
        self.assertIn("wrote", tools.call("write_file", path="a/b.txt",
                                          content="hello"))
        self.assertEqual(tools.call("read_file", path="a/b.txt"), "hello")
        self.assertIn("edited", tools.call(
            "edit_file", path="a/b.txt", old="hello", new="world"))
        self.assertEqual(tools.call("read_file", path="a/b.txt"), "world")
        listing = tools.call("list_dir", path="a")
        self.assertIn("b.txt", listing)

    def test_edit_missing_pattern(self):
        tools.call("write_file", path="f.txt", content="abc")
        with self.assertRaises(ToolError):
            tools.call("edit_file", path="f.txt", old="zzz", new="q")

    def test_jail_dotdot(self):
        with self.assertRaises(ToolError):
            tools.call("write_file", path="../../evil.txt",
                       content="x")
        with self.assertRaises(ToolError):
            tools.call("read_file", path="a/../../../etc/passwd")

    def test_jail_symlink(self):
        os.symlink("/etc", os.path.join(self.tmp, "link"))
        with self.assertRaises(ToolError):
            tools.call("read_file", path="link/passwd")

    def test_unknown_tool(self):
        with self.assertRaises(ToolError):
            tools.call("nope")

    def test_manifest(self):
        m = tools.manifest()
        names = {t["name"] for t in m}
        self.assertTrue({"run_python", "read_file", "write_file",
                         "edit_file", "list_dir", "trim"} <= names)
        for t in m:
            self.assertIn(t["side_effects"], ("none", "read", "write"))
            self.assertIsInstance(t["sandboxed"], bool)
        rp = next(t for t in m if t["name"] == "run_python")
        self.assertTrue(rp["sandboxed"])

    def test_run_python_tool(self):
        out = tools.call("run_python", code="print(6 * 7)")
        self.assertIn("42", out)
        self.assertIn("[exit 0]", out)

    def test_run_python_isolated(self):
        out = tools.call(
            "run_python",
            code="import socket; "
                 "socket.create_connection(('8.8.8.8', 53), timeout=3)")
        self.assertNotIn("[exit 0]", out)

    def test_tool_execute_result(self):
        t = tools.all_tools()["write_file"]
        ok = t.execute(path="r.txt", content="x")
        self.assertTrue(ok.success)
        self.assertIn("wrote", ok.content)
        bad = t.execute(path="../../evil.txt", content="x")
        self.assertFalse(bad.success)
        self.assertIn("escapes", bad.error)

    def test_execute_bad_args_no_raise(self):
        t = tools.all_tools()["read_file"]
        r = t.execute()  # missing path
        self.assertFalse(r.success)

    def test_execute_unknown_tool(self):
        r = tools.execute("nope")
        self.assertFalse(r.success)
        self.assertIn("no such tool", r.error)

    def test_openai_schemas(self):
        for s in tools.openai_schemas():
            self.assertEqual(s["type"], "function")
            fn = s["function"]
            self.assertTrue(fn["name"])
            self.assertTrue(fn["description"])
            params = fn["parameters"]
            self.assertEqual(params["type"], "object")
            self.assertIn("properties", params)
            self.assertIn("required", params)

    def test_bg_job_roundtrip(self):
        started = tools.call(
            "run_python_bg",
            code="import time; print('bg-hi'); time.sleep(1); "
                 "print('bg-bye')",
            timeout_s=15)
        job_id = started.split()[1]
        import time as _t
        final = None
        for _ in range(60):
            out = tools.call("job_poll", job_id=job_id)
            if out.startswith(job_id + ": done"):
                final = out
                break
            _t.sleep(0.2)
        self.assertIsNotNone(final, "job never finished: %s" % out)
        self.assertIn("bg-bye", final)
        # finished jobs are reaped from the table
        r = tools.execute("job_poll", job_id=job_id)
        self.assertFalse(r.success)

    def test_job_kill_unknown(self):
        r = tools.execute("job_kill", job_id="job-99999")
        self.assertFalse(r.success)

    def test_trim_tool_compresses_log(self):
        lines = []
        for i in range(300):
            lines.append(
                "2026-10-11T03:%02d:%02dZ INFO checkout request %d served "
                "in %dms" % (i // 60 % 60, i % 60, 1000 + i, 10 + i % 17))
        lines.append("2026-10-11T03:04:31Z ERROR payment client: ledger-db "
                     "timeout after 30s (order 1271)")
        lines.append("    at LedgerClient.charge (ledger/client.go:88)")
        out = tools.call("trim", text="\n".join(lines) + "\n",
                         query="why did the payment fail? find the "
                               "ledger error")
        self.assertIn("ledger-db timeout", out)
        self.assertIn("kept 6 of", out)

    def test_trim_tool_fails_closed_on_short_input(self):
        out = tools.call("trim", text="short\n", query="anything")
        self.assertIn("unchanged", out)
        self.assertIn("below the minimum size", out)

    def test_trim_tool_unknown_point(self):
        r = tools.execute("trim", text="x\n" * 3000, query="q",
                          point="turbo")
        self.assertTrue(r.success)  # trim fails closed, never raises
        self.assertIn("unknown operating point", r.content)


if __name__ == "__main__":
    unittest.main()
