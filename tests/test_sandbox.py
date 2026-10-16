"""Sandbox isolation tests. Each run is a real namespace sandbox."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ceti.sandbox import (RunResult, SandboxError, SandboxPolicy,
                          SandboxRunner, run_python)


def _available():
    try:
        r = run_python("print('ok')", timeout_s=5)
        return r.exit_code == 0 and r.stdout.strip() == "ok"
    except SandboxError:
        return False


AVAILABLE = _available()


@unittest.skipUnless(AVAILABLE, "namespace sandbox unavailable")
class TestSandbox(unittest.TestCase):
    def test_hello(self):
        r = run_python("print('hello')")
        self.assertEqual(r.exit_code, 0)
        self.assertEqual(r.stdout.strip(), "hello")

    def test_stdin_and_files(self):
        r = SandboxRunner().run(
            ["python3", "/work/main.py"],
            SandboxPolicy(cpu_seconds=6),
            stdin="ignored",
            files={"main.py": "print(open('/work/data.txt').read())",
                   "data.txt": "staged"})
        self.assertEqual(r.stdout.strip(), "staged")

    def test_no_host_write(self):
        # Writes outside /work and /tmp fail or do not touch the host.
        # Host read-only bind mounts (e.g. /usr) fail loudly with EROFS.
        r = run_python("open('/usr/host_proof', 'w').write('x')")
        self.assertNotEqual(r.exit_code, 0)

    def test_ro_bind(self):
        r = run_python("open('/usr/evil', 'w').write('x')")
        self.assertNotEqual(r.exit_code, 0)

    def test_no_network(self):
        r = run_python(
            "import socket; "
            "socket.create_connection(('8.8.8.8', 53), timeout=3)",
            timeout_s=8)
        self.assertNotEqual(r.exit_code, 0)

    def test_memory_cap(self):
        r = run_python("x = [0] * 10**9", timeout_s=8)
        self.assertNotEqual(r.exit_code, 0)

    def test_timeout(self):
        r = run_python("import time; time.sleep(30)", timeout_s=3)
        self.assertTrue(r.timed_out)

    def test_pid_namespace(self):
        r = run_python("import os; print(os.getpid())")
        self.assertEqual(r.stdout.strip(), "1")

    def test_output_truncation(self):
        r = run_python("print('x' * 200000)",
                       policy=SandboxPolicy(max_output_kb=1))
        self.assertTrue(r.truncated)
        self.assertLessEqual(len(r.stdout), 1024)

    def test_bad_command(self):
        r = SandboxRunner().run(["/nonexistent/cmd"],
                                SandboxPolicy(cpu_seconds=5))
        self.assertNotEqual(r.exit_code, 0)

    def test_bg_poll_then_done(self):
        from ceti.sandbox import run_python_bg
        job = run_python_bg(
            "import time; print('starting'); time.sleep(2); "
            "print('finished')",
            timeout_s=10)
        try:
            first = job.poll()
            self.assertFalse(first.done)
            self.assertIn("starting", first.stdout)
            self.assertFalse(job.done)
            final = job.wait(timeout=15)
            self.assertTrue(final.done)
            self.assertTrue(job.done)
            self.assertEqual(final.exit_code, 0)
            self.assertIn("finished", final.stdout)
            # poll after done returns the cached final result
            self.assertTrue(job.poll().done)
        finally:
            job.kill()

    def test_bg_kill(self):
        from ceti.sandbox import run_python_bg
        job = run_python_bg("import time; time.sleep(60); print('nope')",
                            timeout_s=60)
        try:
            job.kill()
            final = job.poll()
            self.assertTrue(final.done)
            self.assertNotIn("nope", final.stdout)
        finally:
            job.kill()

    def test_bg_timeout(self):
        from ceti.sandbox import run_python_bg
        job = run_python_bg("import time; time.sleep(30)",
                            timeout_s=2)
        try:
            final = job.wait(timeout=15)
            self.assertTrue(final.done)
            self.assertTrue(final.timed_out)
        finally:
            job.kill()


if __name__ == "__main__":
    unittest.main()
