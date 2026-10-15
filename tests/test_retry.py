import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ceti.retry import retry


class Flaky:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise ValueError("boom %d" % self.calls)
        return "ok"


class TestRetry(unittest.TestCase):
    def test_succeeds_after_flakes(self):
        f = Flaky(2)
        self.assertEqual(retry(f, attempts=3, delay=0), "ok")
        self.assertEqual(f.calls, 3)

    def test_first_try(self):
        f = Flaky(0)
        self.assertEqual(retry(f, delay=0), "ok")
        self.assertEqual(f.calls, 1)

    def test_raises_last_error(self):
        f = Flaky(99)
        with self.assertRaises(ValueError):
            retry(f, attempts=3, delay=0)
        self.assertEqual(f.calls, 3)

    def test_on_retry_called(self):
        seen = []
        f = Flaky(1)
        retry(f, attempts=2, delay=0,
              on_retry=lambda n, e: seen.append((n, str(e))))
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], 1)

    def test_non_matching_exception_not_retried(self):
        def bad():
            raise KeyError("nope")
        with self.assertRaises(KeyError):
            retry(bad, attempts=3, delay=0, exceptions=(ValueError,))

    def test_bad_attempts(self):
        with self.assertRaises(ValueError):
            retry(lambda: 1, attempts=0)


if __name__ == "__main__":
    unittest.main()
