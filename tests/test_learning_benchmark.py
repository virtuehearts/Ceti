"""Held-out learning benchmark: before/after + leakage check.

The fixture lives only in tests/ — no learning code imports it. The
test fails if ANY leak is found (the check is real, not ceremonial),
and requires learning to measurably improve accuracy on questions the
baseline cannot answer.
"""
import json
import os
import unittest

from ceti.learning_benchmark import run_before_after


def _fixture_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "testdata_learning_benchmark.json")


def _cassettes_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "cassettes")


class TestLearningBenchmark(unittest.TestCase):
    def test_before_after_with_leakage_check(self):
        with open(_fixture_path(), encoding="utf-8") as f:
            data = json.load(f)
        benchmark = data["benchmark"]
        learning = data["learning"]
        # 60-80 held-out questions, one learning pair each.
        self.assertGreaterEqual(len(benchmark), 60)
        self.assertLessEqual(len(benchmark), 80)
        self.assertEqual(len(learning), len(benchmark))

        result = run_before_after(benchmark, learning,
                                   seed_dirs=[_cassettes_dir()])
        before = result["before"]
        after = result["after"]
        print(f"\nlearning benchmark: before={before['correct']}/"
              f"{before['total']} ({before['accuracy']:.1%}) "
              f"after={after['correct']}/{after['total']} "
              f"({after['accuracy']:.1%}) delta={result['delta']:+.1%}")
        if result["leaks"]:
            for leak in result["leaks"]:
                print(f"LEAK: {leak}")
        # The leakage check is real: any leak fails the test.
        self.assertEqual(result["leaks"], [],
                         f"{len(result['leaks'])} leak(s) found")
        # The baseline (fake backend, empty brain) must not already
        # know the answers, and learning must measurably improve it.
        self.assertLess(before["accuracy"], 0.10,
                        "baseline already answers the benchmark — "
                        "the set is not held out")
        self.assertGreater(after["accuracy"], before["accuracy"],
                           "learning did not improve accuracy")
        self.assertGreaterEqual(after["accuracy"], 0.85,
                                "after-learning accuracy too low; "
                                "failures: %s" % [
                                    (r["question"], r["response_after"])
                                    for r in result["rows"]
                                    if not r["correct_after"]])


if __name__ == "__main__":
    unittest.main()
