"""Integration: lobes on a bus — modules -> graph -> composer.

The Brain wires the pieces together: module files load at boot and
build the graph (temporal lobe); the graph serves structured lookups;
the composer (frontal lobe) composes explain answers from graph facts
(+ a library hit); learning goes through one gate (learn_fact) that
checks the graph (modules + learned facts) before persisting.
"""
import os
import tempfile
import unittest

from ceti.backends import ChatBackend
from ceti.brain import Brain
from ceti.config import BrainConfig

FRANCE_MODULE = """<?xml version="1.0" encoding="UTF-8"?>
<module domain="geography" name="france" version="1"
        source="test" confidence="1.0">
  <fact subject="France" predicate="capital" object="Paris"/>
  <fact subject="France" predicate="currency" object="Euro"/>
</module>
"""


class FakeLocal(ChatBackend):
    name = "fake-local"

    def complete(self, system, user, max_tokens=512, temperature=0.3):
        return "n/a"


def make_brain_with_module(module_xml=FRANCE_MODULE):
    tmp = tempfile.mkdtemp()
    cfg = BrainConfig(data_dir=tmp, adjudicate_with_llm=False,
                      memory_backend="sqlite")
    with open(os.path.join(tmp, "seed.aiml"), "w") as f:
        f.write('<?xml version="1.0"?><aiml version="1.0">'
                '<category><pattern>HELLO</pattern>'
                '<template>Hi.</template></category></aiml>')
    os.makedirs(os.path.join(tmp, "cassettes", "geography"),
                exist_ok=True)
    with open(os.path.join(tmp, "cassettes", "geography",
                           "france.aiml"), "w") as f:
        f.write(module_xml)
    return Brain(config=cfg, local_backend=FakeLocal(),
                 cloud_backends=[])


class TestLobes(unittest.TestCase):
    def test_module_fact_served_from_graph(self):
        b = make_brain_with_module()
        ans = b.ask("what is the capital of France", learn=False)
        self.assertIn("Paris", ans.text)
        self.assertEqual(ans.trace.get("memory_source"), "graph")
        b.close()

    def test_learn_writes_to_graph(self):
        b = make_brain_with_module()
        ok = b._learn("when was the commodore 64 introduced",
                      "The Commodore 64 was introduced in January 1982.",
                      0.9, source="local")
        self.assertTrue(ok)
        self.assertEqual(
            b._graph.lookup("Commodore 64", "introduced"),
            "January 1982")
        b.close()

    def test_graph_gate_blocks_conflict_with_module(self):
        # The module (confidence 1.0) outranks a learned 0.9 claim.
        b = make_brain_with_module()
        ok = b._learn("what is the capital of France",
                      "The capital of France is London.",
                      0.9, source="local")
        self.assertFalse(ok)
        self.assertIn("Paris", b._learn_block_reason)
        self.assertEqual(b._graph.lookup("France", "capital"), "Paris")
        b.close()

    def test_composer_explains_from_graph(self):
        b = make_brain_with_module()
        ans = b.ask("tell me about France", learn=False)
        self.assertIn("Paris", ans.text)
        self.assertEqual(ans.trace.get("memory_source"), "composer")
        b.close()

    def test_learned_fact_reports_facts_origin(self):
        # A fact learned via _learn is served from the graph but
        # reports origin "facts" (it lives in the learned store).
        b = make_brain_with_module()
        b._learn("when was the commodore 64 introduced",
                 "The Commodore 64 was introduced in January 1982.",
                 0.9, source="local")
        ans = b.ask("what year did the commodore 64 come out",
                    learn=False)
        self.assertEqual(
            ans.text,
            "The Commodore 64 was introduced in January 1982.")
        self.assertEqual(ans.trace.get("memory_source"), "facts")
        b.close()


if __name__ == "__main__":
    unittest.main()
