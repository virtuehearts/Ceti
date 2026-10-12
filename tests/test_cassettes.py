import os
import tempfile
import unittest

from ceti.aiml import AimlBrain
from ceti.cassettes import CassetteRack, slugify


class TestCassettes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.rack = CassetteRack(os.path.join(self.tmp, "cassettes"))

    def test_slugify(self):
        self.assertEqual(slugify("Dial-Up Modems!"), "dial-up-modems")

    def test_create_and_list(self):
        p = self.rack.create("Dial-Up Modems")
        self.assertTrue(os.path.exists(p))
        self.assertIn("dial-up-modems", self.rack.list())

    def test_load_paths(self):
        self.rack.create("Topic A")
        self.rack.create("Topic B")
        paths = self.rack.load_paths("all")
        self.assertEqual(len(paths), 2)
        self.assertEqual(self.rack.load_paths(["topic-a"]), paths[:1])
        self.assertEqual(self.rack.load_paths(["nope"]), [])

    def test_eject(self):
        self.rack.create("Topic A")
        self.assertTrue(self.rack.eject("topic-a"))
        self.assertNotIn("topic-a", self.rack.list())
        self.assertFalse(self.rack.eject("topic-a"))

    def test_cassette_priority(self):
        # learned > cassette > seed on duplicate patterns
        seed = os.path.join(self.tmp, "seed.aiml")
        with open(seed, "w") as f:
            f.write('<?xml version="1.0"?><aiml version="1.0">'
                    '<category><pattern>WHAT IS X</pattern>'
                    '<template>seed answer</template></category></aiml>')
        cas = self.rack.create("test")
        writer = AimlBrain(learned_path=cas)
        writer.learn("What is X?", "cassette answer")
        b = AimlBrain(seed_path=seed, extra_paths=[cas])
        self.assertEqual(b.respond("what is x"), "cassette answer")
        # learned file beats the cassette
        learned = os.path.join(self.tmp, "learned.aiml")
        b2 = AimlBrain(seed_path=seed, learned_path=learned,
                       extra_paths=[cas])
        b2.learn("What is X?", "learned answer")
        self.assertEqual(b2.respond("what is x"), "learned answer")


if __name__ == "__main__":
    unittest.main()
