import json
import os
import tempfile
import unittest

from ceti import board as board_mod
from ceti import peers as peers_mod
from ceti import share as share_mod
from ceti import signing
from ceti.datasets import DatasetStore

try:
    signing._crypto()
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False


@unittest.skipUnless(HAS_CRYPTO, "ecdsa not installed")
class TestPeers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.book = peers_mod.PeerBook(self.tmp)
        self.alice = signing.generate_identity()
        self.bob = signing.generate_identity()

    def test_add_derives_from_key(self):
        p = self.book.add(self.bob.public_hex(), note="bob")
        self.assertEqual(p["node_id"], self.bob.node_id)
        self.assertEqual(p["fingerprint"], self.bob.fingerprint)
        self.assertEqual(p["note"], "bob")
        self.assertEqual(self.book.get(self.bob.fingerprint)["node_id"],
                         self.bob.node_id)

    def test_add_is_idempotent(self):
        self.book.add(self.bob.public_hex(), note="first")
        p = self.book.add(self.bob.public_hex(), note="second")
        self.assertEqual(p["note"], "second")
        self.assertEqual(len(self.book.list()), 1)

    def test_card_round_trip(self):
        card = peers_mod.identity_card(self.bob)
        self.assertNotIn("secret", json.dumps(card))
        p = self.book.add_card(card, note="via card")
        self.assertEqual(p["fingerprint"], self.bob.fingerprint)

    def test_forged_card_rejected(self):
        card = peers_mod.identity_card(self.bob)
        card["fingerprint"] = "CETI-DEAD-BEEF-CAFE"
        with self.assertRaises(ValueError):
            peers_mod.card_to_peer(card)

    def test_bad_key_rejected(self):
        with self.assertRaises(ValueError):
            self.book.add("deadbeef")


@unittest.skipUnless(HAS_CRYPTO, "ecdsa not installed")
class TestBoard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.bd = board_mod.Board(self.tmp)
        self.alice = signing.generate_identity()
        self.bob = signing.generate_identity()

    def test_post_and_read(self):
        post = self.bd.post(self.alice, "hello", "first post")
        valid, rejected = self.bd.list()
        self.assertEqual(len(valid), 1)
        self.assertEqual(rejected, [])
        self.assertEqual(self.bd.read(1)["subject"], "hello")
        self.assertEqual(self.bd.read(2), None)
        self.assertEqual(post["author"]["fingerprint"],
                         self.alice.fingerprint)

    def test_tampered_post_rejected_not_shown(self):
        self.bd.post(self.alice, "hello", "first post")
        valid, _ = self.bd.list()
        pid = board_mod.post_id(valid[0])
        path = os.path.join(self.tmp, "board", "posts", pid + ".json")
        with open(path) as f:
            tampered = json.load(f)
        tampered["body"] = "forged body"
        with open(path, "w") as f:
            json.dump(tampered, f)
        valid, rejected = self.bd.list()
        self.assertEqual(valid, [])
        self.assertEqual(len(rejected), 1)

    def test_import_from_another_node(self):
        post = self.bd.post(self.alice, "from alice", "hi bob")
        pid = board_mod.post_id(post)
        src = os.path.join(self.tmp, "board", "posts", pid + ".json")
        other = board_mod.Board(tempfile.mkdtemp())
        imported = other.import_post(src)
        self.assertEqual(imported["subject"], "from alice")
        valid, _ = other.list()
        self.assertEqual(len(valid), 1)

    def test_import_tampered_refused(self):
        post = self.bd.post(self.alice, "hello", "body")
        pid = board_mod.post_id(post)
        src = os.path.join(self.tmp, "board", "posts", pid + ".json")
        with open(src) as f:
            bad = json.load(f)
        bad["author"]["fingerprint"] = "CETI-DEAD-BEEF-CAFE"
        badpath = src + ".bad"
        with open(badpath, "w") as f:
            json.dump(bad, f)
        other = board_mod.Board(tempfile.mkdtemp())
        with self.assertRaises(ValueError):
            other.import_post(badpath)
        valid, _ = other.list()
        self.assertEqual(valid, [])

    def test_post_id_is_content_address(self):
        p1 = self.bd.post(self.alice, "same", "same body")
        # same content from the same key at the same second collides
        # only if created matches; ids differ across distinct posts
        p2 = self.bd.post(self.bob, "other", "other body")
        self.assertNotEqual(board_mod.post_id(p1),
                            board_mod.post_id(p2))


@unittest.skipUnless(HAS_CRYPTO, "ecdsa not installed")
class TestShareFetch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = DatasetStore(os.path.join(self.tmp, "datasets"))
        self.alice = signing.generate_identity()
        from ceti.aiml import AimlBrain
        from ceti.vectormem import VectorMemory

        def fake_embed(texts):
            import zlib
            vecs = []
            for t in texts:
                v = [0.0] * 64
                for w in t.lower().split():
                    v[zlib.crc32(w.encode()) % 64] += 1.0
                n = sum(x * x for x in v) ** 0.5 or 1.0
                vecs.append([x / n for x in v])
            return vecs

        self.mem = VectorMemory(os.path.join(self.tmp, "local.db"),
                                embed_fn=fake_embed)
        self.aiml = AimlBrain(
            learned_path=os.path.join(self.tmp, "learned.aiml"))

    def test_share_produces_verifiable_tarball(self):
        src = os.path.join(self.tmp, "notes.txt")
        with open(src, "w") as f:
            f.write("shared knowledge\n")
        manifest, _bundle, tarball = share_mod.share(
            self.store, [src], identity=self.alice)
        self.assertTrue(os.path.exists(tarball))
        ok, reasons = self.store.verify_tarball(tarball)
        self.assertTrue(ok, reasons)
        self.assertEqual(manifest["signature"]["fingerprint"],
                         self.alice.fingerprint)

    def test_fetch_merges_peer_cassette(self):
        aiml_src = os.path.join(self.tmp, "peer.aiml")
        with open(aiml_src, "w") as f:
            f.write('<?xml version="1.0"?>\n<aiml version="1.0">\n'
                    "<category><pattern>WHAT IS GOPHER</pattern>"
                    "<template>A menu-driven protocol.</template>"
                    "</category>\n</aiml>\n")
        _m, _b, tarball = share_mod.share(
            self.store, [aiml_src], identity=self.alice)
        rep = share_mod.fetch(self.store, self.mem, self.aiml, tarball)
        self.assertEqual(rep["aiml_added"], 1)
        self.assertEqual(rep["peer"], self.alice.fingerprint)

    def test_fetch_tampered_tarball_refused(self):
        src = os.path.join(self.tmp, "notes.txt")
        with open(src, "w") as f:
            f.write("shared knowledge\n")
        _m, _b, tarball = share_mod.share(
            self.store, [src], identity=self.alice)
        # flip a byte mid-file: trailing garbage is ignored by gzip,
        # a mid-file flip breaks the stream or the manifest hash
        with open(tarball, "r+b") as f:
            f.seek(100)
            byte = f.read(1)
            f.seek(100)
            f.write(bytes([byte[0] ^ 0xFF]))
        with self.assertRaises(ValueError):
            share_mod.fetch(self.store, self.mem, self.aiml, tarball)


class TestGopherMenu(unittest.TestCase):
    def test_menu_lists_all_options(self):
        from ceti.gopher import menu_text
        text = menu_text()
        for line in ("1. browse nodes", "2. read the board",
                     "3. shared cassettes", "4. shared files",
                     "5. post to the board", "6. chat room",
                     "7. mail", "0. back"):
            self.assertIn(line, text)


if __name__ == "__main__":
    unittest.main()
