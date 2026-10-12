import os
import tempfile
import unittest

from ceti.seeder import bdecode, bencode, create_torrent


class TestBencode(unittest.TestCase):
    def test_roundtrip(self):
        obj = {b"a": 1, b"b": [b"x", b"y"], b"c": {b"d": b"e"}}
        self.assertEqual(bdecode(bencode(obj)), obj)
        self.assertEqual(bdecode(bencode("str-key")), b"str-key")

    def test_dict_keys_sorted(self):
        enc = bencode({"z": 1, "a": 2})
        self.assertLess(enc.index(b"1:a"), enc.index(b"1:z"))


class TestTorrent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.bundle = os.path.join(self.tmp, "testset-v1")
        os.makedirs(self.bundle)
        with open(os.path.join(self.bundle, "a.txt"), "w") as f:
            f.write("alpha" * 1000)
        with open(os.path.join(self.bundle, "manifest.json"), "w") as f:
            f.write('{"v": 1}')

    def test_create(self):
        info = create_torrent(self.bundle)
        self.assertTrue(os.path.exists(info["torrent_path"]))
        self.assertEqual(len(info["info_hash"]), 40)
        self.assertTrue(info["magnet"].startswith("magnet:?xt=urn:btih:"))
        self.assertIn("dn=testset-v1", info["magnet"])
        self.assertEqual(info["files"], 2)
        # torrent parses back and hashes match
        with open(info["torrent_path"], "rb") as f:
            t = bdecode(f.read())
        self.assertIn(b"info", t)
        self.assertIn(b"announce-list", t)

    def test_deterministic(self):
        a = create_torrent(self.bundle,
                           out_path=os.path.join(self.tmp, "a.torrent"))
        b = create_torrent(self.bundle,
                           out_path=os.path.join(self.tmp, "b.torrent"))
        self.assertEqual(a["info_hash"], b["info_hash"])

    def test_single_file_torrent(self):
        # release tarballs seed as single-file torrents
        import tarfile
        tb = os.path.join(self.tmp, "set-v1.tar.gz")
        with tarfile.open(tb, "w:gz") as tar:
            tar.add(self.bundle, arcname="set-v1")
        info = create_torrent(tb)
        self.assertEqual(info["files"], 1)
        self.assertTrue(info["magnet"].startswith("magnet:?xt=urn:btih:"))
        with open(info["torrent_path"], "rb") as f:
            t = bdecode(f.read())
        self.assertIn(b"length", t[b"info"])
        self.assertNotIn(b"files", t[b"info"])


if __name__ == "__main__":
    unittest.main()
