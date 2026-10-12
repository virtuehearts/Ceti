import json
import os
import tempfile
import unittest

from ceti import signing
from ceti.datasets import (DatasetStore, GENESIS_HASH, canonical_bytes,
                            merkle_root, sha256_file)

try:
    signing._crypto()
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False


def write_files(d):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "a.txt"), "w") as f:
        f.write("alpha")
    with open(os.path.join(d, "b.txt"), "w") as f:
        f.write("beta")


@unittest.skipUnless(HAS_CRYPTO, "ecdsa package not available")
class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "src")
        write_files(self.src)
        self.store = DatasetStore(os.path.join(self.tmp, "datasets"))
        self.ident = signing.new_test_identity()

    def _files(self):
        return [os.path.join(self.src, "a.txt"),
                os.path.join(self.src, "b.txt")]

    def test_release_and_verify(self):
        m, bundle, tarball = self.store.release("testset", self._files(),
                                       identity=self.ident)
        self.assertEqual(m["version"], 1)
        self.assertEqual(m["prev_hash"], GENESIS_HASH)
        self.assertEqual(len(m["files"]), 2)
        ok, reasons = self.store.verify(bundle)
        self.assertTrue(ok, reasons)

    def test_chain(self):
        m1, b1, _t1 = self.store.release("testset", self._files(),
                                         identity=self.ident)
        with open(os.path.join(self.src, "a.txt"), "w") as f:
            f.write("alpha2")
        m2, b2, _t2 = self.store.release("testset", self._files(),
                                         identity=self.ident)
        self.assertEqual(m2["version"], 2)
        with open(os.path.join(b1, "manifest.json")) as f:
            self.assertEqual(m2["prev_hash"],
                             sha256_file(os.path.join(b1, "manifest.json")))
        ok, _ = self.store.verify(b2)
        self.assertTrue(ok)

    def test_tamper_detected(self):
        m, bundle, tarball = self.store.release("testset", self._files(),
                                       identity=self.ident)
        with open(os.path.join(bundle, "a.txt"), "w") as f:
            f.write("EVIL")
        ok, reasons = self.store.verify(bundle)
        self.assertFalse(ok)
        self.assertTrue(any("mismatch" in r for r in reasons))

    def test_forged_signature_rejected(self):
        # attacker keeps the victim's public key but swaps in a
        # signature made by a different key -> must fail
        m, bundle, tarball = self.store.release("testset", self._files(),
                                       identity=self.ident)
        other = signing.new_test_identity()
        forged = signing.sign_bytes(canonical_bytes(m),
                                    identity=other)["signature"]
        m["signature"]["signature"] = forged
        mp = os.path.join(bundle, "manifest.json")
        with open(mp, "w") as f:
            json.dump(m, f, indent=2, sort_keys=True)
        ok, reasons = self.store.verify(bundle)
        self.assertFalse(ok)
        self.assertIn("signature invalid", reasons)

    def test_release_packs_tarball(self):
        m, bundle, tarball = self.store.release(
            "testset", self._files(), identity=self.ident)
        self.assertTrue(os.path.exists(tarball))
        self.assertTrue(tarball.endswith(".tar.gz"))
        # tarball verifies exactly as a peer would see it
        ok, reasons = self.store.verify_tarball(tarball)
        self.assertTrue(ok, reasons)

    def test_tampered_tarball_rejected(self):
        m, bundle, tarball = self.store.release(
            "testset", self._files(), identity=self.ident)
        # corrupt bytes inside the deflate stream -> CRC failure
        with open(tarball, "rb") as f:
            data = bytearray(f.read())
        for i in range(len(data) // 3, len(data) // 3 + 20):
            data[i] ^= 0xFF
        with open(tarball, "wb") as f:
            f.write(bytes(data))
        ok, reasons = self.store.verify_tarball(tarball)
        self.assertFalse(ok)
        self.assertTrue(reasons)

    def test_merkle_root_order_independent(self):
        h1 = [sha256_file(os.path.join(self.src, "a.txt")),
              sha256_file(os.path.join(self.src, "b.txt"))]
        self.assertEqual(merkle_root(h1), merkle_root(list(reversed(h1))))


if __name__ == "__main__":
    unittest.main()
