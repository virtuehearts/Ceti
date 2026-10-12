import os
import unittest

from ceti import signing

try:
    signing._crypto()
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False


@unittest.skipUnless(HAS_CRYPTO, "ecdsa package not available")
class TestSigning(unittest.TestCase):
    def test_roundtrip(self):
        ident = signing.new_test_identity()
        payload = b'{"dataset": "test", "version": 1}'
        sig = signing.sign_bytes(payload, identity=ident)
        self.assertEqual(sig["node_id"], ident.node_id)
        self.assertEqual(sig["scheme"], signing.SCHEME)
        self.assertTrue(signing.verify_bytes(
            payload, sig["public_hex"], sig["signature"]))

    def test_tamper_detected(self):
        ident = signing.new_test_identity()
        sig = signing.sign_bytes(b"original", identity=ident)
        self.assertFalse(signing.verify_bytes(
            b"tampered", sig["public_hex"], sig["signature"]))

    def test_wrong_key_rejected(self):
        a = signing.new_test_identity()
        b = signing.new_test_identity()
        sig = signing.sign_bytes(b"data", identity=a)
        self.assertFalse(signing.verify_bytes(
            b"data", b.public_hex(), sig["signature"]))

    def test_key_never_leaves(self):
        # sign_bytes returns no secret material
        ident = signing.new_test_identity()
        sig = signing.sign_bytes(b"x", identity=ident)
        blob = str(sig)
        self.assertNotIn(ident._sk.to_string().hex(), blob)

    def test_identity_format(self):
        ident = signing.new_test_identity()
        self.assertTrue(ident.node_id.startswith("jn1"))
        self.assertEqual(len(ident.node_id), 3 + 40)  # jn1 + hash160 hex
        self.assertRegex(ident.fingerprint, r"^CETI-[0-9A-F]{4}-"
                                            r"[0-9A-F]{4}-[0-9A-F]{4}$")
        self.assertEqual(len(bytes.fromhex(ident.public_hex())), 65)

    def test_persistent_identity_roundtrip(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        a = signing.ensure_identity(tmp, quiet=True)
        b = signing.ensure_identity(tmp, quiet=True)
        self.assertEqual(a.fingerprint, b.fingerprint)  # same key reloaded
        self.assertEqual(a.secret_hex(), b.secret_hex())
        st = os.stat(os.path.join(tmp, signing.IDENTITY_FILE))
        self.assertEqual(oct(st.st_mode & 0o777), "0o600")

    def test_identity_file_format(self):
        import json
        import tempfile
        tmp = tempfile.mkdtemp()
        ident = signing.ensure_identity(tmp, quiet=True)
        with open(os.path.join(tmp, signing.IDENTITY_FILE)) as f:
            w = json.load(f)
        # the portable core: raw secret; the rest re-derives from it
        self.assertEqual(w["secret"], ident.secret_hex())
        self.assertEqual(w["public"], ident.public_hex())
        self.assertEqual(w["node_id"], ident.node_id)
        self.assertEqual(w["fingerprint"], ident.fingerprint)
        self.assertEqual(w["scheme"], signing.SCHEME)
        self.assertIn("created", w)
        # round-trip through the dict alone
        c = signing.NodeIdentity.from_identity_dict(w)
        self.assertEqual(c.fingerprint, ident.fingerprint)

    def test_legacy_key_upgraded(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        legacy = signing.new_test_identity()
        with open(os.path.join(tmp, signing.LEGACY_IDENTITY_FILE),
                  "w") as f:
            f.write(legacy.secret_hex() + "\n")
        ident = signing.ensure_identity(tmp, quiet=True)
        # same key, now in identity.dat format
        self.assertEqual(ident.fingerprint, legacy.fingerprint)
        self.assertTrue(os.path.exists(
            os.path.join(tmp, signing.IDENTITY_FILE)))

    def test_legacy_json_upgraded(self):
        import json
        import tempfile
        tmp = tempfile.mkdtemp()
        legacy = signing.new_test_identity()
        with open(os.path.join(tmp, signing.LEGACY_JSON_FILE), "w") as f:
            json.dump(legacy.identity_dict(), f)
        ident = signing.ensure_identity(tmp, quiet=True)
        self.assertEqual(ident.fingerprint, legacy.fingerprint)
        self.assertTrue(os.path.exists(
            os.path.join(tmp, signing.IDENTITY_FILE)))

    def test_custom_fingerprint_prefix(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        os.environ["CETI_FINGERPRINT_PREFIX"] = "JERO"
        try:
            ident = signing.ensure_identity(tmp, quiet=True)
            self.assertRegex(ident.fingerprint, r"^JERO-[0-9A-F]{4}-")
            # prefix persists in identity.dat, stable across env changes
            with open(os.path.join(tmp, signing.IDENTITY_FILE)) as f:
                import json
                self.assertEqual(json.load(f)["prefix"], "JERO")
        finally:
            del os.environ["CETI_FINGERPRINT_PREFIX"]
        reloaded = signing.ensure_identity(tmp, quiet=True)
        self.assertEqual(reloaded.fingerprint, ident.fingerprint)
        self.assertRegex(reloaded.fingerprint, r"^JERO-")

    def test_identity_env_override(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        custom = os.path.join(tmp, "custom-identity.dat")
        os.environ["CETI_IDENTITY"] = custom
        try:
            ident = signing.ensure_identity(tmp, quiet=True)
            self.assertTrue(os.path.exists(custom))
            self.assertEqual(signing.identity_path(tmp), custom)
        finally:
            del os.environ["CETI_IDENTITY"]
        self.assertRegex(ident.fingerprint, r"^CETI-")


if __name__ == "__main__":
    unittest.main()
