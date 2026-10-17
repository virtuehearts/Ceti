"""Key auth: challenge/response on the regula-core identity."""
import tempfile
import unittest

from ceti import keyauth
from ceti.signing import NodeIdentity


class TestKeyAuth(unittest.TestCase):
    def test_roundtrip(self):
        ident = NodeIdentity(bytes([7]) * 32)
        chal = keyauth.issue_challenge(ident.node_id)
        sig = keyauth.sign_challenge(ident, chal)
        self.assertTrue(keyauth.verify_challenge(
            ident.public_hex(), chal, sig))

    def test_wrong_challenge_rejected(self):
        ident = NodeIdentity(bytes([7]) * 32)
        chal = keyauth.issue_challenge(ident.node_id)
        sig = keyauth.sign_challenge(ident, chal)
        self.assertFalse(keyauth.verify_challenge(
            ident.public_hex(), chal + "x", sig))

    def test_wrong_signature_rejected(self):
        ident = NodeIdentity(bytes([7]) * 32)
        chal = keyauth.issue_challenge(ident.node_id)
        self.assertFalse(keyauth.verify_challenge(
            ident.public_hex(), chal, "00" * 70))

    def test_wrong_key_rejected(self):
        ident = NodeIdentity(bytes([7]) * 32)
        other = NodeIdentity(bytes([9]) * 32)
        chal = keyauth.issue_challenge(ident.node_id)
        sig = keyauth.sign_challenge(ident, chal)
        self.assertFalse(keyauth.verify_challenge(
            other.public_hex(), chal, sig))

    def test_bad_inputs_never_raise(self):
        self.assertFalse(keyauth.verify_challenge(
            "not-hex", "auth:x", "00" * 70))
        self.assertFalse(keyauth.verify_challenge(
            "00" * 65, "auth:x", "not-hex"))

    def test_demo(self):
        tmp = tempfile.mkdtemp()
        r = keyauth.demo(tmp)
        self.assertTrue(r["verified"])
        self.assertTrue(r["wrong_challenge_rejected"])
        self.assertTrue(r["wrong_signature_rejected"])
        self.assertTrue(r["challenge"].startswith("auth:"))


if __name__ == "__main__":
    unittest.main()
