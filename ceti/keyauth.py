"""Simple key authentication on the regula-core identity.

The node already has a secp256k1 identity (data/identity.dat, the
regula_core plug point). Key auth is challenge/response on that key:

  1. The verifier issues a challenge: "auth:<node_id>:<nonce>:<ts>"
  2. The holder signs it with the identity key (SHA256d + DER,
     the same form NodeIdentity.sign uses for every signed post,
     bundle, and nodelist).
  3. The verifier checks the signature against the PUBLIC key.
     The secret never leaves the holder. A signature for one
     challenge does not authenticate any other challenge (the
     nonce + timestamp bind it).

No passwords, no shared secrets, no central server. The public key
is the identity; the signature is the proof.

    python -m ceti.keyauth [--data-dir DIR]
"""
from __future__ import annotations

import hashlib
import os
import secrets
import sys
import time

from . import signing as _signing


def issue_challenge(node_id: str) -> str:
    """A fresh challenge: node + random nonce + unix timestamp."""
    return "auth:%s:%s:%d" % (node_id, secrets.token_hex(8),
                              int(time.time()))


def sign_challenge(identity, challenge: str) -> str:
    """Sign a challenge with the identity key. Returns DER hex."""
    return identity.sign(challenge.encode("utf-8"))


def verify_challenge(public_hex: str, challenge: str,
                     signature_hex: str) -> bool:
    """Verify a challenge signature against a PUBLIC key.

    Returns False on any failure (bad key, bad signature, wrong
    challenge) — never raises, so a verifier can treat False as
    "not authenticated" without a try/except.
    """
    try:
        _signing._crypto()
        from ecdsa import SECP256k1, VerifyingKey
        from ecdsa.util import sigdecode_der
        vk = VerifyingKey.from_string(
            bytes.fromhex(public_hex), curve=SECP256k1,
            hashfunc=hashlib.sha256)
        digest = _signing.sha256d(challenge.encode("utf-8"))
        return vk.verify_digest(
            bytes.fromhex(signature_hex), digest,
            sigdecode=sigdecode_der)
    except Exception:
        return False


def demo(data_dir: str | None = None) -> dict:
    """Full round-trip: issue -> sign -> verify, plus the two
    negative cases (wrong challenge, wrong signature). Returns
    the result dict; main() prints it."""
    ident = _signing.ensure_identity(data_dir)
    challenge = issue_challenge(ident.node_id)
    sig = sign_challenge(ident, challenge)
    ok = verify_challenge(ident.public_hex(), challenge, sig)
    wrong_challenge = verify_challenge(
        ident.public_hex(), challenge + "x", sig)
    wrong_sig = verify_challenge(
        ident.public_hex(), challenge, "00" * 70)
    return {
        "node_id": ident.node_id,
        "fingerprint": ident.fingerprint,
        "challenge": challenge,
        "signature": sig,
        "verified": ok,
        "wrong_challenge_rejected": not wrong_challenge,
        "wrong_signature_rejected": not wrong_sig,
    }


def main() -> None:
    data_dir = None
    if "--data-dir" in sys.argv:
        data_dir = sys.argv[sys.argv.index("--data-dir") + 1]
    try:
        r = demo(data_dir)
    except Exception as e:
        print("keyauth error: %s" % e)
        sys.exit(1)
    print("node: %s (%s)" % (r["node_id"], r["fingerprint"]))
    print("challenge: %s" % r["challenge"])
    print("signature: %s..." % r["signature"][:32])
    print("verified: %s" % r["verified"])
    print("wrong challenge rejected: %s" % r["wrong_challenge_rejected"])
    print("wrong signature rejected: %s" % r["wrong_signature_rejected"])
    print("AUTH OK" if r["verified"] and r["wrong_challenge_rejected"]
          and r["wrong_signature_rejected"] else "AUTH FAIL")


if __name__ == "__main__":
    main()
