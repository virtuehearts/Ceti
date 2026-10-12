"""signing — regula_core-style dataset signing, self-contained.

Sets are signed BEFORE seeding, exactly like blocks:
  digest    = SHA256d(canonical manifest bytes)
  signature = deterministic ECDSA (RFC 6979) on secp256k1 over the digest,
              DER-encoded (Bitcoin 0.1 style)

Identity: a secp256k1 keypair whose public half derives the familiar
  node_id     = "jn1" + HEX(HASH160(uncompressed pubkey))
  fingerprint = "JERO-XXXX-XXXX-XXXX"  (first 12 hex of the HASH160)

The identity lives in <data_dir>/identity.dat (mode 0600): the
master key plus derived identifiers plus creation time. It is
generated on first run (see ensure_identity) and overridable with
CETI_IDENTITY. The master key never leaves this file — only the
public key is ever shared, hashed into the node_id/fingerprint, and
signatures prove possession without revealing it. The raw 32-byte
secret is the portable part: anything that speaks secp256k1 —
including regula_core later — can import it. Verification needs
only the public key, so any peer can check a set without trusting the
seeder. Signatures are wire-compatible with sets signed by the Jero
node stack: same digest, same curve, same DER encoding.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from datetime import datetime, timezone

SCHEME = "ECDSA-SHA256d-DER-secp256k1"
IDENTITY_ENV = "CETI_IDENTITY"
IDENTITY_FILE = "identity.dat"
LEGACY_IDENTITY_FILE = "identity.key"  # pre-identity.dat secret-hex format
LEGACY_JSON_FILE = "identity.json"  # brief naming detour, 2026-10-12
# Fingerprint prefix: the human-readable network tag. CETI nodes say
# CETI-; set CETI_FINGERPRINT_PREFIX=JERO to mint Jero-network IDs.
FINGERPRINT_ENV = "CETI_FINGERPRINT_PREFIX"
DEFAULT_FINGERPRINT_PREFIX = "CETI"


def _crypto():
    """Probe/loader for the ecdsa package. Kept for the test skip gate."""
    try:
        import ecdsa  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "the 'ecdsa' package is required for dataset signing: "
            "pip install -r requirements.txt") from e
    return ecdsa


def sha256d(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def hash160(data: bytes) -> bytes:
    try:
        return hashlib.new("ripemd160", data).digest()
    except (ValueError, TypeError) as e:
        raise RuntimeError(
            "this Python lacks RIPEMD160 (needs OpenSSL support); "
            "identity fingerprints cannot be derived") from e


def fingerprint_prefix() -> str:
    """Network tag for fingerprints. Env-overridable, CETI by default."""
    return os.environ.get(FINGERPRINT_ENV,
                          DEFAULT_FINGERPRINT_PREFIX).upper()[:12]


def fingerprint_from_hash160(h160: bytes,
                             prefix: str | None = None) -> str:
    hx = h160.hex().upper()
    p = (prefix or fingerprint_prefix()).upper()[:12]
    return "%s-%s-%s-%s" % (p, hx[0:4], hx[4:8], hx[8:12])


class NodeIdentity:
    """A secp256k1 identity. The secret never leaves this object."""

    def __init__(self, secret: bytes, created: str | None = None,
                 prefix: str | None = None):
        _crypto()
        from ecdsa import SECP256k1, SigningKey
        value = int.from_bytes(secret, "big")
        if not 0 < value < SECP256k1.order:
            raise ValueError("secret outside curve order")
        self._sk = SigningKey.from_string(
            secret, curve=SECP256k1, hashfunc=hashlib.sha256)
        # 65-byte 0x04||x||y: the exact form the Jero node stack
        # hashes for node_id/fingerprint derivation (ecdsa>=0.19).
        self.public_bytes = self._sk.verifying_key.to_string("uncompressed")
        h160 = hash160(self.public_bytes)
        self.node_id = "jn1" + h160.hex()
        self.prefix = (prefix or fingerprint_prefix()).upper()[:12]
        self.fingerprint = fingerprint_from_hash160(h160, self.prefix)
        self.created = created or datetime.now(timezone.utc).isoformat()

    def sign(self, payload: bytes) -> str:
        """Sign SHA256d(payload); DER-encoded hex, Bitcoin 0.1 style."""
        from ecdsa.util import sigencode_der
        digest = sha256d(payload)
        sig = self._sk.sign_digest_deterministic(
            digest, hashfunc=hashlib.sha256, sigencode=sigencode_der)
        return sig.hex()

    def public_hex(self) -> str:
        return self.public_bytes.hex()

    def secret_hex(self) -> str:
        """The portable part: raw 32-byte master key. Never shared."""
        return self._sk.to_string().hex()

    def identity_dict(self) -> dict:
        """Everything regula_core (or a future you) needs to import this
        identity: the secret is the source of truth, the rest is derived."""
        return {
            "scheme": SCHEME,
            "secret": self.secret_hex(),
            "public": self.public_hex(),
            "node_id": self.node_id,
            "prefix": self.prefix,
            "fingerprint": self.fingerprint,
            "created": self.created,
        }

    @classmethod
    def from_identity_dict(cls, d: dict) -> "NodeIdentity":
        # the secret re-derives everything; stored identifiers are
        # informational only, never trusted over the key itself. The
        # stored prefix keeps the fingerprint stable across env changes.
        return cls(bytes.fromhex(d["secret"]),
                   created=d.get("created"),
                   prefix=d.get("prefix"))


def _data_dir_default() -> str:
    from .config import data_dir_default
    return data_dir_default()


def identity_path(data_dir: str | None = None) -> str:
    """Canonical identity file location. CETI_IDENTITY overrides it."""
    if IDENTITY_ENV in os.environ:
        return os.path.expanduser(os.environ[IDENTITY_ENV])
    base = data_dir or _data_dir_default()
    return os.path.join(os.path.abspath(base), IDENTITY_FILE)


def _save_identity(path: str, ident: NodeIdentity) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ident.identity_dict(), f, indent=2)
        f.write("\n")
    os.chmod(path, 0o600)


def generate_identity() -> NodeIdentity:
    """Fresh throwaway identity (tests, peers). Never touches disk."""
    from ecdsa import SECP256k1
    _crypto()
    while True:
        secret = secrets.token_bytes(32)
        if 0 < int.from_bytes(secret, "big") < SECP256k1.order:
            return NodeIdentity(secret)


def new_test_identity() -> NodeIdentity:
    """Alias kept for the test-suite's throwaway keys."""
    return generate_identity()


def identity_exists(data_dir: str | None = None) -> bool:
    """True if an identity file (current or legacy) is already on disk."""
    base = os.path.dirname(identity_path(data_dir))
    return any(os.path.exists(os.path.join(base, f))
               for f in (IDENTITY_FILE, LEGACY_IDENTITY_FILE,
                         LEGACY_JSON_FILE))


def ensure_identity(data_dir: str | None = None,
                    quiet: bool = False) -> NodeIdentity:
    """Load the node identity, generating a fresh one on first run.

    Generate once, save locally (0600), reuse forever. Legacy files
    (identity.key secret-hex, identity.json) are upgraded to
    identity.dat automatically.
    """
    _crypto()
    path = identity_path(data_dir)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return NodeIdentity.from_identity_dict(json.load(f))
    # legacy upgrades: identity.json (brief naming detour) holds a
    # full identity dict; identity.key holds raw secret hex.
    for legacy_file, loader in (
            (LEGACY_JSON_FILE,
             lambda p: NodeIdentity.from_identity_dict(
                 json.load(open(p, encoding="utf-8")))),
            (LEGACY_IDENTITY_FILE,
             lambda p: NodeIdentity(bytes.fromhex(
                 open(p, encoding="utf-8").read().strip())))):
        legacy = os.path.join(os.path.dirname(path), legacy_file)
        if legacy != path and os.path.exists(legacy):
            ident = loader(legacy)
            _save_identity(path, ident)
            return ident
    ident = generate_identity()
    _save_identity(path, ident)
    if not quiet:
        print("new CETI identity: %s (%s)"
              % (ident.fingerprint, path), flush=True)
    return ident


def load_identity(data_dir: str | None = None,
                  quiet: bool = False) -> NodeIdentity:
    """Back-compat alias for ensure_identity."""
    return ensure_identity(data_dir, quiet=quiet)


def sign_bytes(payload: bytes, identity=None,
               data_dir: str | None = None) -> dict:
    """Sign payload with the CETI identity (default: the persistent one,
    generated on first use). Pass an explicit identity in tests."""
    ident = identity or load_identity(data_dir)
    return {
        "node_id": ident.node_id,
        "fingerprint": ident.fingerprint,
        "public_hex": ident.public_hex(),
        "signature": ident.sign(payload),
        "scheme": SCHEME,
    }


def verify_bytes(payload: bytes, public_hex: str,
                 signature_hex: str) -> bool:
    """Verify a signature. Needs no private key — any peer can run this."""
    from ecdsa import SECP256k1, VerifyingKey
    from ecdsa.util import sigdecode_der
    _crypto()
    try:
        vk = VerifyingKey.from_string(bytes.fromhex(public_hex),
                                      curve=SECP256k1,
                                      hashfunc=hashlib.sha256)
        return bool(vk.verify_digest(bytes.fromhex(signature_hex),
                                     sha256d(payload),
                                     sigdecode=sigdecode_der))
    except Exception:
        return False
