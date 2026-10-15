"""peers — the node address book for the Gopher layer.

A peer is a public key. Nothing else is trusted: node_id and
fingerprint are always re-derived from the public key, never taken
from what the peer claims. The peer file is just a cache of that
derivation plus a human note.

Identity cards are the handshake: your node hands out its public
half (never the secret) and the other node files it. No passwords,
no accounts — the key IS the identity.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from . import signing


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def derive_peer(public_hex: str) -> dict:
    """Derive a peer record from a public key. Claimed identifiers
    are never accepted — node_id/fingerprint come from the key."""
    from ecdsa import SECP256k1, VerifyingKey
    signing._crypto()
    try:
        vk = VerifyingKey.from_string(bytes.fromhex(public_hex),
                                      curve=SECP256k1,
                                      hashfunc=__import__("hashlib").sha256)
    except Exception as e:
        raise ValueError("bad public key: %s" % e)
    pub = vk.to_string("uncompressed")
    h160 = signing.hash160(pub)
    return {
        "node_id": "jn1" + h160.hex(),
        "fingerprint": signing.fingerprint_from_hash160(h160),
        "public_hex": public_hex.lower(),
        "scheme": signing.SCHEME,
    }


def identity_card(identity, address: str = "") -> dict:
    """The public half of this node: safe to hand to anyone.
    Address is optional and claimed, never verified — a node
    behind NAT may have no stable address at all."""
    card = {
        "node_id": identity.node_id,
        "fingerprint": identity.fingerprint,
        "public_hex": identity.public_hex(),
        "scheme": signing.SCHEME,
        "created": identity.created,
    }
    if address:
        card["address"] = address
    return card


def card_to_peer(card: dict) -> dict:
    """Validate an identity card: the fingerprint must match the key.
    A card that claims an identity its key does not derive is refused."""
    derived = derive_peer(card["public_hex"])
    if derived["node_id"] != card.get("node_id") or \
            derived["fingerprint"] != card.get("fingerprint"):
        raise ValueError("identity card does not match its public key")
    return derived


class PeerBook:
    """data/peers/<fingerprint>.json — one file per peer."""

    def __init__(self, data_dir: str):
        self.dir = os.path.join(os.path.abspath(data_dir), "peers")
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, fingerprint: str) -> str:
        return os.path.join(self.dir, fingerprint + ".json")

    def add(self, public_hex: str, note: str = "",
            address: str = "") -> dict:
        """File a peer by public key. Re-adding is idempotent:
        same key, same record (note/address updated)."""
        peer = derive_peer(public_hex)
        path = self._path(peer["fingerprint"])
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
            if note:
                existing["note"] = note
            if address:
                existing["address"] = address
            if note or address:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(existing, f, indent=2, sort_keys=True)
                    f.write("\n")
            return existing
        peer["note"] = note
        if address:
            peer["address"] = address
        peer["added"] = _utcnow()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(peer, f, indent=2, sort_keys=True)
            f.write("\n")
        return peer

    def add_card(self, card: dict, note: str = "") -> dict:
        """File a peer from a validated identity card."""
        peer = card_to_peer(card)
        return self.add(peer["public_hex"], note=note,
                        address=card.get("address", ""))

    def get(self, fingerprint: str) -> dict | None:
        path = self._path(fingerprint)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def known(self, public_hex: str) -> bool:
        try:
            return self.get(derive_peer(public_hex)["fingerprint"]) is not None
        except ValueError:
            return False

    def list(self) -> list[dict]:
        out = []
        for fn in sorted(os.listdir(self.dir)):
            if fn.endswith(".json"):
                with open(os.path.join(self.dir, fn),
                          encoding="utf-8") as f:
                    out.append(json.load(f))
        return out
