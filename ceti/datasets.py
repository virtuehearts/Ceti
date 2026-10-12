"""datasets — signed knowledge sets, blockchain-style.

A "set" is a versioned bundle of knowledge files (the database +
cassettes). Before a set is seeded it is:
  1. hashed  — sha256 per file, merkle_root over the sorted file hashes
  2. chained — each manifest carries prev_hash, the sha256 of the
               previous manifest: a tamper-evident ledger, like blocks
  3. signed  — node ECDSA signature over SHA256d(canonical manifest),
               regula_core style. No signature, no seed.
  4. packed  — the bundle is gzip-compressed (tar.gz): this is the
               artifact archivists actually seed via BitTorrent.

Manifests live inside each bundle as manifest.json; every release is
appended to data/datasets/ledger.jsonl. verify() recomputes everything
from the files on disk: any tampering — a single changed byte, a
swapped file, a forged manifest — breaks verification.
"""
from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
from datetime import datetime, timezone

from . import signing

GENESIS_HASH = "0" * 64
GZIP_LEVEL = 6  # size vs. low-spec CPU time


def pack_tarball(bundle: str) -> str:
    """gzip the bundle directory -> <bundle>.tar.gz. Returns the path."""
    tarball = bundle + ".tar.gz"
    tmp = tarball + ".tmp"
    with tarfile.open(tmp, "w:gz", compresslevel=GZIP_LEVEL) as tar:
        tar.add(bundle, arcname=os.path.basename(bundle))
    os.replace(tmp, tarball)
    return tarball


def unpack_tarball(tarball: str, dest: str) -> str:
    """Extract a release tarball; returns the bundle dir path."""
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(dest)
    # the archive holds a single top-level dir
    names = os.listdir(dest)
    if len(names) != 1 or not os.path.isdir(os.path.join(dest, names[0])):
        raise ValueError("bad bundle tarball layout")
    return os.path.join(dest, names[0])


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def merkle_root(file_hashes: list[str]) -> str:
    """sha256 over the sorted file hashes (order-independent)."""
    h = hashlib.sha256()
    for fh in sorted(file_hashes):
        h.update(bytes.fromhex(fh))
    return h.hexdigest()


def canonical_bytes(manifest: dict) -> bytes:
    """Deterministic bytes for signing: everything except the
    signature block, JSON-sorted, no whitespace."""
    unsigned = {k: v for k, v in manifest.items() if k != "signature"}
    return json.dumps(unsigned, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class DatasetStore:
    def __init__(self, datasets_dir: str):
        self.dir = os.path.abspath(datasets_dir)
        os.makedirs(self.dir, exist_ok=True)
        self.ledger = os.path.join(self.dir, "ledger.jsonl")

    # ---- release ------------------------------------------------------
    def _bundle_dir(self, name: str, version: int) -> str:
        return os.path.join(self.dir, "%s-v%d" % (name, version))

    def _read_manifest(self, bundle: str) -> dict:
        with open(os.path.join(bundle, "manifest.json")) as f:
            return json.load(f)

    def latest(self, name: str) -> tuple[dict, str] | None:
        """Return (manifest, bundle_dir) of the newest version."""
        best = None
        for fn in sorted(os.listdir(self.dir)):
            if fn.startswith(name + "-v"):
                bundle = os.path.join(self.dir, fn)
                mp = os.path.join(bundle, "manifest.json")
                if not os.path.isdir(bundle) or not os.path.exists(mp):
                    continue
                m = self._read_manifest(bundle)
                if best is None or m["version"] > best[0]["version"]:
                    best = (m, bundle)
        return best

    def release(self, name: str, files: list[str],
                identity=None, note: str = "") -> tuple[dict, str, str]:
        """Hash, chain, sign, then gzip a new set version.
        Returns (manifest, bundle_dir, tarball_path). The tarball is
        the seed artifact; the bundle dir stays for verification."""
        prev = self.latest(name)
        version = (prev[0]["version"] + 1) if prev else 1
        prev_hash = (sha256_file(
            os.path.join(prev[1], "manifest.json"))
            if prev else GENESIS_HASH)

        bundle = self._bundle_dir(name, version)
        os.makedirs(bundle, exist_ok=True)
        entries = []
        for fp in sorted(files):
            ap = os.path.abspath(fp)
            if os.path.isdir(ap):
                # cassettes dir: bundle each .aiml at top level
                for fn in sorted(os.listdir(ap)):
                    if fn.endswith(".aiml"):
                        src = os.path.join(ap, fn)
                        entries.append(self._bundle_file(src, bundle))
            else:
                entries.append(self._bundle_file(ap, bundle))

        manifest = {
            "dataset": name,
            "version": version,
            "prev_hash": prev_hash,
            "created_utc": _utcnow(),
            "files": entries,
            "merkle_root": merkle_root([e["sha256"] for e in entries]),
            "note": note,
        }
        sig = signing.sign_bytes(canonical_bytes(manifest),
                                 identity=identity)
        manifest["signature"] = sig

        mp = os.path.join(bundle, "manifest.json")
        with open(mp, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")

        tarball = pack_tarball(bundle)

        with open(self.ledger, "a") as f:
            f.write(json.dumps({
                "dataset": name, "version": version,
                "manifest_sha256": sha256_file(mp),
                "merkle_root": manifest["merkle_root"],
                "tarball": os.path.basename(tarball),
                "tarball_sha256": sha256_file(tarball),
                "tarball_bytes": os.path.getsize(tarball),
                "unpacked_bytes": sum(e["bytes"] for e in entries),
                "node_id": sig["node_id"],
                "fingerprint": sig["fingerprint"],
                "created_utc": manifest["created_utc"],
            }, sort_keys=True) + "\n")
        return manifest, bundle, tarball

    @staticmethod
    def _bundle_file(src: str, bundle: str) -> dict:
        import shutil
        fn = os.path.basename(src)
        shutil.copy2(src, os.path.join(bundle, fn))
        return {"path": fn, "sha256": sha256_file(src),
                "bytes": os.path.getsize(src)}

    # ---- verify ---------------------------------------------------------
    def verify_tarball(self, tarball: str) -> tuple[bool, list[str]]:
        """Verify a release tarball exactly as a peer would: extract to
        a temp dir, then run full bundle verification."""
        import shutil
        tmp = tempfile.mkdtemp(prefix="dataset-verify-")
        try:
            try:
                bundle = unpack_tarball(tarball, tmp)
            except Exception as e:
                return False, ["tarball unreadable: %s" % e]
            return self.verify(bundle)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def verify(self, bundle: str) -> tuple[bool, list[str]]:
        """Full verification of a bundle: files, merkle root, chain,
        signature. Returns (ok, reasons)."""
        reasons: list[str] = []
        mp = os.path.join(bundle, "manifest.json")
        if not os.path.exists(mp):
            return False, ["no manifest.json in bundle"]
        manifest = self._read_manifest(bundle)

        # 1. files on disk match the manifest
        hashes = []
        for e in manifest.get("files", []):
            fp = os.path.join(bundle, e["path"])
            if not os.path.exists(fp):
                reasons.append("missing file: %s" % e["path"])
                continue
            h = sha256_file(fp)
            hashes.append(h)
            if h != e["sha256"]:
                reasons.append("hash mismatch: %s" % e["path"])
            if os.path.getsize(fp) != e["bytes"]:
                reasons.append("size mismatch: %s" % e["path"])

        # 2. merkle root
        if hashes and merkle_root(hashes) != manifest.get("merkle_root"):
            reasons.append("merkle root mismatch")

        # 3. chain continuity
        if manifest["version"] > 1:
            prev_bundle = self._bundle_dir(manifest["dataset"],
                                           manifest["version"] - 1)
            prev_mp = os.path.join(prev_bundle, "manifest.json")
            if not os.path.exists(prev_mp):
                reasons.append("previous bundle not found")
            elif sha256_file(prev_mp) != manifest.get("prev_hash"):
                reasons.append("prev_hash does not match previous manifest")

        # 4. signature over the canonical bytes
        sig = manifest.get("signature", {})
        if not sig:
            reasons.append("no signature block")
        elif not signing.verify_bytes(
                canonical_bytes(manifest),
                sig.get("public_hex", ""), sig.get("signature", "")):
            reasons.append("signature invalid")

        return (len(reasons) == 0, reasons)
