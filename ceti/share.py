"""share — signed file/cassette bundles for the Gopher layer.

Thin wrapper over the existing signed-release machinery
(datasets.DatasetStore + merge.ingest_tarball): no new format.
Sharing a file cuts a signed, chained, Merkle-rooted tarball;
fetching one verifies signature, hashes, chain and Merkle root
BEFORE anything merges — a bad bundle is refused, never trusted.
"""
from __future__ import annotations

import os
import re

from .datasets import DatasetStore
from .merge import ingest_tarball


def _safe_name(path: str) -> str:
    base = os.path.basename(os.path.abspath(path))
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)
    return name or "share"


def share(store: DatasetStore, paths: list[str], identity=None,
          note: str = "") -> tuple[dict, str, str]:
    """Cut a signed bundle from files (or .aiml dirs) and return
    (manifest, bundle_dir, tarball_path). The tarball is the thing
    you hand to another node."""
    name = "share-" + _safe_name(paths[0]) if len(paths) == 1 \
        else "share-bundle"
    return store.release(name, paths, identity=identity, note=note)


def fetch(store: DatasetStore, memory, aiml, tarball_path: str,
          policy: str = "score") -> dict:
    """Verify a peer tarball, then merge it. Refuses loudly if
    verification fails — see merge.ingest_tarball."""
    return ingest_tarball(store, memory, aiml, tarball_path,
                          policy=policy)
