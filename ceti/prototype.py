"""prototype — the complete vision loop, end to end.

  1. persona: Jero-as-Khan, the voice of the node
  2. ask: local datasets answer first; the LLM only handles
  3. package: knowledge -> signed dataset (hash, chain, sign)
  4. verify: full re-verification from disk (tamper-evident)
  5. seed: mint the .torrent + magnet for the archivists

Run:  python -m brain.prototype [--ask "question"] [--no-package]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .brain import Brain
from .config import BrainConfig
from .datasets import DatasetStore
from .seeder import create_torrent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", default="What is a BBS?",
                    help="question to run through the loop")
    ap.add_argument("--no-package", action="store_true",
                    help="skip the sign+seed step")
    args = ap.parse_args()

    brain = Brain()
    try:
        print("=== persona ===")
        lines = [ln for ln in brain.persona().split("\n")
                 if ln.strip() and not ln.startswith("#")]
        print(lines[0] if lines else "(no persona)")
        print("memory backend: %s | cassettes: %s"
              % (brain.memory_kind, brain.rack.list()))

        print("\n=== ask: %s ===" % args.ask)
        ans = brain.ask(args.ask)
        print("[%s] (%dms)%s" % (
            ans.tier, ans.latency_ms,
            " handled-by-llm" if ans.trace.get("handled") else ""))
        print(ans.text[:400])
        print("trace:", json.dumps(
            {k: v for k, v in ans.trace.items()
             if k in ("intent", "tiers", "handled")}))
    finally:
        brain.close()

    if args.no_package:
        return

    print("\n=== package: sign the set ===")
    data_dir = os.path.abspath(BrainConfig().data_dir)
    files = [os.path.join(data_dir, "knowledge.sqlite3")]
    cas_dir = os.path.join(data_dir, "cassettes")
    if os.path.isdir(cas_dir):
        files.append(cas_dir)
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("nothing to package yet — no knowledge stored")
        return

    store = DatasetStore(os.path.join(data_dir, "datasets"))
    manifest, bundle, tarball = store.release(
        "ceti-knowledge", files, note="prototype release")
    sig = manifest["signature"]
    print("released %s v%d" % (manifest["dataset"], manifest["version"]))
    print("merkle_root: %s" % manifest["merkle_root"][:16] + "...")
    print("signed by: %s (%s)" % (sig["node_id"][:16] + "...",
                                  sig["fingerprint"]))
    import os as _os
    raw = sum(e["bytes"] for e in manifest["files"])
    packed = _os.path.getsize(tarball)
    print("packed: %d -> %d bytes (%.0f%%)"
          % (raw, packed, 100.0 * packed / max(raw, 1)))

    print("\n=== verify (as a peer would: from the tarball) ===")
    ok, reasons = store.verify_tarball(tarball)
    print("verify:", "OK" if ok else "FAILED: %s" % reasons)

    print("\n=== seed ===")
    tinfo = create_torrent(tarball,
                           out_path=tarball + ".torrent")
    print("torrent: %s (%d file, %d bytes)"
          % (tinfo["torrent_path"], tinfo["files"],
             tinfo["total_bytes"]))
    print("info_hash: %s" % tinfo["info_hash"])
    print("magnet:")
    print(tinfo["magnet"][:120] + "...")


if __name__ == "__main__":
    sys.exit(main())
