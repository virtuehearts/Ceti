# Distribution

Knowledge wants to travel. Every signed release is a gzip tarball;
the tarball is the **BitTorrent seed artifact**, and the swarm is the
distribution network — no central server, no account, just archivists
seeding sets to each other.

## The lifecycle

1. **Release.** A `release` agent task (or `DatasetStore.release`)
   cuts the tarball: files hashed, Merkle root computed, chained to
   `prev_hash`, manifest signed with the node identity.
2. **Torrent.** `ceti/seeder.py` writes the `.torrent` for the
   tarball (magnet links included). The seeder announces as the node
   identity, so peers know whose sets they're pulling.
3. **Seed.** `seedbox/` holds a `transmission-daemon` setup script
   (port 51413, DHT + PEX) for a VPS seeder: copy the tarball and
   torrent over, start seeding. `ceti/seedtest.py` proves the wire
   path with a loopback transfer first.
4. **Fetch.** An `ingest` agent task downloads a peer's torrent.
5. **Verify, then merge.** `verify_tarball` checks every file hash,
   the Merkle root, the chain link, and the signature — *before*
   anything is trusted. Then `ceti/merge.py` merges with
   record-identity dedup and score-based conflict resolution. Peer
   vectors are never trusted; embeddings are recomputed locally.

## Release checklist

Cut → torrent → seed → **second-machine download verification**
(a fresh peer must reproduce the hashes) → announce.

## The merge protocol

`ceti/merge.py` ingests a peer's signed bundle with git-like
semantics, minus the git binary (sqlite files don't line-merge, and
a dependency would break the stdlib-only path):

- The bundle is **verified before anything merges** — signature,
  chain, Merkle root.
- Record identity is `sha256(normalized question)`: new records are
  added, identical ones skipped, conflicts resolved by relevance
  score (higher wins, ties keep local) and **logged**.
- Aliases and AIML categories merge; **peer vectors are never
  trusted** — embeddings are recomputed locally.
- The merge is appended to the dataset ledger; the next signed
  release is the commit.
