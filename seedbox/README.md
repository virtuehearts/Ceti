# Seedbox — where the torrents actually seed

## Testing (no seedbox needed)

`brain/seedtest.py` runs a real BitTorrent wire session over
loopback — seeder and leecher as pure-stdlib peers (handshake,
bitfield, interested/unchoke, request/piece) — then hash-checks the
downloaded copy. It proves the minted `.torrent` is valid and the data
moves bit-identically. It does not test internet reachability.

```bash
python -m brain.seedtest data/datasets/brain-knowledge-v2.tar.gz.torrent \
    data/datasets/brain-knowledge-v2.tar.gz --out /tmp/bt-proof
# then verify the downloaded copy as a peer would:
# DatasetStore("data/datasets").verify_tarball("/tmp/bt-proof/downloaded-...")
```

Verified 2026-10-11: v2 tarball transferred bit-identical over the
wire and passed full signature verification on the downloaded copy.

The brain mints `.torrent` files on the sandbox, but the sandbox
cannot seed (NAT: no inbound peers). Seeding happens on connected
machines. This pack sets up the first one: your AlmaLinux VPS.

## Setup (once, as root on the VPS)

```bash
bash setup-seedbox.sh
```

Installs `transmission-daemon`, opens TCP/UDP 51413, enables DHT +
peer exchange, and watches `/var/lib/brain-seeds` — any `.torrent`
dropped there starts seeding automatically.

## Publishing a release

On the sandbox, after `python -m brain.prototype` (or a
`DatasetStore.release()`):

```bash
# copy the release tarball + its torrent to the VPS
scp data/datasets/brain-knowledge-v2.tar.gz \
    data/datasets/brain-knowledge-v2.tar.gz.torrent \
    root@vps:/var/lib/brain-seeds/
# ... and the data file itself so the daemon has something to seed:
# (the .tar.gz IS the data — one file, that is the whole point)
```

The watch dir picks up the `.torrent`; the daemon finds the matching
`.tar.gz` beside it and starts seeding. Verify it's working:

```bash
transmission-remote -l
```

## Archivists

Anyone can seed: hand them the `.torrent` (or the magnet link from the
release output). The signed manifest inside the tarball means no peer
has to trust the seeder — `DatasetStore.verify_tarball()` checks the
hashes, the chain, and the node signature on download.

## Notes

- Keep the daemon's RPC on 127.0.0.1 only (as configured). Don't
  expose it.
- One tarball = one torrent = one magnet. Clean.
- When a v3 releases, seed it alongside v2 for a while so peers can
  migrate; then retire the old one.
