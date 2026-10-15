# Gopher hole — node connections, BBS, sharing

The Gopher layer is CETI's social surface: peers, a signed board,
and shared files — in the shape of Gopher menus. Retro surface,
modern trust underneath. Keep it simple: no live transport, no
central server, no accounts. Signed files move between nodes by
whatever means the sysop has (USB, email, torrent later), and
everything is verified on arrival.

## The three pieces

### Peers — `ceti/peers.py`

A peer is a public key. `data/peers/<fingerprint>.json` holds one
record per peer. Node ID and fingerprint are always re-derived
from the public key — claimed identifiers are never trusted.

- `/peer card` — print this node's identity card (public half
  only: node_id, fingerprint, public_hex, scheme, created).
  Hand it to another node; it files you with `/peer add`.
- `/peer add <public key hex>` — file a peer by public key.
  Re-adding is idempotent.
- `/peers` — list filed peers.

### Board — `ceti/board.py`

One JSON file per post in `data/board/posts/<id>.json`, where the
id is the sha256 of the canonical post bytes. Every post is
signed with the author's node identity (ECDSA-SHA256d-DER on
secp256k1, the same scheme as dataset signing).

- `/board` — list valid posts, oldest first. Posts whose
  signature does not verify are never displayed; they are counted
  as rejected, loudly.
- `/board read <n>` — read the nth valid post.
- `/board post <subject> | <body>` — sign and file a post as this
  node.
- `/board import <file>` — verify a post file from another node
  and file it. Any verification failure refuses the import.

A post's signature covers everything except the signature block
itself (canonical JSON, sorted keys). The author's fingerprint
must match the public key, or the post is refused.

### Sharing — `ceti/share.py`

No new format: sharing reuses the signed-release machinery
(`datasets.DatasetStore` + `merge.ingest_tarball`). Sharing a
file (or a directory of `.aiml` cassettes) cuts a signed,
chained, Merkle-rooted tarball. Fetching one verifies signature,
hashes, chain, and Merkle root BEFORE anything merges — a bad
bundle is refused, never trusted. Peer vectors are never trusted;
embeddings are recomputed locally.

- `/share <path>` — cut a signed bundle, print the tarball path.
  Hand the tarball to another node.
- `/fetch <tarball>` — verify + merge a peer's tarball. Prints
  what was added and how many conflicts the local side won.

## The menu — `ceti/gopher.py`

`/gopher` opens the hole:

```
CETI // gopher hole
1. browse nodes (peers)
2. read the board
3. shared cassettes
4. shared files (bundles)
5. post to the board
6. chat room
7. mail
0. back
```

The menu is a thin shell over peers / board / share / chat /
mail — no duplicate logic. Pick a number, get the thing.

## Discovery — `ceti/nodelist.py`

How nodes find each other: a **signed nodelist** plus
**gossip**.

- A nodelist is a signed JSON file listing known nodes:
  fingerprint, public key, optional address. It ships with
  releases and lives at a well-known URL (e.g.
  `https://darknet.ca/nodes`).
- `/nodelist build` — sign a nodelist from this node's peer
  book. `/nodelist fetch <url|file>` — fetch, verify, file
  every peer. `/nodelist show` — list filed peers.
- `/gossip export` — sign your peer list to a file to hand
  over. `/gossip import <file>` — verify and merge a peer's
  list. Same format, same verification as the nodelist.
- The whole list is refused if ANY entry fails: bad signature,
  signer whose fingerprint does not match its key, or a node
  entry whose fingerprint does not match its key. Notes are
  stripped before publishing — only public routing info ships.
- Addresses are claimed, never verified. A key is
  self-verifying by derivation; an address is just a hint.

## Chat — `ceti/chat.py`

`/chat <room>` enters a room: a directory of signed messages
under `data/chat/<room>/`. The loop polls every 2s and prints
new messages as they arrive; what you type is signed and
posted. Two terminals sharing a room directory chat live.
`/quit` (or Ctrl-C) leaves. Tampered messages are excluded
from polling, never displayed.

## Mail — `ceti/mail.py`

`/mail <fingerprint> <subject> | <body>` sends a signed
direct message to one peer — refused if the recipient is not
a filed peer (no mail to strangers). `/mail` lists the inbox;
`/mail import <file>` verifies and files an incoming message.
Messages addressed to someone else are skipped, never read.

## Upload / download

`/upload <file>` cuts a signed bundle and drops the tarball
in `data/drop/`. `/download` lists the drop; `/download
<file>` verifies and merges it. The drop is the exchange
point — bundles move from there by whatever means the sysop
has.

## Trust rules (the whole layer, in one paragraph)

The key IS the identity. The signature IS the authorship proof.
Nothing unsigned is displayed or merged. Nothing is trusted
because of who sent it — the signature and the hashes are
checked, every time, on every arrival. A bad signature is a
loud refusal, never a silent skip.

## What is deliberately NOT here yet

- Live transport: no node-to-node network calls. Bundles and
  posts travel as files. The transport (and torrent seeding of
  bundles) is the next step, not this one.
- Federated board sync: posts do not auto-propagate. Import is
  explicit.
- Moderation: there is no delete or ban. A post lives in its
  file; the sysop removes the file.
