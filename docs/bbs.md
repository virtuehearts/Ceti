# BBS front end

CETI grows a face: a **message board in the BBS tradition** —
boards, threads, posts, file areas — served with a terminal UI that
would feel at home on a dial-up line.

## The three phases

- **Phase 1 — local board.** Message bases in sqlite, file areas on
  disk, served through the REPL first (`/bbs`), ANSI-friendly.
  Single node, single sysop.
- **Phase 2 — signed posts.** Every post signed with the node
  identity. Moderation becomes key-based: the sysop's key grants,
  members post, signatures prove authorship. No passwords on the
  wire.
- **Phase 3 — federated boards.** Peer sync through the merge
  protocol: a post's record identity is the hash of its content,
  conflicts are logged, and the next signed release is the commit —
  the same machinery as dataset merge. File and image serving
  arrives with the regula_core plug.

## Why a board

The board is the human side of the swarm: datasets are the mind,
the board is where the minds meet. A renegade BBS for the
sovereign-stack era — your node, your boards, your rules, signed
and seedable like everything else CETI knows.
