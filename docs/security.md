# Security model

Four boundaries, each enforced in code, not convention:

1. **Identity.** The master key lives in `identity.dat` (0600) and
   never leaves the machine. Only the public key is shared. Every
   signature — dataset releases today, posts and messages tomorrow
   — proves possession without revealing it.
2. **Data.** The PII stripper runs before cloud calls *and* before
   persistence. The local model may see the raw query; nothing else
   does. Learned knowledge contains no personal data by
   construction.
3. **Ingestion.** Peer bundles are verified (signature → chain →
   Merkle → file hashes) before anything merges, and peer vectors
   are never trusted — embeddings are recomputed locally. A
   malicious peer can at worst waste your bandwidth.
4. **Execution.** Agent code runs only inside the QEMU sandbox
   (planned) — snapshot/restore, no network by default, capped
   resources. Tools declare side effects in their manifests.

Threats explicitly out of scope for now: a compromised host (if
they own the box they own `identity.dat`), and Sybil peers (the
web-of-trust work in the roadmap addresses this).
