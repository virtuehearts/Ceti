"""Node-to-node file transfer tests.

Verifies end-to-end file and dataset transfers between CETI nodes using:
- Signed bundle packaging and fetching (`share.share` / `share.fetch` / `merge.ingest_tarball`)
- BitTorrent wire protocol transfers (`seeder` + `seedtest`)
- Drop directory upload and download workflows
- Direct signed mail and nodelist exchange between nodes
"""

import json
import os
import tempfile
import unittest

from ceti import chat as chat_mod
from ceti import mail as mail_mod
from ceti import nodelist as nl_mod
from ceti import peers as peers_mod
from ceti import seeder as seeder_mod
from ceti import seedtest as seedtest_mod
from ceti import share as share_mod
from ceti import signing
from ceti.aiml import AimlBrain
from ceti.datasets import DatasetStore
from ceti.vectormem import VectorMemory

try:
    signing._crypto()
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False


def _fake_embed(texts):
    import zlib
    vecs = []
    for t in texts:
        v = [0.0] * 64
        for w in t.lower().split():
            v[zlib.crc32(w.encode()) % 64] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        vecs.append([x / n for x in v])
    return vecs


@unittest.skipUnless(HAS_CRYPTO, "ecdsa package required for node identity tests")
class TestNodeToFileTransfers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

        # Node A setup
        self.node_a_dir = os.path.join(self.tmp, "node_a")
        os.makedirs(self.node_a_dir, exist_ok=True)
        self.ident_a = signing.generate_identity()
        self.store_a = DatasetStore(os.path.join(self.node_a_dir, "datasets"))
        self.mem_a = VectorMemory(os.path.join(self.node_a_dir, "local.db"), embed_fn=_fake_embed)
        self.aiml_a = AimlBrain(learned_path=os.path.join(self.node_a_dir, "learned.aiml"))

        # Node B setup
        self.node_b_dir = os.path.join(self.tmp, "node_b")
        os.makedirs(self.node_b_dir, exist_ok=True)
        self.ident_b = signing.generate_identity()
        self.store_b = DatasetStore(os.path.join(self.node_b_dir, "datasets"))
        self.mem_b = VectorMemory(os.path.join(self.node_b_dir, "local.db"), embed_fn=_fake_embed)
        self.aiml_b = AimlBrain(learned_path=os.path.join(self.node_b_dir, "learned.aiml"))

        # Register each other as peers
        self.peers_a = peers_mod.PeerBook(self.node_a_dir)
        self.peers_a.add(self.ident_b.public_hex(), note="Node B")

        self.peers_b = peers_mod.PeerBook(self.node_b_dir)
        self.peers_b.add(self.ident_a.public_hex(), note="Node A")

    def test_signed_bundle_file_transfer(self):
        """Test cutting a signed bundle on Node A and transferring/fetching it on Node B."""
        # Create source file on Node A
        doc_path = os.path.join(self.node_a_dir, "manual.txt")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("CETI Node-to-Node Transfer Protocol Manual\n")

        manifest, _bundle_dir, tarball_path = share_mod.share(
            self.store_a, [doc_path], identity=self.ident_a, note="Node A Manual"
        )
        self.assertTrue(os.path.exists(tarball_path))
        self.assertEqual(manifest["signature"]["fingerprint"], self.ident_a.fingerprint)

        # Node B fetches and ingests Node A's bundle
        report = share_mod.fetch(self.store_b, self.mem_b, self.aiml_b, tarball_path)
        self.assertEqual(report["peer"], self.ident_a.fingerprint)

        # Confirm Node B's dataset store verified and registered the bundle
        verified, reasons = self.store_b.verify_tarball(tarball_path)
        self.assertTrue(verified, reasons)

    def test_tampered_bundle_transfer_rejected(self):
        """Test that a tampered bundle transferred between nodes is refused."""
        doc_path = os.path.join(self.node_a_dir, "secret.txt")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("Authentic content from Node A\n")

        _m, _b, tarball_path = share_mod.share(self.store_a, [doc_path], identity=self.ident_a)

        # Corrupt bytes in tarball during transfer
        with open(tarball_path, "r+b") as f:
            f.seek(50)
            b = f.read(1)
            f.seek(50)
            f.write(bytes([b[0] ^ 0xFF]))

        # Node B attempts to fetch/ingest
        with self.assertRaises(ValueError):
            share_mod.fetch(self.store_b, self.mem_b, self.aiml_b, tarball_path)

    def test_bittorrent_wire_transfer(self):
        """Test file transfer over BitTorrent wire protocol between seeder and leecher."""
        # Node A creates a release tarball
        data_file = os.path.join(self.node_a_dir, "data_payload.bin")
        with open(data_file, "wb") as f:
            f.write(b"NODE_A_PAYLOAD_DATA_" * 1000)

        _m, _b, tarball_path = self.store_a.release("node_a_release", [data_file], identity=self.ident_a)

        # Mint a .torrent for the tarball
        torrent_info = seeder_mod.create_torrent(tarball_path)
        torrent_path = torrent_info["torrent_path"]
        self.assertTrue(os.path.exists(torrent_path))

        # Node B downloads from Node A over loopback BitTorrent connection
        download_dir = os.path.join(self.node_b_dir, "downloads")
        res = seedtest_mod.run(torrent_path, tarball_path, out_dir=download_dir)

        self.assertTrue(res["identical"])
        self.assertTrue(os.path.exists(res["downloaded"]))

        # Node B verifies the downloaded tarball's signature
        ok, reasons = self.store_b.verify_tarball(res["downloaded"])
        self.assertTrue(ok, reasons)

    def test_drop_upload_and_download(self):
        """Test Node A uploading to drop directory and Node B downloading from drop."""
        drop_dir = os.path.join(self.tmp, "shared_drop")
        os.makedirs(drop_dir, exist_ok=True)

        file_a = os.path.join(self.node_a_dir, "dataset_a.txt")
        with open(file_a, "w", encoding="utf-8") as f:
            f.write("Important dataset for drop transfer\n")

        # Node A uploads (cuts signed release in drop directory)
        _m, _b, tarball = share_mod.share(self.store_a, [file_a], identity=self.ident_a)
        drop_tarball = os.path.join(drop_dir, os.path.basename(tarball))
        with open(tarball, "rb") as src, open(drop_tarball, "wb") as dst:
            dst.write(src.read())

        # Node B downloads/ingests from drop
        report = share_mod.fetch(self.store_b, self.mem_b, self.aiml_b, drop_tarball)
        self.assertEqual(report["peer"], self.ident_a.fingerprint)

    def test_mail_and_nodelist_file_exchange(self):
        """Test direct mail and signed nodelist exchange between nodes."""
        # Mail transfer from A to B
        box_a = mail_mod.MailBox(self.node_a_dir)
        box_b = mail_mod.MailBox(self.node_b_dir)

        msg = box_a.send(self.ident_a, self.ident_b.fingerprint, "File Transfer Alert", "Package is ready.", self.peers_a)
        outbox_file = os.path.join(self.node_a_dir, "mail", "outbox", mail_mod.mail_id(msg) + ".json")
        self.assertTrue(os.path.exists(outbox_file))

        box_b.import_mail(outbox_file)
        valid_mail, rejected = box_b.inbox(self.ident_b.fingerprint)
        self.assertEqual(len(valid_mail), 1)
        self.assertEqual(valid_mail[0]["subject"], "File Transfer Alert")

        # Nodelist export from A and import on B
        nl_doc = nl_mod.build_nodelist(self.peers_a.list(), self.ident_a)
        nl_path = os.path.join(self.node_a_dir, "nodes.json")
        nl_mod.save_nodelist(nl_doc, nl_path)

        fetched_nl = nl_mod.fetch_nodelist(nl_path)
        ok, _ = nl_mod.verify_nodelist(fetched_nl)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
