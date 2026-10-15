import json
import os
import tempfile
import unittest

from ceti import chat as chat_mod
from ceti import mail as mail_mod
from ceti import nodelist as nl_mod
from ceti import peers as peers_mod
from ceti import signing

try:
    signing._crypto()
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False


@unittest.skipUnless(HAS_CRYPTO, "ecdsa not installed")
class TestNodelist(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.sysop = signing.generate_identity()
        self.alice = signing.generate_identity()
        self.bob = signing.generate_identity()
        self.book_a = peers_mod.PeerBook(
            os.path.join(self.tmp, "a"))
        self.book_a.add(self.alice.public_hex(), note="alice",
                        address="node-a.example:9001")
        self.book_a.add(self.bob.public_hex(), note="bob")

    def test_build_verify_merge_round_trip(self):
        doc = nl_mod.build_nodelist(self.book_a.list(), self.sysop)
        self.assertEqual(len(doc["nodes"]), 2)
        ok, reason = nl_mod.verify_nodelist(doc)
        self.assertTrue(ok, reason)
        book_b = peers_mod.PeerBook(os.path.join(self.tmp, "b"))
        rep = nl_mod.merge_nodelist(book_b, doc)
        self.assertEqual(rep, {"added": 2, "known": 0, "total": 2})
        peer = book_b.get(self.alice.fingerprint)
        self.assertEqual(peer["address"], "node-a.example:9001")
        # second merge is all-known
        rep = nl_mod.merge_nodelist(book_b, doc)
        self.assertEqual(rep["added"], 0)
        self.assertEqual(rep["known"], 2)

    def test_tampered_node_entry_refused(self):
        doc = nl_mod.build_nodelist(self.book_a.list(), self.sysop)
        doc["nodes"][0]["fingerprint"] = "CETI-DEAD-BEEF-CAFE"
        ok, _ = nl_mod.verify_nodelist(doc)
        self.assertFalse(ok)
        book_b = peers_mod.PeerBook(os.path.join(self.tmp, "b"))
        with self.assertRaises(ValueError):
            nl_mod.merge_nodelist(book_b, doc)
        self.assertEqual(book_b.list(), [])

    def test_forged_signer_refused(self):
        doc = nl_mod.build_nodelist(self.book_a.list(), self.sysop)
        doc["signature"]["signature"] = "00" * 70
        ok, _ = nl_mod.verify_nodelist(doc)
        self.assertFalse(ok)

    def test_fetch_from_file(self):
        doc = nl_mod.build_nodelist(self.book_a.list(), self.sysop)
        path = os.path.join(self.tmp, "nodes.json")
        nl_mod.save_nodelist(doc, path)
        fetched = nl_mod.fetch_nodelist(path)
        self.assertEqual(fetched["signer"]["fingerprint"],
                         self.sysop.fingerprint)
        ok, _ = nl_mod.verify_nodelist(fetched)
        self.assertTrue(ok)

    def test_nodelist_strips_notes(self):
        doc = nl_mod.build_nodelist(self.book_a.list(), self.sysop)
        # "alice" contains 'l', which never appears in hex — so any
        # occurrence would be a leaked note, not key material
        self.assertNotIn("alice", json.dumps(doc))
        for entry in doc["nodes"]:
            self.assertNotIn("note", entry)


@unittest.skipUnless(HAS_CRYPTO, "ecdsa not installed")
class TestMail(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.alice = signing.generate_identity()
        self.bob = signing.generate_identity()
        self.book_a = peers_mod.PeerBook(
            os.path.join(self.tmp, "a"))
        self.book_a.add(self.bob.public_hex(), note="bob")
        self.box_a = mail_mod.MailBox(os.path.join(self.tmp, "a"))
        self.box_b = mail_mod.MailBox(os.path.join(self.tmp, "b"))

    def test_send_import_read(self):
        msg = self.box_a.send(self.alice, self.bob.fingerprint,
                              "hello", "meet at the hole", self.book_a)
        out = os.path.join(self.tmp, "a", "mail", "outbox",
                           mail_mod.mail_id(msg) + ".json")
        self.assertTrue(os.path.exists(out))
        imported = self.box_b.import_mail(out)
        self.assertEqual(imported["subject"], "hello")
        valid, rejected = self.box_b.inbox(self.bob.fingerprint)
        self.assertEqual(len(valid), 1)
        self.assertEqual(rejected, [])
        self.assertEqual(valid[0]["body"], "meet at the hole")

    def test_send_to_unknown_peer_refused(self):
        stranger = signing.generate_identity()
        with self.assertRaises(ValueError):
            self.box_a.send(self.alice, stranger.fingerprint,
                            "hi", "body", self.book_a)

    def test_tampered_mail_refused(self):
        msg = self.box_a.send(self.alice, self.bob.fingerprint,
                              "hello", "body", self.book_a)
        out = os.path.join(self.tmp, "a", "mail", "outbox",
                           mail_mod.mail_id(msg) + ".json")
        with open(out) as f:
            bad = json.load(f)
        bad["body"] = "forged"
        badpath = out + ".bad"
        with open(badpath, "w") as f:
            json.dump(bad, f)
        with self.assertRaises(ValueError):
            self.box_b.import_mail(badpath)
        valid, _ = self.box_b.inbox(self.bob.fingerprint)
        self.assertEqual(valid, [])

    def test_inbox_only_shows_my_mail(self):
        msg = self.box_a.send(self.alice, self.bob.fingerprint,
                              "for bob", "body", self.book_a)
        out = os.path.join(self.tmp, "a", "mail", "outbox",
                           mail_mod.mail_id(msg) + ".json")
        self.box_b.import_mail(out)
        valid, _ = self.box_b.inbox(self.alice.fingerprint)
        self.assertEqual(valid, [])


@unittest.skipUnless(HAS_CRYPTO, "ecdsa not installed")
class TestChat(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.alice = signing.generate_identity()
        self.bob = signing.generate_identity()
        self.room = chat_mod.ChatRoom(self.tmp, "lobby")

    def test_post_poll_history(self):
        self.room.post(self.alice, "hello all")
        self.room.post(self.bob, "hey alice")
        hist = self.room.history()
        # same-second posts order by content id, not arrival —
        # compare as a set
        self.assertEqual(sorted(m["body"] for m in hist),
                         ["hello all", "hey alice"])
        new, all_ids, rejected = self.room.poll(set())
        self.assertEqual(len(new), 2)
        self.assertEqual(rejected, 0)
        new2, _, _ = self.room.poll(all_ids)
        self.assertEqual(new2, [])

    def test_tampered_message_excluded(self):
        self.room.post(self.alice, "real")
        valid = self.room.history()
        mid = chat_mod.chat_id(valid[0])
        path = os.path.join(self.tmp, "chat", "lobby", mid + ".json")
        with open(path) as f:
            bad = json.load(f)
        bad["body"] = "forged"
        with open(path, "w") as f:
            json.dump(bad, f)
        self.assertEqual(self.room.history(), [])
        _new, _ids, rejected = self.room.poll(set())
        self.assertEqual(rejected, 1)

    def test_rooms_isolated(self):
        other = chat_mod.ChatRoom(self.tmp, "other")
        self.room.post(self.alice, "in lobby")
        other.post(self.alice, "in other")
        self.assertEqual(len(self.room.history()), 1)
        self.assertEqual(len(other.history()), 1)


if __name__ == "__main__":
    unittest.main()
