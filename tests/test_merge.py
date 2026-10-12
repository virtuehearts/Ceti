import os
import sqlite3
import tempfile
import unittest

from ceti import signing
from ceti.aiml import AimlBrain, normalize_pattern
from ceti.datasets import DatasetStore
from ceti.merge import ingest_tarball, merge_bundle, record_id
from ceti.vectormem import VectorMemory

try:
    signing._crypto()
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False


def fake_embed(texts):
    import zlib
    vecs = []
    for t in texts:
        v = [0.0] * 64
        for w in t.lower().split():
            v[zlib.crc32(w.encode()) % 64] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        vecs.append([x / n for x in v])
    return vecs


def make_db(path, rows, aliases=()):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE qa (id INTEGER PRIMARY KEY,"
                " question TEXT NOT NULL, answer TEXT NOT NULL,"
                " score REAL NOT NULL DEFAULT 0, created REAL NOT NULL,"
                " embedding BLOB NOT NULL)")
    con.execute("CREATE UNIQUE INDEX qa_q ON qa(question)")
    con.execute("CREATE TABLE aliases (phrase TEXT PRIMARY KEY,"
                " expansion TEXT NOT NULL)")
    for q, a, s in rows:
        con.execute("INSERT INTO qa (question, answer, score, created,"
                    " embedding) VALUES (?, ?, ?, ?, ?)",
                    (q, a, s, 0.0, b"\x00" * 64))
    for p, e in aliases:
        con.execute("INSERT INTO aliases VALUES (?, ?)", (p, e))
    con.commit()
    con.close()


def write_aiml(path, cats):
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<aiml version="1.0">\n')
        for pat, tpl in cats:
            f.write("<category><pattern>%s</pattern>"
                    "<template>%s</template></category>\n" % (pat, tpl))
        f.write("</aiml>\n")


class TestMerge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mem = VectorMemory(os.path.join(self.tmp, "local.db"),
                                embed_fn=fake_embed)
        self.mem.add("What is a BBS?", "A dial-up message board.", 0.8)
        self.mem.add("What is FidoNet?", "A BBS network.", 0.5)
        self.mem.add("What is Telnet?", "Remote login protocol.", 0.9)
        self.aiml = AimlBrain(
            learned_path=os.path.join(self.tmp, "learned.aiml"))

        # peer bundle: identical row, new row, conflict (peer wins on
        # score), conflict (local wins on score — same record_id via
        # normalization, different raw text), new + conflicting aliases,
        # new + conflicting AIML categories
        self.bundle = os.path.join(self.tmp, "peer-bundle")
        os.makedirs(self.bundle)
        make_db(os.path.join(self.bundle, "knowledge.sqlite3"), [
            ("What is a BBS?", "A dial-up message board.", 0.8),  # same
            ("What is USENET?", "Distributed discussion.", 0.7),  # new
            ("What is FidoNet?", "A global BBS network.", 0.9),    # peer wins
            ("what is telnet?", "Old remote login.", 0.2),        # local wins
        ], aliases=[("bbs", "bulletin board system"),             # new
                    ("modem", "local-override")])                 # conflict
        # local conflicting alias: keep-local policy
        self.mem.add_alias("modem", "modulator demodulator")
        write_aiml(os.path.join(self.bundle, "peer.aiml"), [
            ("WHAT IS A BBS", "A dial-up message board."),   # new pattern
            ("WHAT IS USENET", "Distributed discussion."),   # new pattern
        ])
        self.aiml.learn("what is usenet", "Something else entirely.")

    def test_record_id_stable(self):
        self.assertEqual(record_id("What is a BBS?"),
                         record_id("what IS a bbs?"))

    def test_merge_bundle(self):
        rep = merge_bundle(self.mem, self.aiml, self.bundle)
        self.assertEqual(rep["qa_added"], 1)      # USENET
        self.assertEqual(rep["qa_updated"], 1)    # FidoNet peer wins
        # same-row + local-wins conflict + dup question all skipped
        self.assertEqual(rep["qa_skipped"], 2)
        self.assertEqual(len(rep["conflicts"]), 2)
        self.assertEqual(rep["aliases_added"], 1)
        self.assertEqual(rep["aliases_skipped"], 1)
        self.assertEqual(rep["aiml_added"], 1)    # BBS pattern
        self.assertEqual(rep["aiml_skipped"], 1)  # USENET learned already
        self.assertEqual(len(rep["aiml_conflicts"]), 1)

        rows = {q: (a, s) for q, a, s in self.mem.db.execute(
            "SELECT question, answer, score FROM qa")}
        self.assertEqual(rows["What is a BBS?"][0],
                         "A dial-up message board.")  # untouched
        self.assertEqual(rows["What is FidoNet?"],
                         ("A global BBS network.", 0.9))  # peer won
        self.assertIn("What is USENET?", rows)
        # conflicting local alias kept
        self.assertEqual(self.mem.db.execute(
            "SELECT expansion FROM aliases WHERE phrase='modem'"
            ).fetchone()[0], "modulator demodulator")
        # merged AIML pattern answers
        self.assertEqual(self.aiml.learned_template(
            normalize_pattern("what is a bbs")), "A dial-up message board.")

    def test_merge_is_idempotent(self):
        merge_bundle(self.mem, self.aiml, self.bundle)
        rep = merge_bundle(self.mem, self.aiml, self.bundle)
        self.assertEqual(rep["qa_added"], 0)
        self.assertEqual(rep["qa_updated"], 0)
        self.assertEqual(rep["aiml_added"], 0)


@unittest.skipUnless(HAS_CRYPTO, "ecdsa package not available")
class TestIngest(unittest.TestCase):
    def test_ingest_verifies_first(self):
        tmp = tempfile.mkdtemp()
        store = DatasetStore(os.path.join(tmp, "datasets"))
        mem = VectorMemory(os.path.join(tmp, "local.db"),
                           embed_fn=fake_embed)
        aiml = AimlBrain(learned_path=os.path.join(tmp, "learned.aiml"))
        ident = signing.new_test_identity()

        src = os.path.join(tmp, "src")
        os.makedirs(src)
        make_db(os.path.join(src, "knowledge.sqlite3"),
                [("What is a BBS?", "A dial-up board.", 0.9)])
        write_aiml(os.path.join(src, "peer.aiml"),
                   [("WHAT IS A BBS", "A dial-up board.")])
        m, bundle, tarball = store.release(
            "peer-knowledge",
            [os.path.join(src, "knowledge.sqlite3"),
             os.path.join(src, "peer.aiml")],
            identity=ident)

        rep = ingest_tarball(store, mem, aiml, tarball)
        self.assertEqual(rep["qa_added"], 1)
        self.assertEqual(rep["aiml_added"], 1)
        self.assertEqual(rep["peer"], ident.fingerprint)
        # merge recorded in the ledger
        with open(store.ledger) as f:
            lines = [l for l in f if '"type": "merge"' in l]
        self.assertEqual(len(lines), 1)

    def test_ingest_refuses_tampered(self):
        tmp = tempfile.mkdtemp()
        store = DatasetStore(os.path.join(tmp, "datasets"))
        mem = VectorMemory(os.path.join(tmp, "local.db"),
                           embed_fn=fake_embed)
        aiml = AimlBrain(learned_path=os.path.join(tmp, "learned.aiml"))
        ident = signing.new_test_identity()
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        make_db(os.path.join(src, "knowledge.sqlite3"),
                [("Q?", "A.", 0.5)])
        m, bundle, tarball = store.release(
            "peer-knowledge", [os.path.join(src, "knowledge.sqlite3")],
            identity=ident)
        with open(tarball, "r+b") as f:
            data = bytearray(f.read())
            for i in range(len(data) // 3, len(data) // 3 + 20):
                data[i] ^= 0xFF
            f.seek(0)
            f.write(bytes(data))
        with self.assertRaises(ValueError):
            ingest_tarball(store, mem, aiml, tarball)


if __name__ == "__main__":
    unittest.main()
