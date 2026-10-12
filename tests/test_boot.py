import json
import os
import socket
import sqlite3
import tempfile
import time
import unittest

from ceti import signing
from ceti.boot import boot
from ceti.cli import heartbeat_tick
from ceti.config import BrainConfig

try:
    signing._crypto()
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def write_aiml(path, n=2):
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<aiml version="1.0">\n')
        for i in range(n):
            f.write("<category><pattern>PATTERN %d</pattern>"
                    "<template>answer %d</template></category>\n" % (i, i))
        f.write("</aiml>\n")


def make_data_dir(with_persona=True, corrupt_aiml=False):
    d = tempfile.mkdtemp()
    if with_persona:
        with open(os.path.join(d, "persona.md"), "w") as f:
            f.write("# Persona\nTest voice.\n")
    if corrupt_aiml:
        with open(os.path.join(d, "seed.aiml"), "w") as f:
            f.write("<aiml><category><pattern>oops")
    else:
        write_aiml(os.path.join(d, "seed.aiml"))
    write_aiml(os.path.join(d, "learned.aiml"), n=1)
    con = sqlite3.connect(os.path.join(d, "knowledge.sqlite3"))
    con.execute("CREATE TABLE qa (id INTEGER PRIMARY KEY,"
                " question TEXT, answer TEXT, score REAL, created REAL,"
                " embedding BLOB)")
    con.execute("CREATE TABLE aliases (phrase TEXT PRIMARY KEY,"
                " expansion TEXT)")
    con.execute("INSERT INTO qa VALUES (1,'q','a',0.5,0,x'00')")
    con.commit()
    con.close()
    with open(os.path.join(d, "topics.txt"), "w") as f:
        f.write("topic one\ntopic two\n")
    return d


def cfg_for(d):
    return BrainConfig(
        data_dir=d,
        embed_url="http://127.0.0.1:%d" % free_port(),
        ollama_url="http://127.0.0.1:%d" % free_port())


class TestBoot(unittest.TestCase):
    def setUp(self):
        for k in ("OPENROUTER_API_KEY", "NVIDIA_API_KEY"):
            os.environ.pop(k, None)

    def test_full_boot(self):
        rep = boot(cfg_for(make_data_dir()))
        by_name = {c.name: c for c in rep.checks}
        self.assertTrue(rep.ok)
        self.assertEqual(by_name["aiml"].status, "ok")
        self.assertIn("3 categories", by_name["aiml"].detail)
        self.assertEqual(by_name["knowledge"].status, "ok")
        # nothing listening: services degrade, boot survives
        self.assertEqual(by_name["embeddings"].status, "warn")
        self.assertEqual(by_name["ollama"].status, "warn")
        self.assertEqual(by_name["cloud"].status, "warn")
        self.assertEqual(by_name["datasets"].status, "warn")
        self.assertIn("2 queued", by_name["topics"].detail)
        self.assertIn("power-on self test", rep.text)

    def test_missing_persona_fails(self):
        rep = boot(cfg_for(make_data_dir(with_persona=False)))
        self.assertFalse(rep.ok)
        by_name = {c.name: c for c in rep.checks}
        self.assertEqual(by_name["persona"].status, "fail")

    def test_corrupt_aiml_fails(self):
        rep = boot(cfg_for(make_data_dir(corrupt_aiml=True)))
        self.assertFalse(rep.ok)
        by_name = {c.name: c for c in rep.checks}
        self.assertEqual(by_name["aiml"].status, "fail")

    def test_missing_db_warns_without_creating_one(self):
        # a fresh install has no knowledge DB yet: warn, don't refuse
        d = make_data_dir()
        os.remove(os.path.join(d, "knowledge.sqlite3"))
        rep = boot(cfg_for(d))
        by_name = {c.name: c for c in rep.checks}
        self.assertEqual(by_name["knowledge"].status, "warn")
        self.assertTrue(rep.ok)  # warnings don't block startup
        # boot must not create files as a side effect
        self.assertFalse(os.path.exists(os.path.join(d, "knowledge.sqlite3")))

    @unittest.skipUnless(HAS_CRYPTO, "ecdsa package not available")
    def test_identity_generated_on_boot(self):
        # first run: identity is generated, fingerprint reported
        d = make_data_dir()
        rep = boot(cfg_for(d))
        by_name = {c.name: c for c in rep.checks}
        self.assertEqual(by_name["identity"].status, "ok")
        self.assertIn("new identity generated", by_name["identity"].detail)
        ident_file = os.path.join(d, signing.IDENTITY_FILE)
        self.assertTrue(os.path.exists(ident_file))
        # second boot: same identity, loaded not generated
        rep2 = boot(cfg_for(d))
        by2 = {c.name: c for c in rep2.checks}
        self.assertIn("loaded", by2["identity"].detail)
        self.assertIn(by_name["identity"].detail.split()[0],
                      by2["identity"].detail)  # same fingerprint

    @unittest.skipUnless(HAS_CRYPTO, "ecdsa package not available")
    def test_first_run_note_explains_identity(self):
        from ceti.boot import FIRST_RUN_IDENTITY_NOTE
        d = make_data_dir()
        rep = boot(cfg_for(d))
        self.assertEqual(len(rep.notes), 1)
        self.assertIn("identity.dat", rep.notes[0])
        self.assertIn("never leaves this machine", rep.notes[0])
        # second boot: no notes, identity already exists
        rep2 = boot(cfg_for(d))
        self.assertEqual(rep2.notes, [])

    @unittest.skipUnless(HAS_CRYPTO, "ecdsa package not available")
    def test_dataset_verified(self):
        from ceti.datasets import DatasetStore
        d = make_data_dir()
        store = DatasetStore(os.path.join(d, "datasets"))
        f = os.path.join(d, "f.txt")
        open(f, "w").write("data")
        ident = signing.new_test_identity()
        store.release("testset", [f], identity=ident)
        rep = boot(cfg_for(d))
        by_name = {c.name: c for c in rep.checks}
        self.assertEqual(by_name["datasets"].status, "ok")
        self.assertIn(ident.fingerprint, by_name["datasets"].detail)


class TestHeartbeat(unittest.TestCase):
    def test_tick_delivers_result(self):
        hb = {"running": False, "result": None, "enabled": True}
        heartbeat_tick(hb, cycle_fn=lambda: {"skipped": "test"})
        deadline = time.time() + 10
        while hb["result"] is None and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(hb["result"], {"skipped": "test"})
        self.assertFalse(hb["running"])

    def test_tick_respects_disabled(self):
        hb = {"running": False, "result": None, "enabled": False}
        heartbeat_tick(hb, cycle_fn=lambda: {"skipped": "test"})
        time.sleep(0.3)
        self.assertIsNone(hb["result"])

    def test_tick_never_raises(self):
        hb = {"running": False, "result": None, "enabled": True}

        def boom():
            raise RuntimeError("research exploded")
        heartbeat_tick(hb, cycle_fn=boom)
        deadline = time.time() + 10
        while hb["result"] is None and time.time() < deadline:
            time.sleep(0.05)
        self.assertIn("error", hb["result"])
        self.assertFalse(hb["running"])


if __name__ == "__main__":
    unittest.main()
