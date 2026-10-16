"""boot — power-on self test for the brain.

Like a BIOS POST: every subsystem is loaded AND verified before the
prompt appears. Core failures (config, persona, AIML, knowledge DB)
refuse to start; degraded services (identity without ecdsa, embeddings,
ollama, cloud, datasets) are reported loudly but bootable — the tier
ladder already knows how to fall through them.

    python -m ceti.boot
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

from .cassettes import CassetteRack
from .config import BrainConfig
from .datasets import DatasetStore
from . import signing

CORE = ("config", "persona", "custom local AIML engine", "knowledge")
TIMEOUT = 5


class Check:
    def __init__(self, name: str, status: str, detail: str):
        assert status in ("ok", "warn", "fail")
        self.name, self.status, self.detail = name, status, detail


FIRST_RUN_IDENTITY_NOTE = """\
  -- first run: created this node's identity --
  identity.dat holds the master key (mode 0600). It never leaves this machine.
  Only the public key is ever shared, as the fingerprint above —
  signatures prove it's you without revealing it.
  It signs dataset releases now; later it signs network messages,
  torrent seeds, and the BBS front end.
  /identity shows the details anytime. Back up identity.dat like a key."""


class BootReport:
    def __init__(self, checks: list[Check], elapsed_s: float,
                 notes: list[str] | None = None):
        self.checks = checks
        self.elapsed_s = elapsed_s
        self.notes = notes or []

    @property
    def ok(self) -> bool:
        """Bootable: no core check failed."""
        return all(c.status != "fail" for c in self.checks
                   if c.name in CORE)

    @property
    def text(self) -> str:
        from . import colour as _colour
        n_ok = sum(1 for c in self.checks if c.status == "ok")
        n_warn = sum(1 for c in self.checks if c.status == "warn")
        n_fail = sum(1 for c in self.checks if c.status == "fail")
        lines = [_colour.paint("CETI BIOS v0.3.1 — power-on self test",
                               _colour.BOLD, _colour.GREEN)]
        tag_colour = {"ok": _colour.GREEN, "warn": _colour.AMBER,
                      "fail": _colour.RED}
        for c in self.checks:
            tag = {"ok": "  ok  ", "warn": " warn ",
                   "fail": " FAIL "}[c.status]
            lines.append("%s %-10s %s"
                         % (_colour.paint("[" + tag + "]",
                                          tag_colour[c.status]),
                            c.name, c.detail))
        verdict = ("all systems nominal" if n_fail == 0 and n_warn == 0
                   else "degraded — tiers will fall through gaps"
                   if self.ok else "CORE FAILURE — refusing to start")
        lines.append("boot: %d ok, %d warn, %d fail in %.1fs — %s"
                     % (n_ok, n_warn, n_fail, self.elapsed_s, verdict))
        return "\n".join(lines)


def _http_json(url: str, payload: dict | None = None,
               timeout: int = TIMEOUT) -> dict:
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def check_config(cfg: BrainConfig) -> Check:
    try:
        d = os.path.abspath(cfg.data_dir)
        if not os.path.isdir(d):
            raise OSError("data_dir missing: %s" % d)
        return Check("config", "ok", "data_dir=%s" % d)
    except Exception as e:
        return Check("config", "fail", str(e))


def check_persona(cfg: BrainConfig) -> Check:
    p = os.path.join(os.path.abspath(cfg.data_dir), cfg.persona_file)
    try:
        with open(p, encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return Check("persona", "fail", "%s is empty" % p)
        first = text.splitlines()[0].lstrip("# ").strip()[:60]
        return Check("persona", "ok",
                     "%s (%d chars)" % (first, len(text)))
    except OSError as e:
        return Check("persona", "fail", str(e))


def check_aiml(cfg: BrainConfig) -> Check:
    base = os.path.abspath(cfg.data_dir)
    paths = cfg.aiml_paths()
    rack = CassetteRack(os.path.join(base, cfg.cassette_dir))
    paths += rack.load_paths(cfg.cassettes)
    total, files = 0, 0
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            root = ET.parse(p).getroot()
        except ET.ParseError as e:
            return Check("custom local AIML engine", "fail",
                         "unparseable %s: %s" % (p, e))
        total += len(root.findall("category"))
        files += 1
    if files == 0:
        return Check("custom local AIML engine", "fail", "no AIML files found")
    return Check("custom local AIML engine", "ok",
                 "%d categories across %d files (%d cassettes)"
                 % (total, files, len(rack.list())))


def check_identity(cfg: BrainConfig) -> Check:
    """Node identity: generate on first run, report the fingerprint
    every boot. Never a core failure — without ecdsa the brain runs,
    only signing is disabled."""
    ident_file = signing.identity_path(cfg.data_dir)
    existed = os.path.exists(ident_file) or any(
        os.path.exists(os.path.join(os.path.dirname(ident_file), f))
        for f in (signing.LEGACY_IDENTITY_FILE, signing.LEGACY_JSON_FILE))
    try:
        ident = signing.ensure_identity(cfg.data_dir, quiet=True)
    except Exception as e:
        return Check("identity", "warn", "unavailable: %s" % e)
    note = "loaded" if existed else "new identity generated"
    return Check("identity", "ok",
                 "%s (%s)" % (ident.fingerprint, note))


def check_knowledge(cfg: BrainConfig) -> Check:
    p = cfg.knowledge_path()
    alt = cfg.memory_path()
    # knowledge_path() is the primary store (vector backend);
    # memory_path() is the TF-IDF fallback. Check whichever
    # exists, prefer the primary.
    target = p if os.path.exists(p) else alt
    if not os.path.exists(target):
        # never create a database as a side effect of booting; a
        # missing DB just means a fresh install — it is created on
        # first learn, and the brain answers from AIML until then.
        return Check("knowledge", "warn",
                     "no database yet — starts empty, learns as it goes")
    try:
        con = sqlite3.connect(target)
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"qa", "aliases"} <= tables:
            return Check("knowledge", "fail",
                         "%s missing tables" % target)
        n_qa = con.execute("SELECT COUNT(*) FROM qa").fetchone()[0]
        n_al = con.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]
        con.close()
        return Check("knowledge", "ok",
                     "sqlite: %d pairs, %d aliases" % (n_qa, n_al))
    except sqlite3.Error as e:
        return Check("knowledge", "fail", "%s: %s" % (target, e))


def check_embeddings(cfg: BrainConfig) -> Check:
    t0 = time.time()
    try:
        data = _http_json(cfg.embed_url.rstrip("/") + "/v1/embeddings",
                          {"input": ["boot"]})
        dims = len(data["data"][0]["embedding"])
        ms = (time.time() - t0) * 1000
        return Check("embeddings", "ok",
                     "%s %dd (%.0fms)" % (cfg.embed_url, dims, ms))
    except Exception as e:
        return Check("embeddings", "warn",
                     "%s unreachable — sqlite fallback (%s)"
                     % (cfg.embed_url, _short_err(e)))


def check_ollama(cfg: BrainConfig) -> Check:
    t0 = time.time()
    try:
        data = _http_json(cfg.ollama_url.rstrip("/") + "/api/tags")
        names = [m.get("name", "") for m in data.get("models", [])]
        ms = (time.time() - t0) * 1000
        if any(cfg.local_model in n for n in names):
            return Check("ollama", "ok",
                         "%s ready (%.0fms)" % (cfg.local_model, ms))
        return Check("ollama", "warn",
                     "up but %s not loaded" % cfg.local_model)
    except Exception as e:
        return Check("ollama", "warn",
                     "%s unreachable — local tier dead (%s)"
                     % (cfg.ollama_url, _short_err(e)))


def check_cloud(cfg: BrainConfig) -> Check:
    have = [n for n, k in (("openrouter", BrainConfig.openrouter_key()),
                           ("nvidia", BrainConfig.nvidia_key())) if k]
    if have:
        return Check("cloud", "ok", "keys: %s" % ",".join(have))
    return Check("cloud", "warn", "no keys — oracle tier disabled")


def check_datasets(cfg: BrainConfig) -> Check:
    d = os.path.join(os.path.abspath(cfg.data_dir), "datasets")
    ledger = os.path.join(d, "ledger.jsonl")
    if not os.path.exists(ledger):
        return Check("datasets", "warn", "no releases yet")
    last = None
    try:
        with open(ledger) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if "merkle_root" in e:  # a release, not a merge record
                    last = e
    except OSError as e:
        return Check("datasets", "warn", "ledger unreadable: %s" % e)
    if not last:
        return Check("datasets", "warn", "no releases yet")
    tb = os.path.join(d, last.get("tarball", ""))
    if not os.path.exists(tb):
        return Check("datasets", "warn",
                     "ledger says v%s but tarball missing"
                     % last.get("version"))
    ok, reasons = DatasetStore(d).verify_tarball(tb)
    if not ok:
        return Check("datasets", "warn",
                     "LATEST RELEASE FAILS VERIFY: %s" % "; ".join(reasons))
    return Check("datasets", "ok",
                 "%s v%s verified, signed %s"
                 % (last.get("dataset"), last.get("version"),
                    last.get("fingerprint")))


def check_topics(cfg: BrainConfig) -> Check:
    base = os.path.abspath(cfg.data_dir)
    tp = os.path.join(base, "topics.txt")
    if not os.path.exists(tp):
        return Check("topics", "warn", "no topics.txt")
    queued = sum(1 for l in open(tp, encoding="utf-8")
                 if l.strip() and not l.startswith("#"))
    log = os.path.join(base, "researcher_log.sqlite3")
    done = 0
    if os.path.exists(log):
        try:
            con = sqlite3.connect(log)
            done = con.execute("SELECT COUNT(*) FROM log WHERE"
                               " finished IS NOT NULL").fetchone()[0]
            con.close()
        except sqlite3.Error:
            pass
    return Check("topics", "ok", "%d queued, %d complete"
                 % (queued, done))


def _short_err(e: Exception) -> str:
    s = str(e)
    return s[:60] + ("..." if len(s) > 60 else "")


def check_reconcile(cfg: BrainConfig, embeddings_ok: bool) -> Check:
    """Heal the split-brain: rows learned while the embedding server
    was down live in memory.sqlite3 (TF-IDF fallback); fold any that
    are missing from knowledge.sqlite3 back into the vector store.
    Embeddings are recomputed locally."""
    mem_path = cfg.memory_path()
    if not os.path.exists(mem_path):
        return Check("reconcile", "ok", "stores unified")
    if not embeddings_ok:
        return Check("reconcile", "warn",
                     "deferred — embedding server down")
    from .merge import record_id
    from .vectormem import VectorMemory
    try:
        kv = VectorMemory(cfg.knowledge_path(), cfg.embed_url)
        known = {record_id(q) for q, in
                 kv.db.execute("SELECT question FROM qa")}
        con = sqlite3.connect(mem_path)
        moved = 0
        for q, a, s in con.execute(
                "SELECT question, answer, score FROM qa"):
            if record_id(q) not in known:
                kv.upsert(q, a, s)
                moved += 1
        con.close()
        kv.close()
        if moved:
            return Check("reconcile", "ok",
                         "healed: %d rows memory.sqlite3 -> vector"
                         % moved)
        return Check("reconcile", "ok", "stores unified")
    except Exception as e:
        return Check("reconcile", "warn", "heal failed: %s"
                     % _short_err(e))


def boot(cfg: BrainConfig | None = None) -> BootReport:
    """Run every check. Returns the report; never raises."""
    cfg = cfg or BrainConfig()
    t0 = time.time()
    first_run = not signing.identity_exists(cfg.data_dir)
    checks = []
    for fn in (check_config, check_persona, check_aiml, check_identity,
               check_knowledge,
               check_embeddings, check_ollama, check_cloud,
               check_datasets, check_topics):
        try:
            checks.append(fn(cfg))
        except Exception as e:  # a check must never kill the boot
            checks.append(Check(fn.__name__.replace("check_", ""),
                                "fail", "check crashed: %s" % e))
    # heal step, after we know the embedding server's state
    emb_ok = next(c.status == "ok" for c in checks
                  if c.name == "embeddings")
    try:
        checks.append(check_reconcile(cfg, emb_ok))
    except Exception as e:
        checks.append(Check("reconcile", "warn",
                            "check crashed: %s" % e))
    notes = []
    if first_run:
        ident_check = next((c for c in checks if c.name == "identity"),
                           None)
        if ident_check is not None and ident_check.status == "ok":
            notes.append(FIRST_RUN_IDENTITY_NOTE)
    return BootReport(checks, time.time() - t0, notes)


def main() -> None:
    report = boot()
    print(report.text)
    for note in report.notes:
        print(note)
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
