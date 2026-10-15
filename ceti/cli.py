"""Interactive REPL for the brain framework.

    python -m brain.cli

Commands: /stats  /trace  /tiers  /heartbeat  /tasks  /agent
          /identity  /benchmark  /tools  /task  /skills  /ingest
          /story  /quit

On start the BIOS POST runs (brain.boot): every subsystem is loaded
and verified before the prompt appears. If you go quiet for
IDLE_HEARTBEAT_S, a heartbeat fires — one agent tick runs in a
background thread (its own Brain, its own sqlite handles) while the
prompt stays live. Results are delivered before your next prompt.
Failed questions are logged as gaps; the agent turns repeated gaps
into curiosity tasks and goes to learn them.
"""
from __future__ import annotations

import json
import os
import select
import sys
import threading

from .boot import boot
from .brain import Brain

IDLE_HEARTBEAT_S = 600  # 10 quiet minutes -> one research cycle


def _real_cycle() -> dict:
    from .agent import Agent
    b = Brain()  # own connections: never share sqlite across threads
    try:
        a = Agent(b)  # name persists in tasks.db meta
        try:
            return a.tick()
        finally:
            a.close()
    finally:
        b.close()


def heartbeat_tick(hb: dict, cycle_fn=_real_cycle) -> None:
    """Fire one background research cycle. Main thread only spawns;
    the worker never touches shared handles. cycle_fn is injectable
    for tests."""
    if hb["running"] or not hb["enabled"]:
        return
    hb["running"] = True
    print("\n[heartbeat] quiet for %ds — research cycle in background"
          % IDLE_HEARTBEAT_S)

    def work() -> None:
        try:
            res = cycle_fn()
        except Exception as e:  # heartbeat must never kill the REPL
            res = {"error": "%s: %s" % (type(e).__name__, e)}
        hb["result"] = res
        hb["running"] = False

    threading.Thread(target=work, daemon=True).start()


def main() -> int:
    report = boot()
    print(report.text)
    for note in report.notes:
        print(note)
    if not report.ok:
        print("core failure — not starting.")
        return 1

    from .agent import Agent
    brain = Brain()
    agent = Agent(brain)
    brain.gap_log = agent.tasks  # failed questions -> curiosity feed
    # skills: hot-loadable bundles, exposed as list_skills/read_skill
    from . import skills as _skills
    _skill_loader = _skills.ensure_tools_registered()
    for err in _skill_loader.errors:
        print(f"[ warn ] skill  {err}")
    if _skill_loader.skills:
        print(f"[ ok ] skills   {len(_skill_loader.skills)} loaded: "
              f"{', '.join(sorted(_skill_loader.skills))}")
    # first run: seed a weekly self-benchmark to prove schedules work
    if not agent.tasks.pending() and not agent.tasks.get_meta("seeded"):
        agent.tasks.add("benchmark", "weekly self-exam", {},
                        delay_s=7 * 24 * 3600, interval_s=7 * 24 * 3600)
        agent.tasks.set_meta("seeded", "1")
    hb = {"running": False, "result": None, "enabled": True}
    show_trace = False
    print("brain 0.3.0 — %s at the prompt. type a question."
          % agent.name)
    print("/stats /trace /tiers /heartbeat /tasks /agent /identity"
          " /benchmark /tools /task /skills /ingest /story"
          " /gopher /peers /peer /board /share /fetch"
          " /nodelist /gossip /chat /mail /upload /download /tui /quit"
          " — heartbeat: %s" % ("on" if hb["enabled"] else "off"))
    try:
        while True:
            if hb["result"] is not None:
                res = hb["result"]
                hb["result"] = None
                print("\n[heartbeat] agent tick finished:")
                print(json.dumps(res, indent=1)[:1500])
                try:
                    brain.reload_cassettes()
                except Exception:
                    pass
            sys.stdout.write("\nyou> ")
            sys.stdout.flush()
            timeout = IDLE_HEARTBEAT_S if hb["enabled"] else None
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if not ready:
                heartbeat_tick(hb)
                continue
            line = sys.stdin.readline()
            if line == "":
                break  # EOF
            line = line.strip()
            if not line:
                continue
            if line == "/quit":
                break
            if line == "/tools":
                from . import tools as _tools
                for t in _tools.manifest():
                    sb = "sandbox" if t["sandboxed"] else "host"
                    print(f"  {t['name']:12s} [{t['side_effects']:4s}/{sb:7s}] {t['description']}")
                print(f"  work root: {_tools.work_root()}")
                print("  invoke: /call <tool> key=value ... (\\n in values becomes a newline)")
                continue
            if line.startswith("/call "):
                import shlex as _shlex
                from . import tools as _tools
                try:
                    parts = _shlex.split(line[6:])
                    name = parts[0]
                    kwargs = {}
                    for p in parts[1:]:
                        k, v = p.split("=", 1)
                        kwargs[k] = v.replace("\\n", "\n")
                    print(_tools.call(name, **kwargs))
                except (ValueError, IndexError):
                    print("usage: /call <tool> [key=value ...]")
                except Exception as e:
                    print(f"tool error: {e}")
                continue
            if line.startswith("/task "):
                task = line[6:].strip()
                if not task:
                    print("usage: /task <what the agent should do>")
                    continue
                backend = next(
                    (b for b in brain.clouds
                     if getattr(b, "api_key", None)
                     and hasattr(b, "complete_with_tools")), None)
                if backend is None:
                    print("agent loop needs a cloud backend with an API "
                          "key (OPENROUTER_API_KEY or NVIDIA_API_KEY).")
                    continue
                from . import agent_loop as _al
                try:
                    sess = _al.AgentSession(brain.cfg, backend)
                except _al.AgentError as e:
                    print(f"agent error: {e}")
                    continue
                print(f"[agent] session {sess.session_id} "
                      f"({getattr(backend, 'name', '?')})")
                try:
                    print(sess.run(task))
                finally:
                    sess.close()
                continue
            if line == "/skills" or line.startswith("/skills "):
                from . import skills as _skills
                _loader = _skills.get_loader()
                arg = line[7:].strip()
                if arg == "reload":
                    _loader.reload()
                    _skills.ensure_tools_registered()
                    print(f"reloaded: {len(_loader.skills)} skills")
                for _name in sorted(_loader.skills):
                    _s = _loader.skills[_name]
                    print(f"  {_name:16s} {_s.description}")
                for _err in _loader.errors:
                    print(f"  ! {_err}")
                if not _loader.skills:
                    print(f"  (none — add bundles to {_loader.dir})")
                continue
            if line == "/ingest":
                from . import ingest as _ing
                try:
                    _lib = _ing.Library(brain.cfg.data_dir)
                except Exception as e:
                    print(f"ingest error: {e}")
                    continue
                if not _lib.ping():
                    print("ingest: embedding server unreachable "
                          "(http://127.0.0.1:11435) — start llama-server")
                    _lib.close()
                    continue
                print(f"ingest: scanning {_lib.libdir}")
                for r in _lib.scan():
                    if r["status"] == "ingested":
                        print(f"  + {r['filename']}: {r['chunks']} "
                              f"chunks, {r['chars']} chars")
                    elif r["status"] == "already-ingested":
                        print(f"  = {r['filename']}: already ingested")
                    else:
                        print(f"  ! {r['filename']}: "
                              f"{r.get('error', r['status'])}")
                print(f"  library: {_lib.count_chunks()} chunks total")
                _lib.close()
                continue
            if line.startswith("/story"):
                from . import story as _st
                parts = line.split(None, 2)
                cmd = parts[1] if len(parts) > 1 else ""
                arg = parts[2] if len(parts) > 2 else ""
                if cmd not in ("new", "write", "status", "compile",
                               "list"):
                    print("usage: /story new <premise> | write <id> "
                          "[pages] [count] | status <id> | compile <id> "
                          "| list")
                    continue
                backend = next(
                    (b for b in brain.clouds
                     if getattr(b, "api_key", None)), None)
                if cmd in ("new", "write") and backend is None:
                    print("story writer needs a cloud backend with an "
                          "API key (OPENROUTER_API_KEY or NVIDIA_API_KEY).")
                    continue
                from . import ingest as _ing
                _lib = _ing.Library(brain.cfg.data_dir)
                _w = _st.StoryWriter(brain.cfg, backend, library=_lib)
                try:
                    if cmd == "list":
                        for s in _w.list_stories():
                            st = _w.status(s)
                            print(f"  {s}: {st['blocks']} blocks, "
                                  f"{st['words']} words "
                                  f"({st['pages']} pages)")
                        if not _w.list_stories():
                            print("  (no stories yet)")
                    elif cmd == "new":
                        if not arg:
                            print("usage: /story new <premise>")
                        else:
                            sid = _w.new_story(arg)
                            print(f"story {sid} — bible written")
                    elif cmd == "status":
                        st = _w.status(arg)
                        print(f"  {st['story_id']}: {st['blocks']} "
                              f"blocks, {st['words']} words "
                              f"({st['pages']} pages)")
                    elif cmd == "compile":
                        r = _w.compile(arg)
                        print(f"compiled {r['chapters']} chapters, "
                              f"{r['words']} words -> {r['md']}")
                    elif cmd == "write":
                        sp = arg.split()
                        if not sp:
                            print("usage: /story write <id> [pages] "
                                  "[count]")
                        else:
                            sid, pages = sp[0], int(sp[1]) if len(sp) > 1 else 3
                            count = int(sp[2]) if len(sp) > 2 else 1
                            for _ in range(count):
                                r = _w.next_block(sid, pages)
                                print(f"  chapter {r['block']}: "
                                      f"{r['words']} words -> {r['path']}")
                except (_st.StoryError, ValueError) as e:
                    print(f"story error: {e}")
                finally:
                    _lib.close()
                continue
            if line == "/gopher":
                from . import gopher as _gopher
                _gopher.run_gopher(brain)
                continue
            if line == "/peers":
                from . import peers as _peers
                plist = _peers.PeerBook(brain.cfg.data_dir).list()
                if not plist:
                    print("  (no peers yet — /peer add <public key>)")
                for p in plist:
                    print(f"  {p['fingerprint']}  {p.get('note', '')}")
                continue
            if line.startswith("/peer "):
                from . import peers as _peers
                parts = line.split(None, 2)
                cmd = parts[1] if len(parts) > 1 else ""
                arg = parts[2] if len(parts) > 2 else ""
                book = _peers.PeerBook(brain.cfg.data_dir)
                if cmd == "card":
                    if brain.identity is None:
                        print("no identity: ecdsa not installed")
                    else:
                        print(json.dumps(
                            _peers.identity_card(brain.identity),
                            indent=1))
                elif cmd == "add" and arg:
                    try:
                        p = book.add(arg.split()[0])
                        print(f"peer filed: {p['fingerprint']} "
                              f"({p['node_id']})")
                    except ValueError as e:
                        print(f"peer error: {e}")
                else:
                    print("usage: /peer card | add <public key hex>")
                continue
            if line.startswith("/board"):
                from . import board as _board
                parts = line.split(None, 2)
                cmd = parts[1] if len(parts) > 1 else ""
                arg = parts[2] if len(parts) > 2 else ""
                bd = _board.Board(brain.cfg.data_dir)
                if cmd == "":
                    valid, rejected = bd.list()
                    if not valid:
                        print("  (board empty)")
                    for i, p in enumerate(valid, 1):
                        print(f"  {i}. [{p['author']['fingerprint']}] "
                              f"{p['subject']}")
                    if rejected:
                        print(f"  ! {len(rejected)} rejected post(s) "
                              f"(bad signature — not shown)")
                elif cmd == "read" and arg:
                    try:
                        p = bd.read(int(arg))
                    except ValueError:
                        p = None
                    if p is None:
                        print("no such post")
                    else:
                        print(f"[{p['author']['fingerprint']}] "
                              f"{p['subject']}")
                        print(p["body"])
                elif cmd == "post" and arg:
                    if brain.identity is None:
                        print("no identity: ecdsa not installed")
                    elif " | " not in arg:
                        print("usage: /board post <subject> | <body>")
                    else:
                        subject, body = arg.split(" | ", 1)
                        post = bd.post(brain.identity, subject, body)
                        print(f"posted (signed as "
                              f"{post['author']['fingerprint']})")
                elif cmd == "import" and arg:
                    try:
                        post = bd.import_post(arg)
                        print(f"imported post from "
                              f"{post['author']['fingerprint']}: "
                              f"{post['subject']}")
                    except (OSError, ValueError) as e:
                        print(f"import refused: {e}")
                else:
                    print("usage: /board | read <n> | post <subject> "
                          "| <body> | import <file>")
                continue
            if line.startswith("/share "):
                from . import share as _share
                from .datasets import DatasetStore
                path = line[7:].strip()
                if not os.path.exists(path):
                    print(f"no such file: {path}")
                else:
                    store = DatasetStore(os.path.join(
                        brain.cfg.data_dir, "datasets"))
                    try:
                        _m, _b, tb = _share.share(
                            store, [path], identity=brain.identity)
                        print(f"shared -> {tb}")
                    except Exception as e:
                        print(f"share error: {e}")
                continue
            if line.startswith("/fetch "):
                from . import share as _share
                from .datasets import DatasetStore
                tb = line[7:].strip()
                store = DatasetStore(os.path.join(
                    brain.cfg.data_dir, "datasets"))
                try:
                    rep = _share.fetch(store, brain.memory,
                                       brain.aiml, tb)
                    print(f"fetched from {rep.get('peer')}: "
                          f"+{rep['qa_added']} Q&A, "
                          f"+{rep['aiml_added']} AIML, "
                          f"{rep['conflicts']} conflicts")
                except Exception as e:
                    print(f"fetch refused: {e}")
                continue
            if line.startswith("/nodelist"):
                from . import nodelist as _nl
                from . import peers as _peers
                parts = line.split(None, 2)
                cmd = parts[1] if len(parts) > 1 else ""
                arg = parts[2] if len(parts) > 2 else ""
                book = _peers.PeerBook(brain.cfg.data_dir)
                if cmd == "build":
                    if brain.identity is None:
                        print("no identity: ecdsa not installed")
                    else:
                        doc = _nl.build_nodelist(book.list(),
                                                 brain.identity)
                        path = os.path.join(brain.cfg.data_dir,
                                            "nodelist.json")
                        _nl.save_nodelist(doc, path)
                        print(f"nodelist: {len(doc['nodes'])} nodes, "
                              f"signed as "
                              f"{doc['signer']['fingerprint']} -> {path}")
                elif cmd == "fetch" and arg:
                    try:
                        doc = _nl.fetch_nodelist(arg)
                        rep = _nl.merge_nodelist(book, doc)
                        print(f"nodelist merged: +{rep['added']} new, "
                              f"{rep['known']} known")
                    except (OSError, ValueError) as e:
                        print(f"nodelist refused: {e}")
                elif cmd == "show":
                    plist = book.list()
                    if not plist:
                        print("  (no peers yet)")
                    for p in plist:
                        addr = f"  {p['address']}" if p.get("address") \
                            else ""
                        print(f"  {p['fingerprint']}{addr}")
                else:
                    print("usage: /nodelist build | fetch <url|file> "
                          "| show")
                continue
            if line.startswith("/gossip"):
                from . import nodelist as _nl
                from . import peers as _peers
                parts = line.split(None, 2)
                cmd = parts[1] if len(parts) > 1 else ""
                arg = parts[2] if len(parts) > 2 else ""
                book = _peers.PeerBook(brain.cfg.data_dir)
                if cmd == "export":
                    if brain.identity is None:
                        print("no identity: ecdsa not installed")
                    else:
                        doc = _nl.build_nodelist(book.list(),
                                                 brain.identity)
                        path = os.path.join(
                            brain.cfg.data_dir, "drop",
                            "gossip-%s.json"
                            % brain.identity.fingerprint)
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        _nl.save_nodelist(doc, path)
                        print(f"gossip exported: {len(doc['nodes'])} "
                              f"peers -> {path}")
                elif cmd == "import" and arg:
                    try:
                        doc = _nl.fetch_nodelist(arg)
                        rep = _nl.merge_nodelist(book, doc)
                        print(f"gossip merged: +{rep['added']} new, "
                              f"{rep['known']} known")
                    except (OSError, ValueError) as e:
                        print(f"gossip refused: {e}")
                else:
                    print("usage: /gossip export | import <file>")
                continue
            if line.startswith("/chat"):
                from . import chat as _chat
                parts = line.split(None, 1)
                room = parts[1].strip() if len(parts) > 1 else "lobby"
                _chat.run_chat(brain, room or "lobby")
                continue
            if line.startswith("/mail"):
                from . import mail as _mail
                from . import peers as _peers
                parts = line.split(None, 2)
                cmd = parts[1] if len(parts) > 1 else ""
                arg = parts[2] if len(parts) > 2 else ""
                box = _mail.MailBox(brain.cfg.data_dir)
                book = _peers.PeerBook(brain.cfg.data_dir)
                if cmd == "":
                    if brain.identity is None:
                        print("no identity: ecdsa not installed")
                    else:
                        valid, rejected = box.inbox(
                            brain.identity.fingerprint)
                        if not valid:
                            print("  (inbox empty)")
                        for i, m in enumerate(valid, 1):
                            print(f"  {i}. [{m['from']['fingerprint']}] "
                                  f"{m['subject']}")
                        if rejected:
                            print(f"  ! {len(rejected)} rejected "
                                  f"(bad signature — not shown)")
                elif cmd == "import" and arg:
                    try:
                        m = box.import_mail(arg)
                        print(f"mail imported from "
                              f"{m['from']['fingerprint']}: "
                              f"{m['subject']}")
                    except (OSError, ValueError) as e:
                        print(f"mail refused: {e}")
                elif cmd and arg and " | " in arg:
                    if brain.identity is None:
                        print("no identity: ecdsa not installed")
                    else:
                        subject, body = arg.split(" | ", 1)
                        try:
                            m = box.send(brain.identity, cmd, subject,
                                         body, book)
                            print(f"mail to {cmd} queued in outbox "
                                  f"(hand the file over)")
                        except ValueError as e:
                            print(f"mail error: {e}")
                else:
                    print("usage: /mail | <fingerprint> <subject> "
                          "| <body> | import <file>")
                continue
            if line.startswith("/upload "):
                import shutil as _shutil
                from . import share as _share
                from .datasets import DatasetStore
                path = line[8:].strip()
                if not os.path.exists(path):
                    print(f"no such file: {path}")
                else:
                    store = DatasetStore(os.path.join(
                        brain.cfg.data_dir, "datasets"))
                    try:
                        _m, _b, tb = _share.share(
                            store, [path], identity=brain.identity)
                        drop = os.path.join(brain.cfg.data_dir, "drop")
                        os.makedirs(drop, exist_ok=True)
                        dest = os.path.join(drop,
                                             os.path.basename(tb))
                        _shutil.copy2(tb, dest)
                        print(f"uploaded -> {dest}")
                    except Exception as e:
                        print(f"upload error: {e}")
                continue
            if line.startswith("/download"):
                from . import share as _share
                from .datasets import DatasetStore
                parts = line.split(None, 1)
                name = parts[1].strip() if len(parts) > 1 else ""
                drop = os.path.join(brain.cfg.data_dir, "drop")
                if not name:
                    if not os.path.isdir(drop):
                        print("  (drop empty)")
                    else:
                        items = sorted(i for i in os.listdir(drop)
                                       if i.endswith(".tar.gz"))
                        if not items:
                            print("  (drop empty)")
                        for i in items:
                            print(f"  {i}")
                    print("usage: /download <file> (from the drop)")
                else:
                    tb = os.path.join(drop, os.path.basename(name))
                    if not os.path.exists(tb):
                        print(f"no such file in drop: {name}")
                    else:
                        store = DatasetStore(os.path.join(
                            brain.cfg.data_dir, "datasets"))
                        try:
                            rep = _share.fetch(store, brain.memory,
                                               brain.aiml, tb)
                            print(f"downloaded from {rep.get('peer')}: "
                                  f"+{rep['qa_added']} Q&A, "
                                  f"+{rep['aiml_added']} AIML, "
                                  f"{rep['conflicts']} conflicts")
                        except Exception as e:
                            print(f"download refused: {e}")
                continue
            if line == "/tui":
                from . import tui as _tui
                _tui.run_tui(brain)
                continue
            if line == "/stats":
                print(json.dumps(brain.stats, indent=1))
                if brain.aiml:
                    print(f"aiml categories: {brain.aiml.count()}")
                if brain.memory:
                    print(f"memory pairs: {brain.memory.count()}")
                continue
            if line == "/trace":
                show_trace = not show_trace
                print(f"trace {'on' if show_trace else 'off'}")
                continue
            if line == "/heartbeat":
                hb["enabled"] = not hb["enabled"]
                print(f"heartbeat {'on' if hb['enabled'] else 'off'}")
                continue
            if line == "/tasks":
                for t in agent.tasks.pending():
                    print(f"  #{t['id']} [{t['kind']}] {t['title']}")
                continue
            if line == "/agent":
                print(json.dumps(agent.status(), indent=1)[:2000])
                continue
            if line == "/identity":
                from . import signing as _signing
                ident = brain.identity
                if ident is None:
                    print("no identity: ecdsa not installed "
                          "(pip install -r requirements.txt)")
                else:
                    print(f"fingerprint: {ident.fingerprint}")
                    print(f"node_id:     {ident.node_id}")
                    print(f"identity file: "
                          f"{_signing.identity_path(brain.cfg.data_dir)}")
                    print(f"created:     {ident.created}")
                continue
            if line == "/benchmark":
                from . import benchmark as _bm
                print("self-exam running...")
                print(json.dumps(
                    _bm.run(brain, store=agent.tasks), indent=1)[:2500])
                continue
            if line == "/tiers":
                from . import decide
                for intent in ("smalltalk", "command", "factual", "chat"):
                    print(f"  {intent:10s} -> {decide.plan(intent, brain.cfg)}")
                continue
            ans = brain.ask(line)
            print(ans.display_tag)
            print(ans.text)
            if show_trace:
                print(json.dumps(ans.trace, indent=1)[:2000])
    except KeyboardInterrupt:
        pass
    finally:
        agent.close()
        brain.close()
    print("\nbye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
