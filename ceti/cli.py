"""Interactive REPL for the brain framework.

    python -m brain.cli

Commands: /stats  /trace  /tiers  /heartbeat  /tasks  /agent
          /identity  /benchmark  /tools  /task  /skills  /ingest
          /story  /quit  /tts

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


def handle_tts_command(arg: str) -> None:
    """REPL /tts: speak text via the PocketTTS bridge (tool `tts`).

    Prints the saved path + byte count + duration on success, or a
    loud TTS FAILED line — never silence passed off as speech.
    """
    from . import tts as _tts
    from . import tools as _tools
    try:
        text, voice = _tts.parse_command(arg)
    except _tts.TTSError as e:
        print(f"TTS FAILED: {e}")
        return
    try:
        print(_tools.call("tts", text=text, voice=voice))
    except Exception as e:
        print(f"TTS FAILED: {e}")


# ---- /web: the ChatGPT-like web UI ---------------------------------------
# Runs in a daemon thread with its OWN Brain — never the REPL's
# (sqlite handles are thread-affine). State lives here so
# "/web stop" with no server running is a message, not a crash.
_web_state: dict = {"server": None, "thread": None,
                    "host": None, "port": None}


def handle_web_command(arg: str) -> None:
    """/web        start the web UI (daemon thread, own Brain)
    /web stop   stop it again."""
    from . import web as _web
    arg = (arg or "").strip().lower()
    if arg == "stop":
        srv = _web_state["server"]
        if srv is None:
            print("web UI is not running.")
            return
        try:
            srv.shutdown()
        except Exception as e:
            print(f"web UI stop hit an error: {e}")
        th = _web_state["thread"]
        if th is not None:
            th.join(timeout=10)
        _web_state.update(server=None, thread=None,
                           host=None, port=None)
        print("web UI stopped.")
        return
    if arg:
        print("usage: /web | /web stop")
        return
    if _web_state["server"] is not None:
        print(f"web UI already running at "
              f"http://{_web_state['host']}:{_web_state['port']}")
        return
    host, port = _web.DEFAULT_HOST, _web.DEFAULT_PORT
    try:
        server = _web.make_server(host, port)
    except OSError as e:
        print(f"web UI failed to bind {host}:{port}: {e}")
        return
    except Exception as e:
        print(f"web UI failed to start: {type(e).__name__}: {e}")
        return
    import threading as _threading
    th = _threading.Thread(target=server.serve_forever,
                           daemon=True, name="ceti-web")
    th.start()
    _web_state.update(server=server, thread=th, host=host, port=port)
    print(f"CETI web UI: http://{host}:{port}")
    print("(runs in a background thread with its own brain; "
          "the REPL stays live. /web stop to shut it down.)")


def main() -> int:
    from . import colour as _colour
    from . import help as _help
    # BBS startup: clear, banner, self-test, /help, then the prompt.
    # The TUI never auto-loads — /tui enters it explicitly.
    _colour.clear_screen()
    print(_colour.paint(_colour.BANNER, _colour.GREEN))
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
    print(_help.HELP_TEXT)
    print("brain 0.3.3 — %s at the prompt. type a question."
          % agent.name)
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
            if line == "/tts" or line.startswith("/tts "):
                handle_tts_command(line[4:].strip())
                continue
            if line == "/web" or line.startswith("/web "):
                handle_web_command(line[4:].strip())
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
            if line == "/oracle" or line.startswith("/oracle "):
                q = line[len("/oracle "):].strip()
                if not q:
                    print("usage: /oracle <question>")
                    continue
                # The oracle is the cloud tier with the whole ladder
                # skipped: no calc, no AIML, no memory, no local.
                # PII is still stripped before anything leaves the box.
                if not any(getattr(b, "api_key", None)
                           for b in brain.clouds):
                    print("oracle needs a cloud backend with an API "
                          "key (OPENROUTER_API_KEY or NVIDIA_API_KEY).")
                    continue
                ans = brain.ask(q, force_tier="cloud")
                print(ans.display_tag)
                print(ans.text)
                if show_trace:
                    print(json.dumps(ans.trace, indent=1)[:2000])
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
            if line.startswith("/bus"):
                # BIOS bus opcodes: /bus post | read <q> | compile <topic>
                from . import bus as _bus
                from . import ingest as _ing
                _parts = line.split(None, 2)
                _cmd = _parts[1] if len(_parts) > 1 else ""
                _arg = _parts[2] if len(_parts) > 2 else ""
                try:
                    _lib = _ing.Library(brain.cfg.data_dir)
                except Exception:
                    _lib = None
                _b = _bus.MemoryBus.from_brain(brain, library=_lib)
                try:
                    if _cmd == "post":
                        print(json.dumps(_b.post(), indent=1))
                    elif _cmd == "read":
                        if not _arg:
                            print("usage: /bus read <question>")
                        else:
                            _r = _b.read(_arg)
                            for _f in _r.frames:
                                print(f"  [{_f.store}] { _f.text[:300]}")
                            for _g in _r.gaps:
                                print(f"  [gap] {_g.question}")
                            if not _r.frames:
                                print("  (miss — gap logged)")
                    elif _cmd == "compile":
                        if not _arg:
                            print("usage: /bus compile <topic>")
                        else:
                            _c = _b.compile(_arg, intent="article")
                            print(f"  local coverage: "
                                  f"{_c.local_coverage:.0%} "
                                  f"({len(_c.subquestions)} sub-questions)")
                            print(f"  local frames: {len(_c.frames)}  "
                                  f"gaps: {len(_c.gaps)}")
                            for _g in _c.gaps:
                                print(f"  [gap] {_g.question}")
                    else:
                        print("usage: /bus post | read <question> | "
                              "compile <topic>")
                finally:
                    if _lib is not None:
                        _lib.close()
                continue
            if line.startswith("/story"):
                from . import story as _st
                parts = line.split(None, 2)
                cmd = parts[1] if len(parts) > 1 else ""
                arg = parts[2] if len(parts) > 2 else ""
                if cmd not in ("new", "write", "status", "compile",
                               "list", "article"):
                    print("usage: /story new <premise> | write <id> "
                          "[pages] [count] | status <id> | compile <id> "
                          "| list | article <topic> [q1; q2; ...]")
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
                # article is compile-first via the bus: backend is
                # optional — with no oracle it writes local-only.
                if cmd == "article":
                    try:
                        from . import bus as _bus
                        _b = _bus.MemoryBus.from_brain(
                            brain, library=_lib)
                        _ac = _st.ArticleCompiler(
                            brain.cfg, _b, backend=backend)
                        # "Topic | q1; q2; ..." — sub-questions after |
                        _topic, _, _qs = arg.partition("|")
                        _subs = [s.strip() for s in _qs.split(";")
                                 if s.strip()] or None
                        _r = _ac.compile_article(_topic.strip(),
                                                 subquestions=_subs)
                        print(f"article {_r['topic']}: "
                              f"{_r['local_coverage']:.0%} local, "
                              f"{_r['gaps_filled']} gaps filled, "
                              f"{_r['gaps_open']} open, "
                              f"{_r['words']} words -> {_r['md']}")
                    except (_st.StoryError, ValueError) as e:
                        print(f"story error: {e}")
                    finally:
                        _lib.close()
                    continue
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
            if line == "/help":
                from . import help as _help
                print(_help.HELP_TEXT)
                continue
            if line == "/research" or line.startswith("/research "):
                from . import researcher as _res
                _r = _res.Researcher(brain)
                parts = line.split(None, 2)
                sub = parts[1] if len(parts) > 1 else ""
                arg = parts[2] if len(parts) > 2 else ""
                try:
                    if sub == "":
                        st = _r.status()
                        print(f"research sites: "
                              f"{', '.join(st['sites'])}")
                        print(f"topics queued: {len(_r.topics())} "
                              f"({', '.join(_r.topics()[:5])}"
                              f"{'...' if len(_r.topics()) > 5 else ''})")
                        print(f"interests: "
                              f"{', '.join(st['interests']) or '(none)'}")
                        nb = st["nightly_budget"]
                        print(f"nightly budget: {nb['max_topics']} "
                              f"topics, {nb['max_qa']} Q&A, "
                              f"{nb['max_minutes']} min")
                        print(f"completed: {len(st['completed'])} "
                              f"topics, {st['qa_total']} Q&A recorded")
                    elif sub == "sites":
                        sp = arg.split(None, 1)
                        if not sp:
                            for s in _r.config.sites:
                                print(f"  {s}")
                        elif sp[0] == "add" and len(sp) > 1:
                            print(f"site added: "
                                  f"{_r.config.add_site(sp[1])}")
                        elif sp[0] == "remove" and len(sp) > 1:
                            print(f"site removed: "
                                  f"{_r.config.remove_site(sp[1])}")
                        else:
                            print("usage: /research sites "
                                  "[add|remove <domain>]")
                    elif sub == "topics":
                        sp = arg.split(None, 1)
                        if not sp:
                            for t in _r.topics():
                                print(f"  {t}")
                        elif sp[0] == "add" and len(sp) > 1:
                            print(f"topic added: "
                                  f"{_r.add_topic(sp[1])}")
                        elif sp[0] == "remove" and len(sp) > 1:
                            print(f"topic removed: "
                                  f"{_r.remove_topic(sp[1])}")
                        else:
                            print("usage: /research topics "
                                  "[add|remove <topic>]")
                    elif sub == "interests":
                        sp = arg.split(None, 1)
                        if not sp:
                            for i in _r.config.interests:
                                print(f"  {i}")
                            if not _r.config.interests:
                                print("  (none)")
                        elif sp[0] == "add" and len(sp) > 1:
                            print(f"interest added: "
                                  f"{_r.config.add_interest(sp[1])}")
                        elif sp[0] == "remove" and len(sp) > 1:
                            print(f"interest removed: "
                                  f"{_r.config.remove_interest(sp[1])}")
                        else:
                            print("usage: /research interests "
                                  "[add|remove <text>]")
                    elif sub == "budget":
                        sp = arg.split()
                        if not sp:
                            nb = _r.config.nightly
                            print(f"  max_topics={nb['max_topics']} "
                                  f"max_qa={nb['max_qa']} "
                                  f"max_minutes={nb['max_minutes']}")
                        else:
                            kv = {}
                            for i in range(0, len(sp) - 1, 2):
                                try:
                                    kv[sp[i]] = int(sp[i + 1])
                                except ValueError:
                                    print(f"not a number: {sp[i + 1]}")
                                    kv = None
                                    break
                            if kv is not None:
                                nb = _r.config.set_budget(**kv)
                                print(f"nightly budget: "
                                      f"{nb['max_topics']} topics, "
                                      f"{nb['max_qa']} Q&A, "
                                      f"{nb['max_minutes']} min")
                    else:
                        print("usage: /research [sites|topics|"
                              "interests|budget]")
                except ValueError as e:
                    print(f"research error: {e}")
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
            if line == "/evals" or line.startswith("/evals "):
                from . import evals as _ev
                _suite = line.split(None, 1)[1].strip() \
                    if len(line.split(None, 1)) > 1 else "all"
                if _suite not in ("all", "knowledge", "coding", "tool"):
                    print("usage: /evals [all|knowledge|coding|tool]")
                    continue
                print("evals running (%s) — coding hits the local "
                      "model, this takes a bit..." % _suite)
                _runner = _ev.EvalRunner(brain, gap_log=agent.tasks)
                _rep = _runner.run(None if _suite == "all" else _suite)
                for _name, _d in _rep["suites"].items():
                    print(f"  {_name}: {_d['passed']}/{_d['total']} "
                          f"({ _d['accuracy']:.0%})")
                print(f"  overall: {_rep['passed']}/{_rep['total']} "
                      f"({ _rep['accuracy']:.0%}) in "
                      f"{_rep['elapsed_s']}s")
                for _r in _rep["results"]:
                    _mark = "PASS" if _r["passed"] else "FAIL"
                    print(f"  [{_mark}] {_r['id']}: {_r['detail'][:160]}")
                if _rep["misses"]:
                    print("  misses filed as gaps (curiosity feed).")
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
