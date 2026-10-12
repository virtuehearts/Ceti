"""Interactive REPL for the brain framework.

    python -m brain.cli

Commands: /stats  /trace  /tiers  /heartbeat  /tasks  /agent
          /identity  /benchmark  /quit

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
    # first run: seed a weekly self-benchmark to prove schedules work
    if not agent.tasks.pending() and not agent.tasks.get_meta("seeded"):
        agent.tasks.add("benchmark", "weekly self-exam", {},
                        delay_s=7 * 24 * 3600, interval_s=7 * 24 * 3600)
        agent.tasks.set_meta("seeded", "1")
    hb = {"running": False, "result": None, "enabled": True}
    show_trace = False
    print("brain 0.1.0 — %s at the prompt. type a question."
          % agent.name)
    print("/stats /trace /tiers /heartbeat /tasks /agent /identity"
          " /benchmark /quit"
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
            tag = f"[{ans.tier}{' +learned' if ans.learned else ''}]"
            print(f"{tag} ({ans.latency_ms}ms)")
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
