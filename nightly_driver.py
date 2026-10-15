#!/usr/bin/env python3
"""CETI overnight research drain — one topic at a time, services supervised.

Runs the researcher's topic queue into CETI_DATA_DIR cassettes. Bypasses the
idle memory gate deliberately (Jero's order, 2026-10-12): the box is tight on
RAM, so this driver restarts Ollama / the embedding server between topics if
the OOM killer takes them. Stops at DEADLINE or when topics are done.
"""
import os
import sys
import time
import json
import urllib.request
import subprocess

DATA = "/home/hatch/workspace/ceti/data"
LOG = os.path.join(DATA, "nightly.log")
PIDFILE = "/tmp/ceti_nightly.pid"
DEADLINE = 1791900000  # 2026-10-13 10:00 EDT
OLLAMA = "http://127.0.0.1:11434"
EMBED = "http://127.0.0.1:11435"

os.environ["CETI_DATA_DIR"] = DATA
os.environ["OLLAMA_MODELS"] = "/home/hatch/workspace/models/ollama"
# Wikipedia-only fetching tonight: the multi-engine fetcher touches a new
# random host per topic, and each one pops a CONNECT approval card on Jero's
# phone (2026-10-12). en.wikipedia.org is a single already-seen host, and
# these topics are all encyclopedic anyway.
os.environ["CETI_WIKI_ONLY"] = "1"
sys.path.insert(0, "/home/hatch/workspace/ceti")


def log(msg):
    line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def http_ok(url, timeout=10):
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def services_ok():
    return http_ok(OLLAMA + "/api/tags") and http_ok(EMBED + "/v1/models")


def start_services():
    log("starting ollama serve")
    subprocess.Popen(
        ["bash", "-lc",
         "OLLAMA_MODELS=/home/hatch/workspace/models/ollama "
         "OLLAMA_KEEP_ALIVE=24h "
         "nohup /home/hatch/workspace/bin/ollama serve "
         ">/home/hatch/workspace/ceti/data/ollama.log 2>&1 &"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log("starting embedding server")
    subprocess.Popen(
        ["bash", "-lc",
         "nohup /home/hatch/workspace/llama.cpp-build/build/bin/llama-server "
         "-m /home/hatch/workspace/models/gguf/embeddinggemma-2-UD-Q4_K_XL.gguf "
         "--embedding --host 127.0.0.1 --port 11435 -c 8192 -b 4096 -ub 4096 "
         ">/home/hatch/workspace/ceti/data/embed.log 2>&1 &"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ensure_services():
    for _ in range(6):
        if services_ok():
            return True
        start_services()
        time.sleep(20)
    return services_ok()


def mark_skipped(topic, reason):
    import sqlite3
    db = sqlite3.connect(os.path.join(DATA, "researcher_log.sqlite3"))
    now = time.time()
    db.execute(
        "INSERT OR REPLACE INTO log (topic, started, finished, qa_added,"
        " report) VALUES (?,?,?,?,?)",
        (topic, now, now, 0, "skipped: " + reason))
    db.commit()
    db.close()


def main():
    with open(PIDFILE, "w") as f:
        f.write(str(os.getpid()))
    log("nightly driver start")
    if not ensure_services():
        log("FATAL: services would not come up; exiting")
        return 2
    from ceti.brain import Brain
    from ceti.researcher import Researcher
    brain = Brain()
    # Tonight's box sits at ~0.17GB available (host-level pressure, canary
    # 2026-10-12 20:00 EDT); the 1.0GB default gate in research.py would
    # never pass and the drain would burn the night retrying. 0.15GB is
    # enough for the light fetcher; the distill step's qwen load was
    # canary-proven an hour ago. Library default (1.0) untouched.
    r = Researcher(brain, min_mem_gb=0.15)
    done = 0
    fail_streak = {}
    skipped = []
    try:
        while time.time() < DEADLINE:
            if not services_ok():
                log("service down between topics; restarting")
                if not ensure_services():
                    log("services dead, sleeping 10m before retry")
                    time.sleep(600)
                    continue
            topic = r.next_topic()
            if not topic:
                # queue exhausted: one retry pass over skipped topics
                # (rate limits may have cooled down) if time allows
                if skipped and time.time() < DEADLINE - 3600:
                    topic = skipped.pop(0)
                    db = __import__("sqlite3").connect(
                        os.path.join(DATA, "researcher_log.sqlite3"))
                    db.execute("DELETE FROM log WHERE topic=?", (topic,))
                    db.commit()
                    db.close()
                    fail_streak.pop(topic, None)
                    log("RETRY skipped topic: %s" % topic)
                else:
                    if skipped:
                        log("deadline near; %d skipped topics left: %s"
                            % (len(skipped), ", ".join(skipped)))
                    log("all topics covered; done=%d" % done)
                    break
            log("TOPIC START: %s" % topic)
            try:
                res = r.research_topic(topic)
                if isinstance(res, dict) and res.get("failed"):
                    raise RuntimeError(res["failed"])
                qa = res.get("qa_added", res)
                log("TOPIC DONE: %s -> %s" % (topic, json.dumps(qa)[:200]))
                done += 1
                fail_streak.pop(topic, None)
            except Exception as e:
                n = fail_streak.get(topic, 0) + 1
                fail_streak[topic] = n
                log("TOPIC ERROR (%d/3): %s: %r" % (n, topic, e))
                if n >= 3:
                    mark_skipped(topic, "3 consecutive failures: %r" % e)
                    skipped.append(topic)
                    fail_streak.pop(topic, None)
                    log("SKIPPED for now: %s (will retry at end if time)"
                        % topic)
                # back off: hammering a rate-limited engine only
                # makes the limit worse
                time.sleep(90)
        else:
            log("deadline reached; done=%d tonight" % done)
    finally:
        try:
            brain.close()
        except Exception:
            pass
        log("nightly driver exit, done=%d" % done)
        try:
            os.unlink(PIDFILE)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
