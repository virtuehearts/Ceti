#!/bin/bash
# nightly_guard.sh — cron every 15 min, tonight only.
# Restarts the CETI overnight research driver if it died (reboot/OOM).
# No-ops after the 2026-10-13 10:00 EDT deadline. Does NOT touch the
# sovereign-stack watchdog (stays paused) or web.py/bridge.py (stay down).
DEADLINE=1791900000
now=$(date +%s)
[ "$now" -ge "$DEADLINE" ] && exit 0
if [ -f /tmp/ceti_nightly.pid ] && kill -0 "$(cat /tmp/ceti_nightly.pid)" 2>/dev/null; then
    exit 0
fi
# stale pidfile or no driver: start it
rm -f /tmp/ceti_nightly.pid
cd /home/hatch/workspace/ceti || exit 0
# any topics left? ask the researcher log directly — the driver itself
# exits cleanly when the queue is empty, so a stale "all topics covered"
# line in the log must not block a legitimate retry pass.
remaining=$(/home/hatch/workspace/ceti/.venv/bin/python -c "
import sqlite3, os
data = '/home/hatch/workspace/ceti/data'
topics = [l.strip() for l in open(data + '/topics.txt') if l.strip() and not l.strip().startswith('#')]
db = sqlite3.connect(data + '/researcher_log.sqlite3')
done = {r[0] for r in db.execute('SELECT topic FROM log WHERE finished IS NOT NULL')}
print(sum(1 for t in topics if t not in done))" 2>/dev/null)
[ "${remaining:-0}" -eq 0 ] && exit 0
nohup /home/hatch/workspace/ceti/.venv/bin/python \
    /home/hatch/workspace/ceti/nightly_driver.py \
    >>/home/hatch/workspace/ceti/data/nightly.log 2>&1 &
