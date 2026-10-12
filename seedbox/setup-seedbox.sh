#!/bin/bash
# seedbox setup for the brain's signed datasets — AlmaLinux.
# Run as root on the VPS. Installs transmission-daemon, opens the
# peer port, and points it at /var/lib/brain-seeds.
#
# Then copy this box's data/datasets/*.tar.gz + *.torrent into
# /var/lib/brain-seeds and add them (see README).
set -euo pipefail

SEED_DIR="/var/lib/brain-seeds"
PEER_PORT=51413

dnf install -y transmission-daemon firewalld
mkdir -p "$SEED_DIR"
chown -R transmission:transmission "$SEED_DIR"
chmod 755 "$SEED_DIR"

systemctl stop transmission-daemon || true
# settings.json is rewritten by the daemon; edit while stopped
CFG=/var/lib/transmission/.config/transmission-daemon/settings.json
python3 - "$CFG" <<'EOF'
import json, sys
p = sys.argv[1]
s = json.load(open(p))
s.update({
    "download-dir": "/var/lib/brain-seeds",
    "incomplete-dir-enabled": False,
    "peer-port": 51413,
    "port-forwarding-enabled": True,
    "dht-enabled": True,
    "pex-enabled": True,
    "lpd-enabled": False,
    "rpc-enabled": True,
    "rpc-bind-address": "127.0.0.1",
    "rpc-whitelist-enabled": True,
    "rpc-whitelist": "127.0.0.1",
    "rpc-authentication-required": False,
    "umask": 2,
    "watch-dir-enabled": True,
    "watch-dir": "/var/lib/brain-seeds",
})
json.dump(s, open(p, "w"), indent=2)
print("settings updated")
EOF

firewall-cmd --permanent --add-port=${PEER_PORT}/tcp
firewall-cmd --permanent --add-port=${PEER_PORT}/udp
firewall-cmd --reload

systemctl enable --now transmission-daemon
systemctl enable --now firewalld 2>/dev/null || true
echo "seedbox ready: drop .torrent files into $SEED_DIR (watch-dir picks them up)"
