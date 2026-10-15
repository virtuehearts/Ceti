"""gopher — the Gopher hole: node connections, BBS, and sharing,
in one text menu. Retro surface, modern trust underneath: every
post signed, every bundle verified, every peer a public key.

The menu is a thin shell over peers / board / share — no duplicate
logic. Pick a number, get the thing. 0 goes back to the prompt.
"""
from __future__ import annotations

import os
import sys

from .board import Board
from .peers import PeerBook


def menu_text() -> str:
    return (
        "CETI // gopher hole\n"
        "1. browse nodes (peers)\n"
        "2. read the board\n"
        "3. shared cassettes\n"
        "4. shared files (bundles)\n"
        "5. post to the board\n"
        "6. chat room\n"
        "7. mail\n"
        "0. back"
    )


def _prompt(text: str) -> str:
    sys.stdout.write(text)
    sys.stdout.flush()
    line = sys.stdin.readline()
    return "" if line == "" else line.strip()


def run_gopher(brain) -> None:
    """Menu loop. Reads from stdin like the REPL; EOF goes back."""
    data_dir = brain.cfg.data_dir
    peers = PeerBook(data_dir)
    board = Board(data_dir)
    rack = brain.rack
    while True:
        print(menu_text())
        choice = _prompt("gopher> ")
        if choice in ("", "0", "q", "quit"):
            return
        if choice == "1":
            plist = peers.list()
            if not plist:
                print("  (no peers yet — /peer add <public key>)")
            for p in plist:
                print(f"  {p['fingerprint']}  {p.get('note', '')}")
        elif choice == "2":
            valid, rejected = board.list()
            if not valid:
                print("  (board empty)")
            for i, p in enumerate(valid, 1):
                print(f"  {i}. [{p['author']['fingerprint']}] "
                      f"{p['subject']}")
            if rejected:
                print(f"  ! {len(rejected)} rejected post(s) "
                      f"(bad signature — not shown)")
        elif choice == "3":
            names = rack.list()
            if not names:
                print("  (no cassettes)")
            for n in names:
                print(f"  {n}")
        elif choice == "4":
            ddir = os.path.join(data_dir, "datasets")
            if not os.path.isdir(ddir):
                print("  (no shared files yet)")
            else:
                items = sorted(os.listdir(ddir))
                bundles = [i for i in items
                           if i.endswith(".tar.gz") or
                           os.path.isdir(os.path.join(ddir, i))]
                if not bundles:
                    print("  (no shared files yet)")
                for b in bundles:
                    print(f"  {b}")
        elif choice == "5":
            if brain.identity is None:
                print("no identity: ecdsa not installed")
                continue
            subject = _prompt("subject: ")
            if not subject:
                continue
            body = _prompt("body: ")
            post = board.post(brain.identity, subject, body)
            print(f"posted (signed as {post['author']['fingerprint']})")
        elif choice == "6":
            room = _prompt("room (lobby): ") or "lobby"
            from . import chat as _chat
            _chat.run_chat(brain, room)
        elif choice == "7":
            from . import mail as _mail
            box = _mail.MailBox(data_dir)
            if brain.identity is None:
                print("no identity: ecdsa not installed")
                continue
            valid, rejected = box.inbox(brain.identity.fingerprint)
            if not valid:
                print("  (inbox empty)")
            for i, m in enumerate(valid, 1):
                print(f"  {i}. [{m['from']['fingerprint']}] "
                      f"{m['subject']}: {m['body'][:60]}")
            if rejected:
                print(f"  ! {len(rejected)} rejected "
                      f"(bad signature — not shown)")
        else:
            print("unknown — pick 0-7")
