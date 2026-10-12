"""seedtest — loopback BitTorrent transfer test (stdlib only).

Spins up a minimal seeder and leecher on 127.0.0.1 and moves the
release tarball across a REAL BitTorrent wire session: handshake,
bitfield, interested/unchoke, request/piece. Then the downloaded copy
is hash-checked and run through the dataset verifier.

This proves the minted .torrent is valid and the data transfers
bit-identically. It does not test internet reachability — nothing
inbound can reach this sandbox, so real swarm seeding happens on the
seedbox (see brain/seedbox/).

Usage:
    python -m brain.seedtest <torrent> <data-file> [--out dir]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import socket
import struct
import sys
import threading

from .seeder import bdecode, bencode

BLOCK = 16384  # standard request block size
PSTR = b"BitTorrent protocol"


def _recvn(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed connection")
        buf += chunk
    return buf


def _send_msg(sock: socket.socket, msg_id: int, payload: bytes) -> None:
    sock.sendall(struct.pack(">I", 1 + len(payload))
                 + bytes([msg_id]) + payload)


def _recv_msg(sock: socket.socket) -> tuple[int | None, bytes]:
    raw = _recvn(sock, 4)
    (ln,) = struct.unpack(">I", raw)
    if ln == 0:
        return None, b""  # keep-alive
    data = _recvn(sock, ln)
    return data[0], data[1:]


def _handshake(sock: socket.socket, info_hash: bytes,
               peer_id: bytes) -> None:
    # peer_id MUST be exactly 20 bytes: a short id silently desyncs
    # the stream (fewer bytes read than sent) and every later message
    # parses as garbage.
    assert len(peer_id) == 20, "peer_id must be 20 bytes, got %d" \
        % len(peer_id)
    assert len(info_hash) == 20
    sock.sendall(bytes([len(PSTR)]) + PSTR + b"\x00" * 8
                 + info_hash + peer_id)
    resp = _recvn(sock, 68)
    if resp[28:48] != info_hash:
        raise ConnectionError("info_hash mismatch in handshake")


class Seeder(threading.Thread):
    """Serves one file's pieces to a single leecher."""

    def __init__(self, torrent_path: str, data_path: str):
        super().__init__(daemon=True)
        with open(torrent_path, "rb") as f:
            t = bdecode(f.read())
        self.info = t[b"info"]
        self.info_hash = hashlib.sha1(bencode(self.info)).digest()
        self.piece_len = self.info[b"piece length"]
        self.piece_hashes = t[b"info"][b"pieces"]
        with open(data_path, "rb") as f:
            self.data = f.read()
        self.port: int | None = None
        self.ready = threading.Event()
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            srv.settimeout(30)
            self.port = srv.getsockname()[1]
            self.ready.set()
            conn, _ = srv.accept()
            conn.settimeout(30)
            try:
                self._serve(conn)
            finally:
                conn.close()
            srv.close()
        except Exception as e:  # surfaced to the test, not swallowed
            self.error = e
            self.ready.set()

    def _serve(self, conn: socket.socket) -> None:
        _handshake(conn, self.info_hash, b"-BR0001-seeder------")
        n_pieces = len(self.piece_hashes) // 20
        bf = bytearray((n_pieces + 7) // 8)
        for i in range(n_pieces):
            bf[i // 8] |= 1 << (7 - (i % 8))
        _send_msg(conn, 5, bytes(bf))  # bitfield: we have everything
        while True:  # wait for interested
            msg_id, _ = _recv_msg(conn)
            if msg_id == 2:
                break
        _send_msg(conn, 1, b"")  # unchoke
        try:
            while True:
                msg_id, payload = _recv_msg(conn)
                if msg_id is None:
                    continue
                if msg_id == 6:  # request
                    index, begin, length = struct.unpack(">III", payload)
                    start = index * self.piece_len + begin
                    block = self.data[start:start + length]
                    _send_msg(conn, 7,
                              struct.pack(">II", index, begin) + block)
                elif msg_id == 3:  # not interested -> done
                    break
        except ConnectionError:
            pass  # leecher has everything and hung up: normal exit


def download(torrent_path: str, out_path: str, port: int,
             timeout: int = 30) -> str:
    """Download every piece from the loopback seeder; verify hashes."""
    with open(torrent_path, "rb") as f:
        t = bdecode(f.read())
    info = t[b"info"]
    info_hash = hashlib.sha1(bencode(info)).digest()
    piece_len = info[b"piece length"]
    piece_hashes = info[b"pieces"]
    n_pieces = len(piece_hashes) // 20
    total = info[b"length"]

    sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    sock.settimeout(timeout)
    _handshake(sock, info_hash, b"-BR0001-leecher-----")
    _send_msg(sock, 2, b"")  # interested
    while True:  # wait for unchoke
        msg_id, _ = _recv_msg(sock)
        if msg_id == 1:
            break

    data = bytearray()
    for index in range(n_pieces):
        piece = bytearray()
        want = min(piece_len, total - index * piece_len)
        for begin in range(0, want, BLOCK):
            ln = min(BLOCK, want - begin)
            _send_msg(sock, 6, struct.pack(">III", index, begin, ln))
            while True:
                msg_id, payload = _recv_msg(sock)
                if msg_id == 7:
                    r_index, r_begin = struct.unpack(">II", payload[:8])
                    assert (r_index, r_begin) == (index, begin)
                    piece += payload[8:]
                    break
        if hashlib.sha1(bytes(piece)).digest() != \
                piece_hashes[index * 20:(index + 1) * 20]:
            raise ValueError("piece %d hash mismatch" % index)
        data += piece
    sock.close()

    with open(out_path, "wb") as f:
        f.write(bytes(data))
    return out_path


def run(torrent_path: str, data_path: str,
        out_dir: str | None = None) -> dict:
    """Full loopback test. Returns details incl. byte-identity check."""
    out_dir = out_dir or os.path.dirname(os.path.abspath(data_path))
    os.makedirs(out_dir, exist_ok=True)
    seeder = Seeder(torrent_path, data_path)
    seeder.start()
    if not seeder.ready.wait(30) or seeder.port is None:
        raise RuntimeError("seeder failed to start: %s" % seeder.error)
    out_path = os.path.join(out_dir,
                            "downloaded-" + os.path.basename(data_path))
    try:
        download(torrent_path, out_path, seeder.port)
    finally:
        seeder.join(30)
    if seeder.error:
        raise seeder.error
    with open(data_path, "rb") as f:
        orig = f.read()
    with open(out_path, "rb") as f:
        got = f.read()
    return {"downloaded": out_path,
            "bytes": len(got),
            "identical": orig == got,
            "sha256": hashlib.sha256(got).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("torrent")
    ap.add_argument("data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    info = run(args.torrent, args.data, args.out)
    print("downloaded %d bytes -> %s" % (info["bytes"],
                                         info["downloaded"]))
    print("bit-identical:", info["identical"])
    print("sha256:", info["sha256"])
    if not info["identical"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
