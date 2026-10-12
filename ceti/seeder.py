"""seeder — BitTorrent distribution for signed datasets.

Mints .torrent files and magnet links for a dataset bundle. Pure
stdlib bencode — no client needed to CREATE a torrent.

Seeding itself needs a connected machine: nothing inbound can reach
this sandbox through NAT, so torrents are minted here and seeded by
archivists on machines with real connectivity (transmission-cli or
any client can seed the .torrent directly). The manifest inside the
bundle is already signed, so any peer can verify what they download
without trusting the seeder.

Usage:
    python -m brain.seeder <bundle_dir> [--out dir]
"""
from __future__ import annotations

import hashlib
import os
import urllib.parse

PIECE_LENGTH = 262144  # 256 KiB

TRACKERS = [
    "udp://tracker.openbittorrent.com:80/announce",
    "udp://tracker.opentrackr.org:1337/announce",
    "udp://exodus.desync.com:6969/announce",
]


# ---- bencode ----------------------------------------------------------
def bencode(obj) -> bytes:
    if isinstance(obj, int):
        return b"i" + str(obj).encode() + b"e"
    if isinstance(obj, bytes):
        return str(len(obj)).encode() + b":" + obj
    if isinstance(obj, str):
        return bencode(obj.encode("utf-8"))
    if isinstance(obj, list):
        return b"l" + b"".join(bencode(x) for x in obj) + b"e"
    if isinstance(obj, dict):
        out = b"d"
        for k in sorted(obj.keys(), key=lambda x: x if isinstance(x, bytes)
                        else x.encode("utf-8")):
            out += bencode(k) + bencode(obj[k])
        return out + b"e"
    raise TypeError("cannot bencode %r" % type(obj))


def bdecode(data: bytes):
    def _dec(i):
        c = data[i:i + 1]
        if c == b"i":
            j = data.index(b"e", i)
            return int(data[i + 1:j]), j + 1
        if c == b"l":
            lst, i = [], i + 1
            while data[i:i + 1] != b"e":
                v, i = _dec(i)
                lst.append(v)
            return lst, i + 1
        if c == b"d":
            d, i = {}, i + 1
            while data[i:i + 1] != b"e":
                k, i = _dec(i)
                v, i = _dec(i)
                d[k] = v
            return d, i + 1
        j = data.index(b":", i)
        n = int(data[i:j])
        return data[j + 1:j + 1 + n], j + 1 + n
    val, _ = _dec(0)
    return val


# ---- torrent minting ----------------------------------------------------
def _iter_files(bundle: str):
    """Yield (relpath_parts, abspath, size) for every file, sorted."""
    out = []
    for root, _dirs, files in os.walk(bundle):
        for fn in sorted(files):
            ap = os.path.join(root, fn)
            rel = os.path.relpath(ap, os.path.dirname(bundle))
            out.append((rel.split(os.sep), ap, os.path.getsize(ap)))
    return out


def create_torrent(path: str, out_path: str | None = None,
                   piece_length: int = PIECE_LENGTH,
                   trackers: list[str] | None = None) -> dict:
    """Mint a .torrent for a file (e.g. a release tarball) or a bundle
    directory. Single-file mode when given a file.
    Returns {torrent_path, info_hash, magnet, files, total_bytes}."""
    trackers = trackers if trackers is not None else TRACKERS
    path = os.path.abspath(path)
    name = os.path.basename(path)

    if os.path.isfile(path):
        info = _single_file_info(path, name, piece_length)
        n_files, total = 1, os.path.getsize(path)
    else:
        files = _iter_files(path)
        info, n_files, total = _multi_file_info(path, name, files,
                                                piece_length)
    return _finalize(info, name, path, out_path, trackers,
                     n_files, total)


def _hash_pieces(paths: list[str], piece_length: int) -> bytes:
    pieces, buf = b"", b""
    for ap in paths:
        with open(ap, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                buf += chunk
                while len(buf) >= piece_length:
                    pieces += hashlib.sha1(buf[:piece_length]).digest()
                    buf = buf[piece_length:]
    if buf:
        pieces += hashlib.sha1(buf).digest()
    return pieces


def _single_file_info(path: str, name: str,
                      piece_length: int) -> dict:
    return {"name": name.encode("utf-8"),
            "piece length": piece_length,
            "pieces": _hash_pieces([path], piece_length),
            "length": os.path.getsize(path)}


def _multi_file_info(path: str, name: str, files: list,
                     piece_length: int) -> tuple[dict, int, int]:
    file_entries, total = [], 0
    for parts, ap, size in files:
        file_entries.append({"length": size,
                             "path": [p.encode("utf-8") for p in parts]})
        total += size
    info = {"name": name.encode("utf-8"),
            "piece length": piece_length,
            "pieces": _hash_pieces([ap for _, ap, _ in files],
                                   piece_length),
            "files": file_entries}
    return info, len(files), total


def _finalize(info: dict, name: str, path: str, out_path: str | None,
              trackers: list[str], n_files: int,
              total: int) -> dict:
    torrent = {"announce": trackers[0].encode("utf-8"),
               "announce-list": [[t.encode("utf-8")] for t in trackers],
               "creation date": 0,
               "created by": b"brain-seeder 0.1",
               "info": info}
    encoded = bencode(torrent)
    info_hash = hashlib.sha1(bencode(info)).hexdigest()

    out_path = out_path or (path + ".torrent")
    with open(out_path, "wb") as f:
        f.write(encoded)

    # real-world magnet shape: literal colons in xt, trackers encoded
    parts = ["xt=urn:btih:" + info_hash,
             "dn=" + urllib.parse.quote(name)]
    parts += ["tr=" + urllib.parse.quote(t, safe="")
              for t in trackers]
    magnet = "magnet:?" + "&".join(parts)
    return {"torrent_path": out_path, "info_hash": info_hash,
            "magnet": magnet,
            "files": n_files, "total_bytes": total}


def main() -> None:
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="release tarball or bundle directory")
    ap.add_argument("--out", help="where to write the .torrent")
    args = ap.parse_args()
    info = create_torrent(args.path, args.out)
    print(json.dumps({k: v for k, v in info.items()
                      if k != "magnet"}, indent=1))
    print("magnet:")
    print(info["magnet"])


if __name__ == "__main__":
    main()
