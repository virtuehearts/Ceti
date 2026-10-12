import os
import tempfile
import unittest

from ceti.seeder import create_torrent
from ceti.seedtest import run


class TestLoopback(unittest.TestCase):
    def test_loopback_transfer(self):
        tmp = tempfile.mkdtemp()
        data = os.path.join(tmp, "payload.bin")
        with open(data, "wb") as f:
            f.write(os.urandom(60000))  # spans multiple 16k blocks
        tinfo = create_torrent(data,
                               out_path=os.path.join(tmp, "p.torrent"))
        info = run(tinfo["torrent_path"], data,
                   out_dir=os.path.join(tmp, "dl"))
        self.assertTrue(info["identical"])
        self.assertEqual(info["bytes"], 60000)


if __name__ == "__main__":
    unittest.main()
