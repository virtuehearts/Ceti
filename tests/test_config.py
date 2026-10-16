import os
import unittest

from ceti.config import BrainConfig, data_dir_default


class TestConfig(unittest.TestCase):
    def test_data_dir_default_is_package_relative(self):
        os.environ.pop("CETI_DATA_DIR", None)
        d = data_dir_default()
        self.assertTrue(d.endswith(os.path.join("ceti", "..", "data"))
                        or d.endswith("data"))
        self.assertEqual(BrainConfig().data_dir, d)

    def test_ceti_data_dir_override(self):
        os.environ["CETI_DATA_DIR"] = "~/my-brain-data"
        try:
            d = data_dir_default()
            self.assertEqual(d, os.path.expanduser("~/my-brain-data"))
            self.assertEqual(BrainConfig().data_dir, d)
        finally:
            del os.environ["CETI_DATA_DIR"]

    def test_cloud_defaults_fit_reasoning_models(self):
        # Reasoning models burn tokens on their thinking trace and
        # take a minute to answer; the old 512-token / 45s defaults
        # starved them to null content. Lightning is the tested
        # default: it answers and drives tools.
        cfg = BrainConfig()
        self.assertEqual(cfg.cloud_model,
                         "nvidia/nemotron-3.5-lightning:free")
        self.assertEqual(cfg.cloud_max_tokens, 2048)
        self.assertEqual(cfg.cloud_timeout, 180)

    def test_signing_respects_override(self):
        from ceti import signing
        os.environ["CETI_DATA_DIR"] = "/tmp/ceti-dd-test"
        try:
            self.assertEqual(
                signing.identity_path(),
                "/tmp/ceti-dd-test/identity.dat")
        finally:
            del os.environ["CETI_DATA_DIR"]


if __name__ == "__main__":
    unittest.main()
