import unittest

from ceti import pii


class TestPii(unittest.TestCase):
    def test_email(self):
        out, found = pii.strip("mail me at jero@darknet.ca please")
        self.assertIn("[REDACTED_EMAIL]", out)
        self.assertNotIn("jero@darknet.ca", out)
        self.assertIn("EMAIL", found)

    def test_phone(self):
        out, found = pii.strip("call 416-555-0199 tomorrow")
        self.assertIn("[REDACTED_PHONE]", out)
        self.assertIn("PHONE", found)

    def test_ipv4(self):
        out, found = pii.strip("server at 192.168.1.10 went down")
        self.assertIn("[REDACTED_IPV4]", out)

    def test_api_key(self):
        out, found = pii.strip("key=sk-abc123DEF456ghi789")
        self.assertIn("[REDACTED_API_KEY]", out)

    def test_name_intro(self):
        out, found = pii.strip("my name is John Smith")
        self.assertIn("[REDACTED_NAME_INTRO]", out)

    def test_clean_text_untouched(self):
        out, found = pii.strip("what is the capital of france")
        self.assertEqual(out, "what is the capital of france")
        self.assertEqual(found, [])

    def test_looks_personal(self):
        self.assertTrue(pii.looks_personal("what is my phone number"))
        self.assertTrue(pii.looks_personal("mail bob@example.com"))
        self.assertFalse(pii.looks_personal("what is a BBS"))


if __name__ == "__main__":
    unittest.main()
