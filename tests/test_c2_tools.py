"""C2 console tools: dictionary, thesaurus, memopad, weather."""
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from ceti import tools


def _fake_urlopen(payload):
    resp = mock.MagicMock()
    resp.__enter__.return_value = resp
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    return resp


DICT_PAYLOAD = [{
    "word": "modem",
    "meanings": [{
        "partOfSpeech": "noun",
        "definitions": [
            {"definition": "A device that modulates and demodulates."},
            {"definition": "Second definition.",
             "synonyms": ["coupler"], "antonyms": []}],
    }],
}]

WTTR_PAYLOAD = {
    "current_condition": [{
        "temp_C": "7", "humidity": "60", "windspeedKmph": "12",
        "winddir16Point": "NW",
        "weatherDesc": [{"value": "Partly cloudy"}],
    }],
    "nearest_area": [{
        "areaName": [{"value": "Toronto"}],
        "country": [{"value": "Canada"}],
    }],
}


class TestC2Tools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._old = os.environ.get("CETI_WORK_ROOT")
        os.environ["CETI_WORK_ROOT"] = self.tmp

    def tearDown(self):
        if self._old is None:
            del os.environ["CETI_WORK_ROOT"]
        else:
            os.environ["CETI_WORK_ROOT"] = self._old

    def test_dictionary_api(self):
        with mock.patch("ceti.tools.urllib.request.urlopen",
                        return_value=_fake_urlopen(DICT_PAYLOAD)):
            r = tools.execute("dictionary", word="modem")
        self.assertTrue(r.success, r.error)
        self.assertIn("modulates", r.content)

    def test_dictionary_local_fallback(self):
        with mock.patch("ceti.tools.urllib.request.urlopen",
                        side_effect=OSError("offline")):
            r = tools.execute("dictionary", word="cypherpunk")
        self.assertTrue(r.success, r.error)
        self.assertIn("privacy", r.content)

    def test_dictionary_unknown(self):
        with mock.patch("ceti.tools.urllib.request.urlopen",
                        side_effect=OSError("offline")):
            r = tools.execute("dictionary", word="zzzq")
        self.assertFalse(r.success)
        self.assertIn("no definition", r.error)

    def test_thesaurus_api(self):
        with mock.patch("ceti.tools.urllib.request.urlopen",
                        return_value=_fake_urlopen(DICT_PAYLOAD)):
            r = tools.execute("thesaurus", word="modem")
        self.assertTrue(r.success, r.error)
        self.assertIn("coupler", r.content)

    def test_thesaurus_local_fallback(self):
        with mock.patch("ceti.tools.urllib.request.urlopen",
                        side_effect=OSError("offline")):
            r = tools.execute("thesaurus", word="cipher")
        self.assertTrue(r.success, r.error)
        self.assertIn("code", r.content)

    def test_memopad_append_read_clear(self):
        self.assertEqual(tools.call("memopad", action="append",
                                    text="first note"), "noted.")
        self.assertEqual(tools.call("memopad", action="append",
                                    text="second note"), "noted.")
        all_notes = tools.call("memopad", action="read")
        self.assertIn("first note", all_notes)
        self.assertIn("second note", all_notes)
        last = tools.call("memopad", action="read", limit=1)
        self.assertIn("second note", last)
        self.assertNotIn("first note", last)
        self.assertEqual(tools.call("memopad", action="clear"),
                         "memopad cleared.")
        self.assertIn("empty", tools.call("memopad", action="read"))

    def test_memopad_bad_action(self):
        r = tools.execute("memopad", action="explode")
        self.assertFalse(r.success)

    def test_weather_api(self):
        with mock.patch("ceti.tools.urllib.request.urlopen",
                        return_value=_fake_urlopen(WTTR_PAYLOAD)):
            r = tools.execute("weather", place="Toronto")
        self.assertTrue(r.success, r.error)
        self.assertIn("Toronto", r.content)
        self.assertIn("7°C", r.content)
        self.assertIn("Partly cloudy", r.content)

    def test_weather_failure_is_loud(self):
        with mock.patch("ceti.tools.urllib.request.urlopen",
                        side_effect=OSError("offline")):
            r = tools.execute("weather", place="Toronto")
        self.assertFalse(r.success)
        self.assertIn("fetch failed", r.error)


if __name__ == "__main__":
    unittest.main()
