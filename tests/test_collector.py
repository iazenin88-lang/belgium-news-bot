import importlib.util
import sys
import time
import types
import unittest
import xml.etree.ElementTree as ElementTree
from types import SimpleNamespace
from unittest.mock import patch


if importlib.util.find_spec("feedparser") is None:
    feedparser_stub = types.ModuleType("feedparser")

    def parse_feed_stub(payload):
        root = ElementTree.fromstring(payload)
        entries = []
        for item in root.findall(".//item"):
            entries.append({
                "title": item.findtext("title"),
                "link": item.findtext("link"),
            })
        return SimpleNamespace(entries=entries)

    feedparser_stub.parse = parse_feed_stub
    sys.modules["feedparser"] = feedparser_stub

if importlib.util.find_spec("requests") is None:
    requests_stub = types.ModuleType("requests")

    class StubSession:
        pass

    requests_stub.Session = StubSession
    sys.modules["requests"] = requests_stub

if importlib.util.find_spec("bs4") is None:
    bs4_stub = types.ModuleType("bs4")
    bs4_stub.BeautifulSoup = lambda *_args, **_kwargs: None
    sys.modules["bs4"] = bs4_stub

if importlib.util.find_spec("newspaper") is None:
    newspaper_stub = types.ModuleType("newspaper")

    class StubArticle:
        def __init__(self, *_args, **_kwargs):
            self.text = ""

        def set_html(self, *_args, **_kwargs):
            return None

        def parse(self):
            return None

    newspaper_stub.Article = StubArticle
    sys.modules["newspaper"] = newspaper_stub

try:
    from supabase import create_client as _create_client
except (ImportError, ModuleNotFoundError):
    supabase_stub = types.ModuleType("supabase")
    supabase_stub.create_client = lambda *_args, **_kwargs: None
    sys.modules["supabase"] = supabase_stub

from collector import (
    REQUEST_HEADERS,
    RequestPacer,
    build_candidate,
    collect_source,
    fetch_feed,
    source_uses_feed_only,
)


class FakeResponse:
    def __init__(self, content=b"", content_type="application/atom+xml"):
        self.content = content
        self.text = content.decode("utf-8", errors="replace")
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, response=None):
        self.response = response or FakeResponse()
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


class FakeQuery:
    def __init__(self, database, table_name):
        self.database = database
        self.table_name = table_name
        self.operation = None
        self.row = None

    def select(self, *_args):
        self.operation = "select"
        return self

    def eq(self, *_args):
        return self

    def in_(self, *_args):
        return self

    def insert(self, row):
        self.operation = "insert"
        self.row = dict(row)
        return self

    def execute(self):
        if self.operation == "select":
            return SimpleNamespace(
                data=[{"fingerprint": value} for value in self.database.existing]
            )
        if self.operation == "insert":
            self.database.inserted.append(self.row)
            return SimpleNamespace(data=[self.row])
        raise AssertionError("Unexpected fake database operation")


class FakeDatabase:
    def __init__(self, existing=None):
        self.existing = set(existing or [])
        self.inserted = []

    def table(self, table_name):
        return FakeQuery(self, table_name)


def feed_entry():
    return {
        "link": "https://vrtnws.be/p.example",
        "title": "Belgian housing rule",
        "summary": "A useful summary from the official VRT feed.",
        "published_parsed": time.gmtime(0),
    }


class CollectorAccessTests(unittest.TestCase):
    def test_vrt_is_feed_only(self):
        self.assertTrue(source_uses_feed_only({
            "url": "https://www.vrt.be/vrtnws/nl.rss.articles.xml",
        }))
        self.assertFalse(source_uses_feed_only({
            "url": "https://www.politico.eu/feed/",
        }))

    def test_feed_request_identifies_the_robot(self):
        xml = b"""<?xml version='1.0'?><rss><channel>
        <item><title>News</title><link>https://example.com/news</link></item>
        </channel></rss>"""
        session = FakeSession(FakeResponse(xml))

        feed = fetch_feed("https://example.com/feed", session)

        self.assertEqual(len(feed.entries), 1)
        self.assertEqual(
            session.calls[0][1]["headers"]["User-Agent"],
            REQUEST_HEADERS["User-Agent"],
        )

    def test_html_response_is_not_silently_treated_as_empty_feed(self):
        session = FakeSession(FakeResponse(b"<html></html>", "text/html"))
        with self.assertRaisesRegex(ValueError, "returned HTML"):
            fetch_feed("https://example.com/feed", session)

    def test_vrt_new_article_is_saved_without_opening_article_page(self):
        database = FakeDatabase()
        source = {
            "id": 1,
            "name": "VRT NWS",
            "url": "https://www.vrt.be/vrtnws/nl.rss.articles.xml",
        }
        feed = SimpleNamespace(entries=[feed_entry()])

        with patch("collector.fetch_feed", return_value=feed):
            new, duplicate = collect_source(
                database,
                source,
                FakeSession(),
                RequestPacer(3),
                extractor=lambda *_args: self.fail("article page was opened"),
            )

        self.assertEqual((new, duplicate), (1, 0))
        self.assertEqual(len(database.inserted), 1)
        self.assertIsNone(database.inserted[0]["content"])

    def test_duplicate_is_checked_before_opening_article_page(self):
        entry = feed_entry()
        known_fingerprint = build_candidate(2, entry)["fingerprint"]
        database = FakeDatabase(existing=[known_fingerprint])
        source = {
            "id": 2,
            "name": "Other source",
            "url": "https://example.com/feed.xml",
        }
        feed = SimpleNamespace(entries=[entry])

        with patch("collector.fetch_feed", return_value=feed):
            new, duplicate = collect_source(
                database,
                source,
                FakeSession(),
                RequestPacer(3),
                extractor=lambda *_args: self.fail("duplicate page was opened"),
            )

        self.assertEqual((new, duplicate), (0, 1))
        self.assertEqual(database.inserted, [])

    def test_article_requests_are_spaced_three_seconds_apart(self):
        clock_values = iter([10.0, 11.0, 13.0])
        sleeps = []
        pacer = RequestPacer(
            3,
            sleep_fn=sleeps.append,
            clock=lambda: next(clock_values),
        )

        pacer.wait()
        pacer.wait()

        self.assertEqual(sleeps, [2.0])


if __name__ == "__main__":
    unittest.main()
