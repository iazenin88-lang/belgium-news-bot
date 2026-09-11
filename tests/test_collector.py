import importlib.util
import re
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

    class StubSoup:
        def __init__(self, value, *_args, **_kwargs):
            self.value = value

        def get_text(self, separator="", strip=False):
            value = re.sub(r"<[^>]+>", separator, self.value)
            return value.strip() if strip else value

    bs4_stub.BeautifulSoup = StubSoup
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
    BRUSSELS_TIMES_ARTICLE_API,
    BRUSSELS_TIMES_ENTRY_LIMIT,
    REQUEST_HEADERS,
    RequestPacer,
    brussels_times_article_id,
    build_candidate,
    collect_source,
    extract_brussels_times_summary,
    fetch_brussels_times_article,
    fetch_feed,
    parse_google_news_sitemap,
    source_is_brussels_times,
    source_uses_feed_only,
)


class FakeResponse:
    def __init__(
        self,
        content=b"",
        content_type="application/atom+xml",
        json_payload=None,
    ):
        self.content = content
        self.text = content.decode("utf-8", errors="replace")
        self.headers = {"Content-Type": content_type}
        self.json_payload = json_payload

    def raise_for_status(self):
        return None

    def json(self):
        if self.json_payload is None:
            raise ValueError("No JSON payload configured")
        return self.json_payload


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


def brussels_times_entry(article_id=2308789, minute=0):
    return {
        "link": (
            f"https://www.brusselstimes.com/{article_id}/"
            f"example-article-{article_id}"
        ),
        "title": f"Brussels Times article {article_id}",
        "published_parsed": time.strptime(
            f"2026-09-09T06:{minute:02d}:00Z",
            "%Y-%m-%dT%H:%M:%SZ",
        ),
    }


class CollectorAccessTests(unittest.TestCase):
    def test_vrt_is_feed_only(self):
        self.assertTrue(source_uses_feed_only({
            "url": "https://www.vrt.be/vrtnws/nl.rss.articles.xml",
        }))
        self.assertFalse(source_uses_feed_only({
            "url": "https://www.politico.eu/feed/",
        }))

    def test_brussels_times_source_is_recognized_by_domain(self):
        self.assertTrue(source_is_brussels_times({
            "url": "https://www.brusselstimes.com/google-news-sitemap.xml",
        }))
        self.assertTrue(source_is_brussels_times({
            "url": "https://brusselstimes.com/google-news-sitemap.xml",
        }))
        self.assertFalse(source_is_brussels_times({
            "url": "https://notbrusselstimes.com/feed.xml",
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

    def test_google_news_sitemap_is_parsed_newest_first(self):
        xml = b"""<?xml version='1.0' encoding='UTF-8'?>
        <urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'
                xmlns:news='http://www.google.com/schemas/sitemap-news/0.9'>
          <url>
            <loc>https://www.brusselstimes.com/100/older</loc>
            <news:news>
              <news:publication_date>2026-09-08T10:00:00+00:00</news:publication_date>
              <news:title>Older article</news:title>
            </news:news>
          </url>
          <url>
            <loc>https://www.brusselstimes.com/101/newer</loc>
            <news:news>
              <news:publication_date>2026-09-09T06:10:00Z</news:publication_date>
              <news:title>Newer article</news:title>
            </news:news>
          </url>
        </urlset>"""
        session = FakeSession(FakeResponse(xml, "text/xml"))

        sitemap = fetch_feed(
            "https://www.brusselstimes.com/google-news-sitemap.xml",
            session,
        )

        self.assertEqual(
            [entry["title"] for entry in sitemap.entries],
            ["Newer article", "Older article"],
        )
        self.assertEqual(
            build_candidate(3, sitemap.entries[0])["published_at"],
            "2026-09-09T06:10:00Z",
        )

    def test_non_sitemap_urlset_is_rejected_by_sitemap_parser(self):
        with self.assertRaisesRegex(ValueError, "not a sitemap"):
            parse_google_news_sitemap(b"<rss />")

    def test_brussels_times_article_id_is_read_from_path(self):
        self.assertEqual(
            brussels_times_article_id(
                "https://www.brusselstimes.com/2308789/example-title"
            ),
            2308789,
        )
        self.assertIsNone(
            brussels_times_article_id(
                "https://www.brusselstimes.com/example-title"
            )
        )

    def test_brussels_times_fetches_only_clean_publisher_abstract(self):
        response = FakeResponse(
            content_type="application/json",
            json_payload={
                "id": 2308789,
                "seo_description": "  A <strong>short</strong> publisher abstract.  ",
                "sub": "Fallback abstract",
                "content": "Full article body must not be stored.",
            },
        )
        session = FakeSession(response)

        summary = extract_brussels_times_summary(
            "https://www.brusselstimes.com/2308789/example-title",
            session,
        )

        self.assertEqual(summary, "A short publisher abstract.")
        self.assertEqual(session.calls[0][0], BRUSSELS_TIMES_ARTICLE_API)
        self.assertEqual(session.calls[0][1]["params"], {"id": 2308789})
        self.assertEqual(
            session.calls[0][1]["headers"]["User-Agent"],
            REQUEST_HEADERS["User-Agent"],
        )
        self.assertEqual(
            session.calls[0][1]["headers"]["Accept"],
            "application/json",
        )

    def test_brussels_times_article_metadata_is_validated(self):
        response = FakeResponse(
            content_type="application/json",
            json_payload={
                "id": 2308789,
                "title": "A historical Brussels Times article",
                "published": "2026-09-09 06:00:00",
                "seo_description": "Short publisher description.",
            },
        )
        session = FakeSession(response)

        metadata = fetch_brussels_times_article(
            "https://www.brusselstimes.com/2308789/example-article",
            session,
        )

        self.assertEqual(metadata["id"], 2308789)
        self.assertEqual(metadata["title"], "A historical Brussels Times article")
        self.assertEqual(session.calls[0][0], BRUSSELS_TIMES_ARTICLE_API)

    def test_brussels_times_article_metadata_rejects_wrong_id(self):
        response = FakeResponse(
            content_type="application/json",
            json_payload={"id": 999999, "title": "Wrong article"},
        )
        session = FakeSession(response)

        metadata = fetch_brussels_times_article(
            "https://www.brusselstimes.com/2308789/example-article",
            session,
        )

        self.assertIsNone(metadata)

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

    def test_brussels_times_duplicate_is_checked_before_api_request(self):
        entry = brussels_times_entry()
        known_fingerprint = build_candidate(3, entry)["fingerprint"]
        database = FakeDatabase(existing=[known_fingerprint])
        source = {
            "id": 3,
            "name": "Brussels Times",
            "url": "https://www.brusselstimes.com/google-news-sitemap.xml",
        }

        with patch(
            "collector.fetch_feed",
            return_value=SimpleNamespace(entries=[entry]),
        ):
            new, duplicate = collect_source(
                database,
                source,
                FakeSession(),
                RequestPacer(3),
                extractor=lambda *_args: self.fail("article page was opened"),
                brussels_times_extractor=lambda *_args: self.fail(
                    "duplicate triggered an API request"
                ),
            )

        self.assertEqual((new, duplicate), (0, 1))
        self.assertEqual(database.inserted, [])

    def test_brussels_times_new_article_saves_summary_but_not_body(self):
        database = FakeDatabase()
        source = {
            "id": 3,
            "name": "Brussels Times",
            "url": "https://www.brusselstimes.com/google-news-sitemap.xml",
        }
        requested = []

        def summary_extractor(url, _session):
            requested.append(url)
            return "A concise description supplied by Brussels Times."

        with patch(
            "collector.fetch_feed",
            return_value=SimpleNamespace(entries=[brussels_times_entry()]),
        ):
            new, duplicate = collect_source(
                database,
                source,
                FakeSession(),
                RequestPacer(0),
                extractor=lambda *_args: self.fail("article page was opened"),
                brussels_times_extractor=summary_extractor,
            )

        self.assertEqual((new, duplicate), (1, 0))
        self.assertEqual(len(requested), 1)
        self.assertEqual(
            database.inserted[0]["summary"],
            "A concise description supplied by Brussels Times.",
        )
        self.assertIsNone(database.inserted[0]["content"])

    def test_brussels_times_bootstrap_is_bounded(self):
        database = FakeDatabase()
        source = {
            "id": 3,
            "name": "Brussels Times",
            "url": "https://www.brusselstimes.com/google-news-sitemap.xml",
        }
        entries = [
            brussels_times_entry(2308000 + index, index % 60)
            for index in range(BRUSSELS_TIMES_ENTRY_LIMIT + 5)
        ]
        requested = []

        with patch(
            "collector.fetch_feed",
            return_value=SimpleNamespace(entries=entries),
        ):
            new, duplicate = collect_source(
                database,
                source,
                FakeSession(),
                RequestPacer(0),
                brussels_times_extractor=lambda url, _session: (
                    requested.append(url) or "Publisher abstract"
                ),
            )

        self.assertEqual((new, duplicate), (BRUSSELS_TIMES_ENTRY_LIMIT, 0))
        self.assertEqual(len(requested), BRUSSELS_TIMES_ENTRY_LIMIT)
        self.assertEqual(len(database.inserted), BRUSSELS_TIMES_ENTRY_LIMIT)

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
