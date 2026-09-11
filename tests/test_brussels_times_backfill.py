import importlib.util
import importlib.machinery
import re
import sys
import types
import unittest
import xml.etree.ElementTree as ElementTree
from types import SimpleNamespace


if importlib.util.find_spec("feedparser") is None:
    feedparser_stub = types.ModuleType("feedparser")
    feedparser_stub.__spec__ = importlib.machinery.ModuleSpec(
        "feedparser", loader=None
    )

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
    requests_stub.__spec__ = importlib.machinery.ModuleSpec(
        "requests", loader=None
    )

    class StubSession:
        pass

    requests_stub.Session = StubSession
    sys.modules["requests"] = requests_stub

if importlib.util.find_spec("bs4") is None:
    bs4_stub = types.ModuleType("bs4")
    bs4_stub.__spec__ = importlib.machinery.ModuleSpec("bs4", loader=None)

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
    newspaper_stub.__spec__ = importlib.machinery.ModuleSpec(
        "newspaper", loader=None
    )

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
    supabase_stub.__spec__ = importlib.machinery.ModuleSpec(
        "supabase", loader=None
    )
    supabase_stub.create_client = lambda *_args, **_kwargs: None
    sys.modules["supabase"] = supabase_stub

from backfill_brussels_times import (
    MAX_BACKFILL_URLS,
    backfill_urls,
    build_backfill_row,
    parse_backfill_urls,
    parse_published_at,
)
from collector import RequestPacer


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def execute(self):
        return SimpleNamespace(data=self.data)


class FakeQuery:
    def __init__(self, database, table_name):
        self.database = database
        self.table_name = table_name
        self.filters = []
        self.operation = "select"
        self.row = None

    def select(self, *_args):
        self.operation = "select"
        return self

    def eq(self, column, value):
        self.filters.append(("eq", column, value))
        return self

    def like(self, column, value):
        self.filters.append(("like", column, value))
        return self

    def limit(self, *_args):
        return self

    def insert(self, row):
        self.operation = "insert"
        self.row = dict(row)
        return self

    def execute(self):
        if self.operation == "insert":
            row = dict(self.row)
            row["id"] = self.database.next_article_id
            self.database.next_article_id += 1
            self.database.articles.append(row)
            return FakeResponse([row])

        if self.table_name == "sources":
            rows = list(self.database.sources)
        elif self.table_name == "articles":
            rows = list(self.database.articles)
        else:
            raise AssertionError(f"Unexpected table {self.table_name}")

        for operation, column, value in self.filters:
            if operation == "eq":
                rows = [row for row in rows if row.get(column) == value]
            else:
                needle = value.strip("%")
                rows = [row for row in rows if needle in (row.get(column) or "")]
        return FakeResponse(rows)


class FakeDatabase:
    def __init__(self):
        self.sources = [{
            "id": 3,
            "name": "Brussels Times",
            "url": "https://www.brusselstimes.com/google-news-sitemap.xml",
            "enabled": True,
        }]
        self.articles = []
        self.next_article_id = 53133

    def table(self, table_name):
        return FakeQuery(self, table_name)


class BackfillTests(unittest.TestCase):
    def test_parse_urls_deduplicates_and_rejects_other_domains(self):
        url = "https://www.brusselstimes.com/2301967/example?utm_source=test"
        parsed = parse_backfill_urls(f"{url}\n{url}")
        self.assertEqual(parsed, [url])

        with self.assertRaises(ValueError):
            parse_backfill_urls("https://example.com/2301967/article")

    def test_parse_urls_has_a_small_safety_limit(self):
        urls = [
            f"https://www.brusselstimes.com/{index}/article"
            for index in range(MAX_BACKFILL_URLS + 1)
        ]
        with self.assertRaisesRegex(ValueError, "at most"):
            parse_backfill_urls(explicit_urls=urls)

    def test_published_timestamp_is_normalized_to_utc(self):
        self.assertEqual(
            parse_published_at("2026-09-06 19:39:18"),
            "2026-09-06T19:39:18+00:00",
        )

    def test_build_row_uses_short_publisher_abstract(self):
        article_id, row = build_backfill_row(
            3,
            "https://www.brusselstimes.com/brussels/2301967/example",
            {
                "id": 2301967,
                "title": "Historical article",
                "published": "2026-09-06 19:39:18",
                "seo_description": "A <strong>short</strong> abstract.",
                "content": "<p>Full article body must not be stored.</p>",
            },
        )
        self.assertEqual(article_id, 2301967)
        self.assertEqual(row["summary"], "A short abstract.")
        self.assertIsNone(row["content"])
        self.assertEqual(row["published_at"], "2026-09-06T19:39:18+00:00")

    def test_backfill_is_idempotent_by_article_url(self):
        database = FakeDatabase()
        url = "https://www.brusselstimes.com/brussels/2301967/example"
        calls = []

        def metadata_fetcher(requested_url, _session):
            calls.append(requested_url)
            return {
                "id": 2301967,
                "title": "Historical article",
                "published": "2026-09-06 19:39:18",
                "seo_description": "A short abstract.",
            }

        first = backfill_urls(
            database,
            [url],
            pacer=RequestPacer(0),
            metadata_fetcher=metadata_fetcher,
        )
        second = backfill_urls(
            database,
            [url],
            pacer=RequestPacer(0),
            metadata_fetcher=metadata_fetcher,
        )

        self.assertEqual(first, {"inserted": 1, "skipped": 0, "failed": 0})
        self.assertEqual(second, {"inserted": 0, "skipped": 1, "failed": 0})
        self.assertEqual(calls, [url])
        self.assertEqual(len(database.articles), 1)


if __name__ == "__main__":
    unittest.main()
