import hashlib
import os
import re
import time
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import feedparser
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from supabase import create_client


TRACKING_PARAMS_PREFIXES = ("utm_",)
TRACKING_PARAMS_EXACT = {"fbclid", "gclid", "mc_cid", "mc_eid"}

USER_AGENT = (
    "BelgiumNewsBot/1.0 "
    "(+https://github.com/iazenin88-lang/belgium-news-bot)"
)
REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": (
        "application/atom+xml, application/rss+xml, application/xml, "
        "text/xml, text/html;q=0.8"
    ),
}
REQUEST_TIMEOUT_SECONDS = 20
ARTICLE_REQUEST_INTERVAL_SECONDS = float(
    os.getenv("ARTICLE_REQUEST_INTERVAL_SECONDS", "3")
)
FINGERPRINT_QUERY_BATCH_SIZE = 40

# VRT supplies a useful abstract in its public feed. Do not open every linked
# page: one feed request per pipeline run is sufficient for collection.
FEED_ONLY_DOMAINS = {"vrt.be"}


class RequestPacer:
    """Keep article-page requests at least ``interval_seconds`` apart."""

    def __init__(
        self,
        interval_seconds: float,
        sleep_fn=time.sleep,
        clock=time.monotonic,
    ):
        self.interval_seconds = max(0.0, interval_seconds)
        self.sleep_fn = sleep_fn
        self.clock = clock
        self.last_request_started_at = None

    def wait(self) -> None:
        now = self.clock()
        if self.last_request_started_at is not None:
            remaining = (
                self.interval_seconds - (now - self.last_request_started_at)
            )
            if remaining > 0:
                self.sleep_fn(remaining)
                now = self.clock()
        self.last_request_started_at = now


def canonicalize_url(url: str) -> str:
    url = url.strip()
    p = urlparse(url)

    scheme = "https" if p.scheme in ("http", "https") else p.scheme
    netloc = p.netloc.lower()

    path = p.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    query_items = []
    for k, v in parse_qsl(p.query, keep_blank_values=True):
        kl = k.lower()
        if kl in TRACKING_PARAMS_EXACT:
            continue
        if any(kl.startswith(pref) for pref in TRACKING_PARAMS_PREFIXES):
            continue
        query_items.append((k, v))

    query = urlencode(query_items, doseq=True)

    return urlunparse((scheme, netloc, path, "", query, ""))


def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def fingerprint(source_id: int, title: str, canonical_url: str) -> str:
    base = f"{source_id}|{norm_text(title)}|{canonical_url}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def registered_domain(hostname: str | None) -> str:
    hostname = (hostname or "").lower().rstrip(".")
    if hostname == "vrt.be" or hostname.endswith(".vrt.be"):
        return "vrt.be"
    return hostname


def source_uses_feed_only(source: dict) -> bool:
    hostname = urlparse(source.get("url") or "").hostname
    return registered_domain(hostname) in FEED_ONLY_DOMAINS


def fetch_feed(feed_url: str, session: requests.Session):
    response = session.get(
        feed_url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers=REQUEST_HEADERS,
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" in content_type:
        raise ValueError(f"Feed returned HTML instead of RSS/Atom: {feed_url}")

    feed = feedparser.parse(response.content)
    if not getattr(feed, "entries", None):
        raise ValueError(f"Feed contains no entries: {feed_url}")
    return feed


def extract_text_from_html(
    url: str,
    session: requests.Session,
) -> str | None:
    """Download a page once and parse the same response with both parsers."""
    try:
        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers=REQUEST_HEADERS,
        )
        response.raise_for_status()
    except Exception:
        return None

    try:
        article = Article(url)
        article.set_html(response.text)
        article.parse()
        text = (article.text or "").strip()
        if len(text) >= 200:
            return text[:50000]
    except Exception:
        pass

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    article_tag = soup.find("article")
    if article_tag:
        text = article_tag.get_text(" ", strip=True)
    else:
        text = soup.get_text(" ", strip=True)

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 100:
        return None
    return text[:50000]


def build_candidate(source_id: int, entry: dict) -> dict | None:
    original_url = entry.get("link")
    if not original_url:
        return None

    canonical_url = canonicalize_url(original_url)
    title = entry.get("title")
    published_at = None
    if entry.get("published_parsed"):
        published_at = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            entry.get("published_parsed"),
        )

    return {
        "source_id": source_id,
        "original_url": original_url,
        "canonical_url": canonical_url,
        "title": title,
        "summary": entry.get("summary"),
        "published_at": published_at,
        "fingerprint": fingerprint(source_id, title or "", canonical_url),
        "content": None,
    }


def existing_fingerprints(sb, source_id: int, fingerprints: list[str]) -> set[str]:
    existing = set()
    for start in range(0, len(fingerprints), FINGERPRINT_QUERY_BATCH_SIZE):
        batch = fingerprints[start:start + FINGERPRINT_QUERY_BATCH_SIZE]
        rows = (
            sb.table("articles")
            .select("fingerprint")
            .eq("source_id", source_id)
            .in_("fingerprint", batch)
            .execute()
            .data
        )
        existing.update(row["fingerprint"] for row in rows)
    return existing


def collect_source(
    sb,
    source: dict,
    session: requests.Session,
    pacer: RequestPacer,
    extractor=extract_text_from_html,
) -> tuple[int, int]:
    source_id = source["id"]
    feed = fetch_feed(source["url"], session)

    candidates = []
    seen = set()
    duplicate_count = 0
    for entry in feed.entries[:100]:
        candidate = build_candidate(source_id, entry)
        if candidate is None:
            continue
        if candidate["fingerprint"] in seen:
            duplicate_count += 1
            continue
        seen.add(candidate["fingerprint"])
        candidates.append(candidate)

    known = existing_fingerprints(
        sb,
        source_id,
        [candidate["fingerprint"] for candidate in candidates],
    )

    new_count = 0
    feed_only = source_uses_feed_only(source)
    for row in candidates:
        if row["fingerprint"] in known:
            duplicate_count += 1
            continue

        if not feed_only:
            pacer.wait()
            row["content"] = extractor(row["canonical_url"], session)

        try:
            sb.table("articles").insert(row).execute()
            new_count += 1
            known.add(row["fingerprint"])
        except Exception:
            duplicate_count += 1

    return new_count, duplicate_count


def main():
    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_SERVICE_KEY"]
    sb = create_client(supabase_url, supabase_key)
    sources = sb.table("sources").select("*").eq("enabled", True).execute().data

    session = requests.Session()
    pacer = RequestPacer(ARTICLE_REQUEST_INTERVAL_SECONDS)
    new_count = 0
    dup_count = 0
    err_count = 0

    for source in sources:
        source_name = source.get("name") or f"source-{source['id']}"
        try:
            source_new, source_dup = collect_source(sb, source, session, pacer)
            new_count += source_new
            dup_count += source_dup
            print(
                f"Source {source_name}: new={source_new} "
                f"dup={source_dup} feed_only={source_uses_feed_only(source)}"
            )
        except Exception as exc:
            err_count += 1
            print(f"Source {source_name}: error={type(exc).__name__}: {exc}")

    print(f"Done. new={new_count} dup={dup_count} err={err_count}")


if __name__ == "__main__":
    main()
