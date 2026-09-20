"""Backfill explicitly selected historical Brussels Times articles.

The publisher's Google News sitemap is intentionally short-lived, so an
article that predates the first successful source run cannot be recovered by
simply increasing the sitemap window. This runner accepts only explicit
Brussels Times URLs, stores the publisher's short abstract, and is idempotent
by URL and article ID.
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlparse

import requests
from supabase import create_client

from collector import (
    ARTICLE_REQUEST_INTERVAL_SECONDS,
    BRUSSELS_TIMES_DOMAIN,
    RequestPacer,
    brussels_times_article_id,
    canonicalize_url,
    clean_summary,
    fetch_brussels_times_article,
    fingerprint,
    hostname_matches,
    source_is_brussels_times,
)


MAX_BACKFILL_URLS = 20


def parse_backfill_urls(
    raw: str = "",
    explicit_urls: list[str] | None = None,
) -> list[str]:
    """Parse, validate and de-duplicate explicit URLs."""
    values = list(explicit_urls or [])
    if raw.strip():
        values.extend(re.split(r"[\n,]+", raw))

    urls: list[str] = []
    seen: set[str] = set()
    for value in values:
        url = value.strip()
        if not url:
            continue
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not hostname_matches(
            parsed.hostname,
            BRUSSELS_TIMES_DOMAIN,
        ):
            raise ValueError(
                "Backfill accepts only http(s) URLs on brusselstimes.com: "
                f"{url}"
            )
        canonical = canonicalize_url(url)
        if canonical in seen:
            continue
        seen.add(canonical)
        urls.append(url)

    if len(urls) > MAX_BACKFILL_URLS:
        raise ValueError(
            f"Backfill accepts at most {MAX_BACKFILL_URLS} URLs per run"
        )
    return urls


def parse_published_at(value: Any) -> str | None:
    """Convert the publisher's timestamp to an explicit UTC ISO timestamp."""
    if not value:
        return None
    text = str(value).strip().replace(" ", "T")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def build_backfill_row(
    source_id: int,
    original_url: str,
    payload: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Build the same article shape as the normal Brussels Times collector."""
    article_id = brussels_times_article_id(original_url)
    if article_id is None:
        raise ValueError(f"URL has no numeric Brussels Times article ID: {original_url}")

    response_id = payload.get("id")
    if response_id is not None and str(response_id) != str(article_id):
        raise ValueError(
            f"Publisher API returned article {response_id}, expected {article_id}"
        )

    title = str(payload.get("title") or "").strip()
    if not title:
        raise ValueError(f"Publisher API returned no title for article {article_id}")

    published_at = parse_published_at(payload.get("published"))
    if not published_at:
        raise ValueError(
            f"Publisher API returned no valid publication time for article {article_id}"
        )

    canonical_url = canonicalize_url(original_url)
    summary = (
        clean_summary(payload.get("seo_description"))
        or clean_summary(payload.get("sub"))
    )
    row = {
        "source_id": source_id,
        "original_url": original_url,
        "canonical_url": canonical_url,
        "title": title,
        "summary": summary,
        "published_at": published_at,
        "fingerprint": fingerprint(source_id, title, canonical_url),
        "content": None,
    }
    return article_id, row


def find_existing_article(
    sb,
    source_id: int,
    original_url: str,
    canonical_url: str,
    article_id: int,
) -> int | None:
    """Find a prior import even if the site's category path changed."""
    for column, value in (
        ("canonical_url", canonical_url),
        ("original_url", original_url),
    ):
        rows = (
            sb.table("articles")
            .select("id")
            .eq("source_id", source_id)
            .eq(column, value)
            .limit(1)
            .execute()
            .data
            or []
        )
        if rows:
            return int(rows[0]["id"])

    # The category prefix is not stable on this publisher. The numeric ID is.
    id_pattern = f"%/{article_id}/%"
    for column in ("canonical_url", "original_url"):
        rows = (
            sb.table("articles")
            .select("id")
            .eq("source_id", source_id)
            .like(column, id_pattern)
            .limit(1)
            .execute()
            .data
            or []
        )
        if rows:
            return int(rows[0]["id"])
    return None


def load_brussels_times_source(sb) -> dict[str, Any]:
    rows = (
        sb.table("sources")
        .select("id,name,url,enabled")
        .eq("name", "Brussels Times")
        .limit(2)
        .execute()
        .data
        or []
    )
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one enabled Brussels Times source, found {len(rows)}"
        )
    source = rows[0]
    if not source.get("enabled") or not source_is_brussels_times(source):
        raise RuntimeError("Brussels Times source is disabled or misconfigured")
    return source


def backfill_urls(
    sb,
    urls: list[str],
    *,
    session: requests.Session | None = None,
    pacer: RequestPacer | None = None,
    metadata_fetcher: Callable[[str, requests.Session], dict | None]
    = fetch_brussels_times_article,
) -> dict[str, int]:
    """Import URLs and return inserted/skipped/failed counts."""
    if not urls:
        return {"inserted": 0, "skipped": 0, "failed": 0}

    source = load_brussels_times_source(sb)
    source_id = int(source["id"])
    session = session or requests.Session()
    pacer = pacer or RequestPacer(ARTICLE_REQUEST_INTERVAL_SECONDS)
    stats = {"inserted": 0, "skipped": 0, "failed": 0}

    for original_url in urls:
        try:
            article_id = brussels_times_article_id(original_url)
            if article_id is None:
                raise ValueError(
                    f"URL has no numeric Brussels Times article ID: {original_url}"
                )
            canonical_url = canonicalize_url(original_url)
            existing_id = find_existing_article(
                sb,
                source_id,
                original_url,
                canonical_url,
                article_id,
            )
            if existing_id is not None:
                print(
                    f"Backfill skip article_id={existing_id}: "
                    f"publisher article {article_id} already exists"
                )
                stats["skipped"] += 1
                continue

            pacer.wait()
            payload = metadata_fetcher(original_url, session)
            if payload is None:
                raise RuntimeError(
                    f"Publisher API returned no valid metadata for article {article_id}"
                )
            _, row = build_backfill_row(source_id, original_url, payload)

            # Re-check after the API request to keep the operation safe if a
            # normal collector run inserted the same article concurrently.
            existing_id = find_existing_article(
                sb,
                source_id,
                original_url,
                row["canonical_url"],
                article_id,
            )
            if existing_id is not None:
                print(
                    f"Backfill skip article_id={existing_id}: "
                    f"publisher article {article_id} arrived concurrently"
                )
                stats["skipped"] += 1
                continue

            sb.table("articles").insert(row).execute()
            print(
                f"Backfill inserted publisher article {article_id}: "
                f"{row['title']}"
            )
            stats["inserted"] += 1
        except Exception as exc:
            print(f"Backfill failed for {original_url}: {type(exc).__name__}: {exc}")
            stats["failed"] += 1

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill explicitly selected historical Brussels Times URLs"
    )
    parser.add_argument(
        "urls",
        nargs="*",
        help="One or more Brussels Times article URLs",
    )
    parser.add_argument(
        "--url",
        dest="option_urls",
        action="append",
        default=[],
        help="Additional URL; may be repeated",
    )
    args = parser.parse_args(argv)

    try:
        urls = parse_backfill_urls(
            os.getenv("BRUSSELS_TIMES_BACKFILL_URLS", ""),
            [*args.urls, *args.option_urls],
        )
    except ValueError as exc:
        parser.error(str(exc))

    if not urls:
        parser.error(
            "provide at least one URL or set BRUSSELS_TIMES_BACKFILL_URLS"
        )

    sb = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"],
    )
    stats = backfill_urls(sb, urls)
    print(
        "Backfill done. "
        f"inserted={stats['inserted']} "
        f"skipped={stats['skipped']} "
        f"failed={stats['failed']}"
    )
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
