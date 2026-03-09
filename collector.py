import os
import re
import hashlib
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import feedparser
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from supabase import create_client


TRACKING_PARAMS_PREFIXES = ("utm_",)
TRACKING_PARAMS_EXACT = {"fbclid", "gclid", "mc_cid", "mc_eid"}


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


def extract_text_from_html(url: str) -> str | None:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        r = requests.get(url, timeout=20, headers=headers)
        r.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    article = soup.find("article")
    if article:
        text = article.get_text(" ", strip=True)
    else:
        text = soup.get_text(" ", strip=True)

    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 300:
        return None

    return text[:50000]


def main():
    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_SERVICE_KEY"]
    sb = create_client(supabase_url, supabase_key)

    sources = sb.table("sources").select("*").eq("enabled", True).execute().data

    new_count = 0
    dup_count = 0
    err_count = 0

    for src in sources:
        sid = src["id"]
        feed_url = src["url"]

        try:
            feed = feedparser.parse(feed_url)
        except Exception:
            err_count += 1
            continue

        for e in feed.entries[:100]:
            original_url = e.get("link")
            if not original_url:
                continue

            canon = canonicalize_url(original_url)
            title = e.get("title", None)
            summary = e.get("summary", None)
            content = extract_text_from_html(canon)

            published_at = None
            if e.get("published_parsed"):
                import time
                published_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", e.published_parsed)

            fp = fingerprint(sid, title or "", canon)

            row = {
                "source_id": sid,
                "original_url": original_url,
                "canonical_url": canon,
                "title": title,
                "summary": summary,
                "published_at": published_at,
                "fingerprint": fp,
                "content": content,
            }

            try:
                sb.table("articles").insert(row).execute()
                new_count += 1
            except Exception:
                dup_count += 1

    print(f"Done. new={new_count} dup={dup_count} err={err_count}")


if __name__ == "__main__":
    main()
