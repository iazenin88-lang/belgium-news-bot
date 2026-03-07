import os
import re
import hashlib
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import feedparser
from supabase import create_client

TRACKING_PARAMS_PREFIXES = ("utm_",)
TRACKING_PARAMS_EXACT = {"fbclid", "gclid"}

def canonicalize_url(url):
    p = urlparse(url)
    query_items = []

    for k, v in parse_qsl(p.query):
        if k.lower().startswith("utm_"):
            continue
        if k.lower() in TRACKING_PARAMS_EXACT:
            continue
        query_items.append((k, v))

    query = urlencode(query_items)

    return urlunparse(("https", p.netloc.lower(), p.path, "", query, ""))


def fingerprint(source_id, title, url):
    base = f"{source_id}|{title}|{url}"
    return hashlib.sha256(base.encode()).hexdigest()


def main():

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]

    sb = create_client(url, key)

    sources = sb.table("sources").select("*").eq("enabled", True).execute().data

    for src in sources:

        feed = feedparser.parse(src["url"])

        for e in feed.entries[:50]:

            link = e.get("link")
            title = e.get("title")

            if not link:
                continue

            canon = canonicalize_url(link)

            fp = fingerprint(src["id"], title, canon)

            row = {
                "source_id": src["id"],
                "original_url": link,
                "canonical_url": canon,
                "title": title,
                "summary": e.get("summary"),
                "fingerprint": fp,
            }

            try:
                sb.table("articles").insert(row).execute()
            except:
                pass


if __name__ == "__main__":
    main()
