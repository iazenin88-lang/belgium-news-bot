# Brussels Times source access

The previous source URL, `https://www.brusselstimes.com/feed`, serves an HTML
page rather than RSS or Atom. The collector therefore uses the publisher's
Google News sitemap instead:

`https://www.brusselstimes.com/google-news-sitemap.xml`

The sitemap supplies the article URL, title and publication time. For each
candidate, the collector checks its fingerprint in Supabase before making any
additional request. Only genuinely new candidates are sent to the publisher's
first-party article API, where the collector reads `seo_description` (or `sub`
as a fallback). It does not store the API's full article body.

Access safeguards:

- requests use the identifying `BelgiumNewsBot/1.0` user agent;
- detail requests are spaced at least three seconds apart by default;
- the initial import is limited to the newest 120 sitemap entries;
- later runs request details only for articles not already in Supabase;
- no account, cookie, browser automation, proxy or identity masking is used.

The delay can be increased with ARTICLE_REQUEST_INTERVAL_SECONDS. The initial
window can be changed with BRUSSELS_TIMES_ENTRY_LIMIT. Only candidates that are
not already in Supabase trigger an API request, and requests remain spaced by
at least three seconds.

## Historical articles

Google News sitemaps are short-lived. To recover an older article, start the
Full news pipeline workflow manually and put one or more explicit Brussels
Times URLs into its optional brussels_times_backfill_urls input (one URL per
line, maximum 20). The backfill validates the publisher article ID, stores only
the publisher's short abstract, checks URL/ID duplicates before and after the
API request, and then lets the normal analyzer and notifier process the new
row. It does not broaden normal scheduled crawling.

To restore the preceding, non-working source URL, run
docs/rollback-brussels-times-source.sql and redeploy the preceding repository
version.
