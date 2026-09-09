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
- the initial import is limited to the newest 40 sitemap entries;
- later runs request details only for articles not already in Supabase;
- no account, cookie, browser automation, proxy or identity masking is used.

The delay can be increased with `ARTICLE_REQUEST_INTERVAL_SECONDS`. The initial
window can be changed with `BRUSSELS_TIMES_ENTRY_LIMIT`, but reducing request
load should take priority over importing old articles.

To restore the preceding, non-working source URL, run
`docs/rollback-brussels-times-source.sql` and redeploy the preceding repository
version.
