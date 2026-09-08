# VRT source access

The collector uses VRT's public Atom feed:

`https://www.vrt.be/vrtnws/nl.rss.articles.xml`

No VRT account, cookies, or browser automation are used.

For VRT, the collector stores the title and abstract supplied by the feed and
does not open every linked article page. This produces one VRT feed request per
scheduled pipeline run.

For sources that require article-page extraction, duplicate records are checked
before any page request. New article-page requests use an identifying user agent
and are spaced at least three seconds apart by default. The interval can be
increased through `ARTICLE_REQUEST_INTERVAL_SECONDS`.

If the VRT feed URL must be rolled back, run `docs/rollback-vrt-feed.sql` and
redeploy the preceding repository version.
