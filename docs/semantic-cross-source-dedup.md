# Cross-source event deduplication

The analyzer now compares each AI candidate with a bounded history of the last
30 days of articles that were proposed, sent, or approved. The history is
limited to 50 rows and is passed into the same analysis request; no second AI
call is needed.

The model distinguishes:

- a different article about the same concrete incident (`is_duplicate_event`;
  blocked when there is no material update), and
- a substantial new development of an already covered event
  (`is_material_update`; still allowed into the editor queue).

The decision is validated in Python. An article id is accepted only when it was
actually present in the supplied history, so malformed model output cannot hide
an unrelated article. Duplicate decisions are stored in
`article_analysis`; rejected queue items are not treated as positive or
negative editorial-topic examples.

The comparison history includes queue statuses `pending`, `sent`,
`awaiting_feedback`, `correction_pending`, `publishing`, `approved`, and
`published`. Rejected stories are excluded because a rejection alone does not
mean that the underlying event was already covered.
