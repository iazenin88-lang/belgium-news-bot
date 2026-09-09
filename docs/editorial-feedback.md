# Editorial feedback loop

The editor queue is a versioned state machine. Telegram only collects the
decision and comment; `analyzer.py` remains the single place that calls OpenAI
and accounts for its cost.

```mermaid
stateDiagram-v2
    pending --> sent: notifier
    sent --> publishing: publish
    publishing --> approved: Telegram accepted post
    sent --> awaiting_feedback: reject
    awaiting_feedback --> rejected: topic or other reason
    awaiting_feedback --> correction_pending: text correction
    correction_pending --> pending: Immediate AI rewrite and revision + 1
    pending --> sent: Immediate correction notifier
```

## Feedback types

+ `text_correction`: the editor comment is used to rewrite the current draft
  against the source article. A dedicated workflow is dispatched immediately;
  it requeues the corrected draft with a new revision and sends that version
  straight back to this Telegram editor chat. The publication buttons remain,
  so the editor still approves or rejects the corrected version.
- `topic_mismatch`: the rejected draft and editor comment become a negative
  example for future relevance decisions.
- `other_rejection`: the reason is stored for audit, but never changes topic
  selection.
- `approved`: recorded automatically as a positive example to keep negative
  feedback from becoming an overly broad topic ban.

Only callbacks for the latest queue revision and Telegram message are accepted.
Database functions claim and apply corrections atomically, so overlapping
scheduled jobs cannot process the same correction twice. The notifier claims a
pending queue row before sending it, so an immediate correction and the regular
pipeline cannot deliver the same revision twice.
