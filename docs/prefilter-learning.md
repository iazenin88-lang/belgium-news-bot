# Human-approved pre-filter learning

The analyzer starts in exploration mode. It keeps only the empty-content and
obvious hard-topic rejections; borderline articles reach the existing AI
relevance decision, not the editor automatically.

Only approved and topic-mismatch decisions train relevance. Text corrections
remain style-only. After at least 50 relevance decisions, the model may propose
bounded positive and negative phrases. The analyzer replays the proposal
against the labelled history and requires at least 95% approval retention
before sending it to Telegram.

The proposal prompt includes a balanced view of the labelled history instead
of only the first few newest records. If a candidate fails replay, the analyzer
retries immediately with the measured retention and rejection coverage. Terms
that occur in any approved item are removed from the negative list before the
replay.

The OpenAI Responses call uses strict JSON Schema output. A malformed response
is still treated as a retryable proposal attempt rather than aborting the whole
learning step.

If generated terms still cover less than 20% of topic declines, a deterministic
fallback greedily adds exact words or short phrases from declined source text.
Fallback terms are rejected when they occur in any approved source. This makes
proposal generation complete without weakening the 95% approval-retention gate;
the editor still sees replay metrics and must activate the policy.

The editor can inspect, activate, or reject the proposal in the editor chat.
Activation ends exploration mode and is atomic. Every proposal and decision is
versioned in the database.
