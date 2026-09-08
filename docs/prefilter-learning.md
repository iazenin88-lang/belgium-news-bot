# Human-approved pre-filter learning

The analyzer starts in exploration mode. It keeps only the empty-content and
obvious hard-topic rejections; borderline articles reach the existing AI
relevance decision, not the editor automatically.

Only approved and topic-mismatch decisions train relevance. Text corrections
remain style-only. After at least 50 relevance decisions, the model may propose
bounded positive and negative phrases. The analyzer replays the proposal
against the labelled history and requires at least 95% approval retention
before sending it to Telegram.

The editor can inspect, activate, or reject the proposal in the editor chat.
Activation ends exploration mode and is atomic. Every proposal and decision is
versioned in the database.
