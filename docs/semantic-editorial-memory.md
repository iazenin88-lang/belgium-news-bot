# Semantic editorial memory

## Product objective

The end state is autonomous editorial approval: the owner should not need to
review routine candidates. Automation may be enabled only after measured replay
and shadow-mode results show both:

1. articles proposed for publication are consistently publishable; and
2. genuinely good articles are not silently rejected by the system.

The channel covers life in Belgium for Russian- and Ukrainian-speaking migrants
in Belgium. A foreign or global story is not relevant merely because it might
affect the EU. It needs a concrete direct consequence for Belgium or a concrete
effect on the rights, obligations, or status of that audience in Belgium.

## Current phase: semantic memory in shadow mode

Every applied `approved` or `topic_mismatch` decision receives a multilingual
embedding using the profile
`text-embedding-3-small:512:editorial-v1`. Before the main relevance model sees a
new candidate, the system retrieves up to five semantically similar approvals
and five semantically similar topic rejections from the complete decision
history. These examples are added to the relevance prompt.

The system also writes a balanced nearest-neighbour prediction to
`editorial_memory_predictions`. `mode = 'shadow'` is an invariant: the semantic
score cannot itself reject or publish an article. A likely approval may only
rescue a borderline article into the main AI evaluation; the main relevance
model and editor workflow still make the actual decision.

Semantic scoring runs before deterministic prefiltering so false negatives are
observable. All non-empty, non-hard-rejected items from Belgian domestic news
sources reach the main AI regardless of the city, commune, district, province,
or region named in the story.

Retrieval uses exact cosine distance while the labelled dataset is small. This
avoids approximate-index recall loss during safety evaluation.

## Evaluation and activation gates

Historical evaluation must be leave-one-out: the article being predicted is
excluded from its own neighbours. Evaluation must report approval retention,
topic-rejection recall, false negatives, false positives, and uncertain cases.

No automatic rejection should be activated merely because aggregate accuracy
looks good. Each false negative must be inspected, and the activation threshold
must protect good news. Live shadow predictions must also be compared with new
human decisions before moving to active mode.

## Fine-tuned personal model milestone

At about 500 reliable relevance decisions, create a versioned supervised
fine-tuning dataset with separate training and held-out evaluation sets. Compare
the fine-tuned model against the semantic-memory baseline on the same metrics.
Adopt it only if it materially improves the two product objectives above.

Semantic memory remains useful after fine-tuning because it supplies recent,
auditable examples and adapts immediately to new decisions without retraining.
