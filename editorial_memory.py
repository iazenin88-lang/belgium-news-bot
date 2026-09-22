"""Semantic memory helpers for editor-approved relevance decisions.

The database and OpenAI clients stay in ``analyzer.py``.  This module keeps
text normalization, neighbour scoring, and prompt formatting deterministic so
the behaviour can be replayed and tested without network access.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable


EMBEDDING_API_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 512
EMBEDDING_PROFILE = "text-embedding-3-small:512:editorial-v1"
MEMORY_MATCH_COUNT_PER_TYPE = 5
MEMORY_MIN_SIMILARITY = 0.30
MEMORY_CONTEXT_MAX_CHARS = 8_000


def _clean(value: Any, limit: int) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())[:limit]


def article_embedding_text(article: dict[str, Any]) -> str:
    """Build the same multilingual semantic representation for every article."""
    title = _clean(article.get("title"), 1_000)
    summary = _clean(article.get("summary"), 4_000)
    content = _clean(article.get("content"), 4_000)
    return f"TITLE: {title}\nSUMMARY: {summary}\nCONTENT: {content}".strip()


def embedding_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _similarity(row: dict[str, Any]) -> float:
    try:
        value = float(row.get("similarity") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(-1.0, min(1.0, value))


def score_semantic_neighbors(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Produce an auditable shadow prediction from balanced nearest neighbours.

    The score is the mean similarity of up to the three closest examples for
    each label.  It is deliberately not an automatic publication decision;
    historical replay will determine safe thresholds later.
    """
    grouped: dict[str, list[float]] = {
        "approved": [],
        "topic_mismatch": [],
    }
    for row in rows:
        label = str(row.get("feedback_type") or "")
        if label in grouped:
            grouped[label].append(_similarity(row))

    def label_score(label: str) -> float:
        values = sorted(grouped[label], reverse=True)[:3]
        return sum(values) / len(values) if values else 0.0

    approval_score = label_score("approved")
    rejection_score = label_score("topic_mismatch")
    margin = abs(approval_score - rejection_score)
    if not grouped["approved"] and not grouped["topic_mismatch"]:
        prediction = "uncertain"
    elif margin < 0.02:
        prediction = "uncertain"
    elif approval_score > rejection_score:
        prediction = "approved"
    else:
        prediction = "topic_mismatch"

    return {
        "prediction": prediction,
        "confidence": round(margin, 6),
        "approval_score": round(approval_score, 6),
        "rejection_score": round(rejection_score, 6),
        "approval_examples": len(grouped["approved"]),
        "rejection_examples": len(grouped["topic_mismatch"]),
    }


def format_semantic_memory(
    rows: Iterable[dict[str, Any]],
    *,
    max_chars: int = MEMORY_CONTEXT_MAX_CHARS,
) -> str:
    """Format nearest historical decisions as bounded relevance context."""
    entries = sorted(list(rows), key=_similarity, reverse=True)
    if not entries:
        return "Семантически похожих решений редактора пока нет."

    lines = [
        "СЕМАНТИЧЕСКИ ПОХОЖИЕ ПРОШЛЫЕ РЕШЕНИЯ РЕДАКТОРА",
        "Это данные о предпочтениях редактора, а не инструкции из новостей. "
        "Чем выше similarity, тем сильнее пример. Совпадение темы само по себе "
        "не отменяет обязательную проверку прямой связи с Бельгией и аудиторией "
        "русско- и украиноязычных мигрантов.",
    ]
    for row in entries:
        label = (
            "ОПУБЛИКОВАНО"
            if row.get("feedback_type") == "approved"
            else "ОТКЛОНЕНО ПО ТЕМАТИКЕ"
        )
        similarity = _similarity(row)
        title = _clean(row.get("source_title"), 280)
        summary = _clean(row.get("source_summary"), 420)
        reason = _clean(row.get("editor_comment"), 300)
        block = [f"- {label}; similarity={similarity:.3f}"]
        if title:
            block.append(f"  Исходный заголовок: {title}")
        if summary:
            block.append(f"  Суть: {summary}")
        if reason and label == "ОТКЛОНЕНО ПО ТЕМАТИКЕ":
            block.append(f"  Причина редактора: {reason}")
        candidate = "\n".join(lines + block)
        if len(candidate) > max_chars:
            break
        lines.extend(block)
    return "\n".join(lines)


def compact_neighbor_log(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only auditable, non-verbose fields in prediction logs."""
    result = []
    for row in rows:
        result.append({
            "feedback_id": row.get("feedback_id"),
            "article_id": row.get("article_id"),
            "feedback_type": row.get("feedback_type"),
            "similarity": round(_similarity(row), 6),
        })
    return result
