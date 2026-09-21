"""Human-approved pre-filter learning helpers."""

from __future__ import annotations

import json
from typing import Any, Iterable


MIN_TRAINING_DECISIONS = 50
MAX_TERMS = 30
MAX_TRAINING_EXAMPLES = 200
MAX_TRAINING_PROMPT_CHARS = 100_000

PROPOSAL_SYSTEM_PROMPT = """
Ты предлагаешь безопасное обновление предварительного фильтра новостей
русскоязычного Telegram-канала о жизни в Бельгии.

Используй только решения редактора «опубликовано» и «не подходит тематика».
Исправления текста не относятся к тематике. Предложи короткие слова или фразы,
которые встречаются в исходных заголовках/описаниях и помогают пропускать
одобренные темы либо отсекать отклонённые. Не предлагай код, SQL или регулярные
выражения. Не создавай широкий запрет из одного примера.

Все отрицательные фразы должны дословно встречаться в исходном тексте хотя бы
одной отклонённой новости и не должны встречаться ни в одной одобренной новости.
Старайся покрыть отрицательными фразами не менее 20% отклонённых новостей,
сохранив не менее 95% одобренных. В примерах могут быть разные языки — добавляй
отдельные точные фразы для нужных языков. Не используй комментарий редактора как
фразу фильтра: комментарий объясняет решение, но фильтр применяется к новости.

Верни строго JSON:
{
  "summary": "Краткое описание изменения",
  "rationale": "Почему это следует из решений редактора",
  "positive_terms": ["phrase"],
  "negative_terms": ["phrase"]
}
""".strip()


def parse_policy_proposal(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = text.removeprefix(fence + "json").removeprefix(fence)
        text = text.removesuffix(fence).strip()
    data = json.loads(text)

    def terms(name: str) -> list[str]:
        value = data.get(name, [])
        if not isinstance(value, list):
            raise ValueError(f"{name} must be a list")
        result: list[str] = []
        for item in value:
            term = " ".join(str(item).lower().split())
            if 2 <= len(term) <= 80 and term not in result:
                result.append(term)
        return result[:MAX_TERMS]

    summary = " ".join(str(data.get("summary") or "").split())[:500]
    rationale = " ".join(str(data.get("rationale") or "").split())[:3000]
    positive = terms("positive_terms")
    negative = terms("negative_terms")
    if not summary or not rationale or not (positive or negative):
        raise ValueError("proposal is incomplete")
    return {
        "summary": summary,
        "rationale": rationale,
        "positive_terms": positive,
        "negative_terms": negative,
    }


def decision_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in ("source_title", "source_summary", "draft_title", "draft_text")
    ).lower()


def policy_prefilter_decision(
    text: str, policy: dict[str, Any] | None
) -> tuple[bool | None, str]:
    """Return True/False for a learned match, or None when policy is silent."""
    if not policy:
        return None, ""
    lowered = text.lower()
    positives = [
        term for term in policy.get("positive_terms", []) if term and term in lowered
    ]
    negatives = [
        term for term in policy.get("negative_terms", []) if term and term in lowered
    ]
    if positives:
        return True, f"Learned positive signal: {positives[0]}"
    if negatives:
        return False, f"Learned negative signal: {negatives[0]}"
    return None, ""


def evaluate_policy(
    rows: Iterable[dict[str, Any]], policy: dict[str, Any]
) -> dict[str, Any]:
    approvals = declines = retained = rejected = 0
    for row in rows:
        kind = row.get("feedback_type")
        if kind not in ("approved", "topic_mismatch") or row.get("status") != "applied":
            continue
        decision, _ = policy_prefilter_decision(decision_text(row), policy)
        if kind == "approved":
            approvals += 1
            if decision is not False:
                retained += 1
        else:
            declines += 1
            if decision is False:
                rejected += 1
    return {
        "approvals": approvals,
        "topic_declines": declines,
        "approval_retention": retained / approvals if approvals else 0.0,
        "decline_rejection": rejected / declines if declines else 0.0,
    }


def proposal_is_safe(metrics: dict[str, Any]) -> bool:
    return (
        metrics.get("approvals", 0) > 0
        and metrics.get("topic_declines", 0) > 0
        and metrics.get("approval_retention", 0.0) >= 0.95
        and metrics.get("decline_rejection", 0.0) >= 0.20
    )


def format_training_examples(
    rows: Iterable[dict[str, Any]],
    limit: int = MAX_TRAINING_PROMPT_CHARS,
) -> str:
    """Format a balanced view of the whole labelled history.

    Rows arrive newest-first.  The old implementation stopped after 12k
    characters, so the model often saw only a handful of the latest decisions
    while replay evaluated its proposal against the full history.  Interleaving
    both labels prevents either class from disappearing when the size cap is
    reached.
    """
    labelled = [
        row
        for row in rows
        if row.get("feedback_type") in ("approved", "topic_mismatch")
        and row.get("status") == "applied"
    ]
    approvals = [row for row in labelled if row.get("feedback_type") == "approved"]
    declines = [
        row for row in labelled if row.get("feedback_type") == "topic_mismatch"
    ]

    balanced: list[dict[str, Any]] = []
    for index in range(max(len(approvals), len(declines))):
        if index < len(approvals):
            balanced.append(approvals[index])
        if index < len(declines):
            balanced.append(declines[index])
        if len(balanced) >= MAX_TRAINING_EXAMPLES:
            break

    parts = [
        f"ВСЕГО РЕШЕНИЙ: {len(labelled)}; "
        f"ОПУБЛИКОВАНО: {len(approvals)}; "
        f"НЕ ПОДХОДИТ ТЕМАТИКА: {len(declines)}"
    ]
    for row in balanced:
        kind = row.get("feedback_type")
        label = "ОПУБЛИКОВАНО" if kind == "approved" else "НЕ ПОДХОДИТ ТЕМАТИКА"
        block = (
            f"{label}\n"
            f"Заголовок: {str(row.get('source_title') or '')[:240]}\n"
            f"Описание: {str(row.get('source_summary') or '')[:360]}\n"
            f"Комментарий: {str(row.get('editor_comment') or '')[:240]}"
        )
        candidate = "\n\n".join(parts + [block])
        if len(candidate) > limit:
            break
        parts.append(block)
    return "\n\n".join(parts)


def remove_unsafe_negative_terms(
    rows: Iterable[dict[str, Any]], policy: dict[str, Any]
) -> dict[str, Any]:
    """Drop negative phrases that occur in any approved historical item."""
    approved_texts = [
        decision_text(row)
        for row in rows
        if row.get("feedback_type") == "approved" and row.get("status") == "applied"
    ]
    safe_negative = [
        term
        for term in policy.get("negative_terms", [])
        if not any(term in text for text in approved_texts)
    ]
    return {**policy, "negative_terms": safe_negative}
