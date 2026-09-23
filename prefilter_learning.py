"""Human-approved pre-filter learning helpers."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Iterable


MIN_TRAINING_DECISIONS = 50
MAX_TERMS = 30
MAX_TRAINING_EXAMPLES = 200
MAX_TRAINING_PROMPT_CHARS = 100_000

BELGIAN_DOMESTIC_SOURCES = {
    "brussels times",
    "hln",
    "vrt nws",
}

BELGIAN_AUTHORITY_TERMS = {
    "administration communale",
    "bourgmestre",
    "brandweer",
    "burgemeester",
    "city council",
    "commune",
    "fire service",
    "gemeente",
    "inspectie",
    "inspection",
    "municipality",
    "police",
    "politie",
    "pompiers",
    "stadsbestuur",
}

BELGIAN_ENFORCEMENT_TERMS = {
    "ban",
    "brandveiligheid",
    "closed",
    "closure",
    "fermé",
    "fermée",
    "fermeture",
    "fire safety",
    "interdiction",
    "licence",
    "license",
    "permis",
    "sanction",
    "sanctie",
    "gesloten",
    "sluiting",
    "uitbatingsvergunning",
    "vergunning",
    "verbod",
}

PROPOSAL_TEXT_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "prefilter_policy_proposal",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "rationale": {"type": "string"},
                "positive_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "negative_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "summary",
                "rationale",
                "positive_terms",
                "negative_terms",
            ],
            "additionalProperties": False,
        },
    }
}

TERM_STOPWORDS = {
    "about", "after", "against", "also", "before", "belgian", "belgium",
    "from", "have", "into", "more", "new", "says", "that", "their",
    "there", "these", "this", "were", "will", "with",
    "aan", "als", "bij", "daar", "deze", "door", "heeft", "maar", "meer",
    "naar", "niet", "nieuwe", "onder", "over", "tegen", "voor", "wordt",
    "zijn", "zich",
    "après", "avant", "avec", "belgique", "dans", "elle", "pour", "plus",
    "sont", "cette",
    "deutschland", "mehr", "nach", "sind", "über", "gegen",
}

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

ГЕОГРАФИЧЕСКАЯ НЕЙТРАЛЬНОСТЬ — ОБЯЗАТЕЛЬНОЕ ПРАВИЛО:
- канал не отдаёт предпочтение отдельным городам, коммунам, районам, провинциям
  или регионам Бельгии;
- никогда не предлагай название бельгийского места как положительный или
  отрицательный сигнал релевантности;
- локальная привязка сама по себе не делает новость ни подходящей, ни
  неподходящей: решающими являются содержание, необычность, значимость и
  интерес события;
- positive_terms оставляй пустым: хорошие материалы теперь распознаются
  семантической памятью и основным AI-анализом, а не географическими словами.

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
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
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


def _contains_whole_term(text: str, term: str) -> bool:
    pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def is_belgian_domestic_source(source_name: str) -> bool:
    """Return whether the source primarily reports news from inside Belgium."""
    source = " ".join((source_name or "").lower().split())
    return source in BELGIAN_DOMESTIC_SOURCES


def has_belgian_enforcement_signal(source_name: str, text: str) -> bool:
    """Detect a concrete Belgian administrative or regulatory action.

    The two-signal requirement deliberately excludes ordinary police and crime
    reports.  It only sends an item to AI when a domestic Belgian source names
    both a public authority and a concrete closure, permit, fire-safety, ban,
    or sanction action.
    """
    if not is_belgian_domestic_source(source_name):
        return False
    lowered = (text or "").lower()
    has_authority = any(
        _contains_whole_term(lowered, term) for term in BELGIAN_AUTHORITY_TERMS
    )
    has_enforcement = any(
        _contains_whole_term(lowered, term) for term in BELGIAN_ENFORCEMENT_TERMS
    )
    return has_authority and has_enforcement


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


def _source_terms(text: str) -> set[str]:
    """Return exact reusable words and short phrases from an article."""
    words = re.findall(r"[^\W\d_][\w'’-]*", text.lower(), flags=re.UNICODE)
    words = [word.strip("'’-") for word in words if word.strip("'’-")]
    terms = {
        word
        for word in words
        if len(word) >= 5 and word not in TERM_STOPWORDS
    }
    for size in (2, 3):
        for index in range(len(words) - size + 1):
            phrase_words = words[index:index + size]
            phrase = " ".join(phrase_words)
            if (
                len(phrase) <= 80
                and any(
                    len(word) >= 5 and word not in TERM_STOPWORDS
                    for word in phrase_words
                )
            ):
                terms.add(phrase)
    return terms


def complete_policy_coverage(
    rows: Iterable[dict[str, Any]],
    policy: dict[str, Any],
    minimum_decline_rejection: float = 0.20,
) -> dict[str, Any]:
    """Complete an AI proposal with exact, replay-safe source phrases."""
    labelled = [
        row
        for row in rows
        if row.get("feedback_type") in ("approved", "topic_mismatch")
        and row.get("status") == "applied"
    ]
    approved_texts = [
        decision_text(row)
        for row in labelled
        if row.get("feedback_type") == "approved"
    ]
    declined_texts = [
        decision_text(row)
        for row in labelled
        if row.get("feedback_type") == "topic_mismatch"
    ]
    if not approved_texts or not declined_texts:
        return policy

    # Positive keyword rules are deliberately disabled. They previously
    # overfit to a few Belgian place names and made location a proxy for
    # relevance. Positive learning now belongs to semantic memory and the
    # main AI evaluation; this deterministic policy only keeps replay-safe
    # exclusions.
    positive_terms: list[str] = []

    coverage: dict[str, set[int]] = defaultdict(set)
    for index, text in enumerate(declined_texts):
        for term in _source_terms(text):
            if not any(term in approved for approved in approved_texts):
                coverage[term].add(index)

    existing = [
        term
        for term in policy.get("negative_terms", [])
        if term in coverage
    ]
    negative_terms: list[str] = []
    covered: set[int] = set()
    for term in sorted(existing, key=lambda item: (-len(coverage[item]), len(item))):
        if term not in negative_terms:
            negative_terms.append(term)
            covered.update(coverage[term])

    target = max(
        1,
        int(len(declined_texts) * minimum_decline_rejection + 0.9999),
    )
    candidates = set(coverage) - set(negative_terms)
    while len(covered) < target and candidates and len(negative_terms) < MAX_TERMS:
        best = max(
            candidates,
            key=lambda term: (
                len(coverage[term] - covered),
                len(coverage[term]),
                -len(term.split()),
                -len(term),
            ),
        )
        candidates.remove(best)
        if not (coverage[best] - covered):
            break
        negative_terms.append(best)
        covered.update(coverage[best])

    added = [term for term in negative_terms if term not in existing]
    rationale = str(policy.get("rationale") or "").strip()
    if added:
        rationale = (
            rationale
            + " Историческая проверка дополнила список точными признаками "
            "из отклонённых источников, отсутствующими в одобренных."
        ).strip()[:3000]
    return {
        **policy,
        "rationale": rationale,
        "positive_terms": positive_terms,
        "negative_terms": negative_terms[:MAX_TERMS],
    }
