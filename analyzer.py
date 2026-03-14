"""
analyzer.py

Назначение:
анализирует новые статьи из таблицы articles и сохраняет результат в article_analysis.

Дополнительно:
- считает стоимость OpenAI за текущий прогон
- пишет статистику в ai_runs
- обновляет ai_balance
- после прогона отправляет в Telegram сообщение о расходах

Важно:
- стоимость считается по заданным в коде тарифам
- остаток баланса является оценочным:
    remaining = starting_balance_usd - spent_total_usd
"""

import json
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import requests
from openai import OpenAI
from supabase import create_client


# -------------------------------------------------------
# Модель OpenAI
# -------------------------------------------------------
MODEL = "gpt-5-mini"


# -------------------------------------------------------
# Тарифы модели
#
# ВАЖНО:
# при необходимости поменяй вручную под актуальные цены.
# Значения ниже задаются в долларах за 1 миллион токенов.
# -------------------------------------------------------
INPUT_COST_PER_1M = Decimal("0.250000")
OUTPUT_COST_PER_1M = Decimal("2.000000")


# -------------------------------------------------------
# Telegram
# -------------------------------------------------------
TELEGRAM_API_BASE = "https://api.telegram.org"


# -------------------------------------------------------
# Системный prompt для AI
# -------------------------------------------------------
SYSTEM_PROMPT = """
Ты анализируешь новостные статьи для Telegram-канала на русском языке
для русскоязычных жителей Бельгии.

Нужно определить:
1. интересна ли новость этой аудитории
2. категория новости
3. важность по шкале 1-10
4. краткая причина
5. короткий пересказ на русском
6. короткий заголовок для Telegram
7. текст поста для Telegram

Считать релевантными в первую очередь:
- миграция, визы, ВНЖ, убежище, украинские беженцы
- жильё, аренда, коммунальные правила
- работа, зарплаты, налоги, пособия
- транспорт, школы, медицина, безопасность
- изменения законов и правил в Бельгии и ЕС, влияющие на жизнь в Бельгии
- важные новости о Брюсселе, Фландрии, Валлонии
- новости соседних стран, если они реально влияют на жизнь людей в Бельгии
- новости про Россию, россиян, русских, если они могут быть значимы
  для русскоязычной аудитории в Бельгии
- отдельные темы развлечений, если они реально заметны для русскоязычной аудитории в Бельгии
  или имеют практический/общественный контекст

Считать нерелевантными:
- обычные мировые новости без практической связи с жизнью в Бельгии
- спорт, криминальные мелочи, если нет практической пользы
- локальные мелочи без заметного влияния на читателей
- развлекательные новости низкой значимости без общественного или практического смысла

Категории используй только из списка:
migration, housing, work, taxes, transport, education, healthcare, social, politics, safety, europe, other

Верни строго JSON без пояснений.
"""

USER_TEMPLATE = """
Проанализируй статью.

Источник: {source_name}
Заголовок: {title}
Summary: {summary}
Content: {content}
URL: {url}

Верни JSON такого вида:
{{
  "is_relevant": true,
  "category": "migration",
  "importance_score": 8,
  "reason": "Коротко почему новость важна",
  "russian_summary": "Короткий пересказ на русском, 2-4 предложения.",
  "telegram_title": "Короткий заголовок",
  "telegram_text": "Готовый короткий текст для Telegram без markdown."
}}
"""


# -------------------------------------------------------
# Ключевые слова для pre-filter
# -------------------------------------------------------
HARD_REJECT_KEYWORDS = {
    "football", "soccer", "champions league", "premier league", "uefa",
    "basketball", "tennis tournament", "formula 1", "motogp",
    "movie star", "red carpet", "fashion week",
    "album release", "box office", "love island",
    "royal gossip", "tv show", "reality show",
    "transfer rumor", "match preview", "match report", "line-up",
    "livestream sports", "sports betting",
}

PASS_KEYWORDS = {
    "belgium", "belgian", "brussels", "flanders", "wallonia", "antwerp", "ghent",
    "belgië", "brussel", "vlaanderen", "wallonië", "gent", "antwerpen",

    "visa", "residence permit", "permit", "asylum", "refugee", "refugees",
    "migrant", "migrants", "immigration", "integration", "expat", "foreign worker",
    "temporary protection", "residency", "residence card",

    "housing", "rent", "rental", "landlord", "tenant", "mortgage",
    "salary", "wage", "employment", "job market", "unemployment",
    "tax", "taxes", "benefit", "benefits", "pension", "allowance",

    "school", "education", "transport", "rail", "train", "tram", "bus",
    "healthcare", "hospital", "doctor", "medicine", "insurance",
    "safety", "police", "court", "law", "legal",

    "ukrainian", "ukrainians", "ukraine",

    "russia", "russian", "russians",
    "россия", "россияне", "русские", "русский",

    "entertainment", "festival", "concert", "cinema", "music", "cultural event",
    "culture", "event", "events",
}

HIGH_SIGNAL_KEYWORDS = {
    "new law", "law", "rules", "policy", "ban", "decision", "court",
    "tax", "taxes", "visa", "permit", "pension", "housing", "rent",
    "transport", "strike", "school", "benefits", "immigration",
    "refugee", "refugees", "asylum", "healthcare", "insurance",
    "border", "customs", "residence", "residency", "work permit",
    "salary", "wage", "unemployment", "education", "police", "safety",
}


# -------------------------------------------------------
# Вспомогательные функции
# -------------------------------------------------------
def get_env(name: str, required: bool = True) -> str:
    """
    Получает переменную окружения.
    """
    value = os.environ.get(name)
    if required and not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value or ""


def get_supabase():
    """
    Создаёт клиент Supabase.
    """
    return create_client(
        get_env("SUPABASE_URL"),
        get_env("SUPABASE_SERVICE_KEY"),
    )


def get_openai():
    """
    Создаёт клиент OpenAI.
    """
    return OpenAI(api_key=get_env("OPENAI_API_KEY"))


def pick_text(response: Any) -> str:
    """
    Достаёт текст из ответа OpenAI Responses API.
    """
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text.strip()

    parts: list[str] = []

    try:
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue

            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    text = getattr(c, "text", "")
                    if text:
                        parts.append(text)
    except Exception:
        pass

    return "\n".join(parts).strip()


def normalize_text(value: Any, max_len: int = 4000) -> str:
    """
    Приводит значение к строке и обрезает по длине.
    """
    if value is None:
        return ""
    return str(value).strip()[:max_len]


def normalize_int(value: Any, default: int = 1) -> int:
    """
    Безопасно приводит значение к int и ограничивает диапазон 1..10.
    """
    try:
        n = int(value)
    except Exception:
        return default

    return max(1, min(10, n))


def contains_any(text: str, keywords: set[str]) -> bool:
    """
    Проверяет, содержит ли текст хотя бы одно ключевое слово.
    """
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)


def count_matches(text: str, keywords: set[str]) -> int:
    """
    Подсчитывает, сколько ключевых слов встретилось в тексте.
    """
    text_lower = text.lower()
    return sum(1 for keyword in keywords if keyword in text_lower)


def quantize_money(value: Decimal) -> Decimal:
    """
    Округляет денежное значение до 6 знаков после запятой.
    """
    return value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


def now_iso() -> str:
    """
    Возвращает текущее время в ISO UTC.
    """
    return datetime.now(timezone.utc).isoformat()


# -------------------------------------------------------
# Telegram helper
# -------------------------------------------------------
def telegram_send_message(bot_token: str, chat_id: str, text: str) -> None:
    """
    Отправляет сообщение в Telegram.
    """
    url = f"{TELEGRAM_API_BASE}/bot{bot_token}/sendMessage"

    response = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
        },
        timeout=30,
    )
    response.raise_for_status()


# -------------------------------------------------------
# Более строгий pre-filter перед OpenAI
# -------------------------------------------------------
def should_send_to_ai(article: dict[str, Any]) -> tuple[bool, str]:
    """
    Более строгий pre-filter перед OpenAI.
    """
    title = normalize_text(article.get("title"), 1000)
    summary = normalize_text(article.get("summary"), 4000)
    content = normalize_text(article.get("content"), 8000)

    combined = f"{title}\n{summary}\n{content}".strip()
    combined_lower = combined.lower()

    if not title and not summary and not content:
        return False, "Нет заголовка, summary и content"

    if contains_any(combined_lower, HARD_REJECT_KEYWORDS):
        return False, "Явно нерелевантная тема (спорт и т.п.)"

    pass_matches = count_matches(combined_lower, PASS_KEYWORDS)
    high_signal_matches = count_matches(combined_lower, HIGH_SIGNAL_KEYWORDS)

    summary_len = len(summary)
    content_len = len(content)

    practical_keywords = {
        "visa", "permit", "residence permit", "asylum", "refugee", "migrant",
        "housing", "rent", "tenant", "landlord", "salary", "wage", "employment",
        "tax", "taxes", "pension", "benefit", "benefits",
        "school", "education", "transport", "train", "tram", "bus",
        "healthcare", "hospital", "insurance", "doctor",
        "police", "court", "law", "legal", "safety",
        "temporary protection", "work permit",
    }

    practical_matches = count_matches(combined_lower, practical_keywords)

    if practical_matches >= 1 and (summary_len >= 80 or content_len >= 250):
        return True, "Есть practical-тема и достаточно содержательный текст"

    russia_keywords = {
        "russia", "russian", "russians",
        "россия", "россияне", "русские", "русский",
    }
    russia_matches = count_matches(combined_lower, russia_keywords)

    if russia_matches >= 1:
        if practical_matches >= 1:
            return True, "Россия/русские + practical-контекст"
        if high_signal_matches >= 2 and (summary_len >= 80 or content_len >= 250):
            return True, "Россия/русские + сильный новостной контекст"

    belgium_keywords = {
        "belgium", "belgian", "brussels", "flanders", "wallonia", "antwerp", "ghent",
        "belgië", "brussel", "vlaanderen", "wallonië", "gent", "antwerpen",
    }
    belgium_matches = count_matches(combined_lower, belgium_keywords)

    if belgium_matches >= 1 and (practical_matches >= 1 or high_signal_matches >= 2):
        return True, "Бельгийский контекст + практическая или сильная тема"

    europe_keywords = {
        "eu", "european union", "europe",
        "netherlands", "france", "germany", "luxembourg",
    }
    europe_matches = count_matches(combined_lower, europe_keywords)

    if europe_matches >= 1 and practical_matches >= 1 and (summary_len >= 80 or content_len >= 250):
        return True, "Европейский контекст с practical-углом"

    entertainment_keywords = {
        "entertainment", "festival", "concert", "cinema", "music",
        "cultural event", "culture", "event", "events",
    }
    entertainment_matches = count_matches(combined_lower, entertainment_keywords)

    if entertainment_matches >= 1:
        if belgium_matches >= 1 and (summary_len >= 80 or content_len >= 250):
            return True, "Развлекательная тема с бельгийским контекстом"
        if high_signal_matches >= 2 and (summary_len >= 100 or content_len >= 300):
            return True, "Развлекательная тема с сильным общественным контекстом"

    if high_signal_matches >= 3 and (summary_len >= 100 or content_len >= 300):
        return True, "Несколько сильных сигналов и содержательный текст"

    if summary_len < 60 and content_len < 180:
        return False, "Слишком мало содержательной информации"

    if pass_matches <= 1 and high_signal_matches <= 1 and practical_matches == 0:
        return False, "Недостаточно сигналов релевантности"

    return False, "Пограничный случай без достаточных оснований для AI"


# -------------------------------------------------------
# Считаем стоимость OpenAI-вызова
# -------------------------------------------------------
def extract_usage_tokens(response: Any) -> tuple[int, int]:
    """
    Пытается достать input/output tokens из ответа OpenAI.
    """
    input_tokens = 0
    output_tokens = 0

    usage = getattr(response, "usage", None)
    if usage:
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    return input_tokens, output_tokens


def calc_cost_usd(input_tokens: int, output_tokens: int) -> Decimal:
    """
    Считает стоимость запроса в долларах.
    """
    input_cost = (Decimal(input_tokens) / Decimal("1000000")) * INPUT_COST_PER_1M
    output_cost = (Decimal(output_tokens) / Decimal("1000000")) * OUTPUT_COST_PER_1M
    return quantize_money(input_cost + output_cost)


# -------------------------------------------------------
# Вызов OpenAI
# -------------------------------------------------------
def analyze_article(client: OpenAI, article: dict[str, Any]) -> tuple[dict[str, Any], int, int, Decimal]:
    """
    Отправляет статью в OpenAI и получает:
    - анализ
    - input tokens
    - output tokens
    - стоимость запроса
    """
    source_name = normalize_text(article.get("source_name"), 200) or "news"
    title = normalize_text(article.get("title"), 1000)
    summary = normalize_text(article.get("summary"), 4000)
    content = normalize_text(article.get("content"), 12000)
    url = normalize_text(article.get("canonical_url") or article.get("original_url"), 1000)

    user_prompt = USER_TEMPLATE.format(
        source_name=source_name,
        title=title,
        summary=summary,
        content=content,
        url=url,
    )

    response = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = pick_text(response)
    if not raw:
        raise ValueError("Empty model response")

    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    data = json.loads(raw)

    input_tokens, output_tokens = extract_usage_tokens(response)
    cost_usd = calc_cost_usd(input_tokens, output_tokens)

    analysis = {
        "is_relevant": bool(data.get("is_relevant", False)),
        "category": normalize_text(data.get("category", "other"), 100),
        "importance_score": normalize_int(data.get("importance_score", 1)),
        "reason": normalize_text(data.get("reason", ""), 1000),
        "russian_summary": normalize_text(data.get("russian_summary", ""), 4000),
        "telegram_title": normalize_text(data.get("telegram_title", ""), 300),
        "telegram_text": normalize_text(data.get("telegram_text", ""), 4000),
    }

    return analysis, input_tokens, output_tokens, cost_usd


# -------------------------------------------------------
# Добавление статьи в очередь редактора
# -------------------------------------------------------
def add_to_editor_queue(sb, article_id: int) -> bool:
    """
    Добавляет статью в editor_queue, если её там ещё нет.
    """
    existing = (
        sb.table("editor_queue")
        .select("id")
        .eq("article_id", article_id)
        .limit(1)
        .execute()
    ).data

    if existing:
        print(f"Queue skip for article_id={article_id}: already in editor_queue")
        return False

    sb.table("editor_queue").insert({
        "article_id": article_id,
        "status": "pending",
    }).execute()

    print(f"Added article_id={article_id} to editor_queue")
    return True


# -------------------------------------------------------
# Быстрая запись нерелевантной статьи без OpenAI
# -------------------------------------------------------
def save_prefilter_rejection(sb, article_id: int, reason: str) -> None:
    """
    Сохраняет результат pre-filter как нерелевантную статью без вызова AI.
    """
    sb.table("article_analysis").insert({
        "article_id": article_id,
        "is_relevant": False,
        "category": "other",
        "importance_score": 1,
        "reason": f"Pre-filter: {reason}",
        "russian_summary": "",
        "telegram_title": "",
        "telegram_text": "",
    }).execute()

    print(f"Saved pre-filter rejection for article_id={article_id}: {reason}")


# -------------------------------------------------------
# Работа с ai_runs / ai_balance
# -------------------------------------------------------
def create_ai_run(sb) -> int:
    """
    Создаёт запись нового прогона и возвращает run_id.
    """
    result = sb.table("ai_runs").insert({
        "run_type": "analyze",
        "started_at": now_iso(),
    }).execute()

    row = result.data[0]
    return row["id"]


def finish_ai_run(
    sb,
    run_id: int,
    processed: int,
    skipped: int,
    queued: int,
    prefilter_rejected: int,
    ai_calls: int,
    input_tokens: int,
    output_tokens: int,
    cost_usd: Decimal,
) -> tuple[Decimal, Decimal, Decimal]:
    """
    Завершает прогон, пишет статистику и обновляет ai_balance.

    Возвращает:
    - starting_balance_usd
    - spent_total_usd
    - remaining_estimated_usd
    """
    sb.table("ai_runs").update({
        "finished_at": now_iso(),
        "processed_count": processed,
        "skipped_count": skipped,
        "queued_count": queued,
        "prefilter_rejected_count": prefilter_rejected,
        "ai_calls_count": ai_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": str(cost_usd),
    }).eq("id", run_id).execute()

    balance_rows = (
        sb.table("ai_balance")
        .select("*")
        .eq("id", 1)
        .limit(1)
        .execute()
    ).data or []

    if not balance_rows:
        raise RuntimeError("ai_balance row with id=1 not found")

    balance = balance_rows[0]
    starting_balance = Decimal(str(balance["starting_balance_usd"]))
    spent_total_old = Decimal(str(balance["spent_total_usd"]))
    spent_total_new = quantize_money(spent_total_old + cost_usd)
    remaining = quantize_money(starting_balance - spent_total_new)

    sb.table("ai_balance").update({
        "spent_total_usd": str(spent_total_new),
        "updated_at": now_iso(),
    }).eq("id", 1).execute()

    return starting_balance, spent_total_new, remaining


def send_run_balance_message(
    bot_token: str,
    chat_id: str,
    run_cost_usd: Decimal,
    spent_total_usd: Decimal,
    remaining_estimated_usd: Decimal,
    ai_calls: int,
    prefilter_rejected: int,
) -> None:
    """
    Отправляет Telegram-сообщение после прогона analyzer.
    """
    text = (
        "📊 OpenAI balance update\n\n"
        f"Last run cost: ${run_cost_usd}\n"
        f"Spent total: ${spent_total_usd}\n"
        f"Estimated remaining: ${remaining_estimated_usd}\n\n"
        f"AI calls this run: {ai_calls}\n"
        f"Pre-filter rejected: {prefilter_rejected}"
    )

    telegram_send_message(bot_token, chat_id, text)


# -------------------------------------------------------
# Основная функция
# -------------------------------------------------------
def main():
    """
    Главная логика:
    - берёт статьи
    - не анализирует уже обработанные повторно
    - применяет pre-filter
    - вызывает OpenAI только для кандидатов
    - добавляет релевантные статьи в editor_queue
    - считает стоимость OpenAI и отправляет баланс в Telegram
    """
    print("Starting analyzer...")

    sb = get_supabase()
    oa = get_openai()

    run_id = create_ai_run(sb)

    result = (
        sb.table("articles")
        .select("*")
        .order("id", desc=True)
        .limit(20)
        .execute()
    )

    rows = result.data or []
    print(f"Loaded articles: {len(rows)}")

    processed = 0
    skipped = 0
    queued = 0
    prefilter_rejected = 0
    ai_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost_usd = Decimal("0")
    errors = 0
    quota_error = False

    for row in rows:
        article_id = row["id"]
        print(f"Processing article_id={article_id}")

        existing = (
            sb.table("article_analysis")
            .select("id,is_relevant,importance_score")
            .eq("article_id", article_id)
            .limit(1)
            .execute()
        ).data

        if existing:
            print(f"Skip article_id={article_id}: already analyzed")
            skipped += 1

            existing_row = existing[0]
            if existing_row.get("is_relevant") and (existing_row.get("importance_score") or 0) >= 6:
                if add_to_editor_queue(sb, article_id):
                    queued += 1

            continue

        article = {
            "source_name": "news",
            "title": row.get("title"),
            "summary": row.get("summary"),
            "content": row.get("content"),
            "canonical_url": row.get("canonical_url"),
            "original_url": row.get("original_url"),
        }

        try:
            send_to_ai, prefilter_reason = should_send_to_ai(article)

            if not send_to_ai:
                save_prefilter_rejection(sb, article_id, prefilter_reason)
                prefilter_rejected += 1
                processed += 1
                continue

            print(f"Sending article_id={article_id} to AI: {prefilter_reason}")

            analysis, input_tokens, output_tokens, cost_usd = analyze_article(oa, article)

            ai_calls += 1
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_cost_usd = quantize_money(total_cost_usd + cost_usd)

            sb.table("article_analysis").insert({
                "article_id": article_id,
                "is_relevant": analysis["is_relevant"],
                "category": analysis["category"],
                "importance_score": analysis["importance_score"],
                "reason": analysis["reason"],
                "russian_summary": analysis["russian_summary"],
                "telegram_title": analysis["telegram_title"],
                "telegram_text": analysis["telegram_text"],
            }).execute()

            processed += 1

            print(
                f"Saved analysis for article_id={article_id}: "
                f"relevant={analysis['is_relevant']}, "
                f"importance={analysis['importance_score']}, "
                f"category={analysis['category']}"
            )

            if analysis["is_relevant"] and analysis["importance_score"] >= 6:
                if add_to_editor_queue(sb, article_id):
                    queued += 1

        except Exception as e:
            error_text = repr(e)
            print(f"ERROR article_id={article_id}: {error_text}")
            errors += 1

            if "insufficient_quota" in error_text or "RateLimitError" in error_text:
                print("Stopping because OpenAI API quota/billing is not available.")
                quota_error = True
                break

    starting_balance, spent_total_usd, remaining_estimated_usd = finish_ai_run(
        sb=sb,
        run_id=run_id,
        processed=processed,
        skipped=skipped,
        queued=queued,
        prefilter_rejected=prefilter_rejected,
        ai_calls=ai_calls,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        cost_usd=total_cost_usd,
    )

    print(
        f"Done. processed={processed} skipped={skipped} "
        f"queued={queued} prefilter_rejected={prefilter_rejected} "
        f"ai_calls={ai_calls} cost_usd={total_cost_usd} errors={errors}"
    )

    # Telegram-уведомление о балансе после каждого прогона
    bot_token = get_env("TELEGRAM_BOT_TOKEN", required=False)
    chat_id = get_env("TELEGRAM_CHAT_ID", required=False)

    if bot_token and chat_id:
        try:
            send_run_balance_message(
                bot_token=bot_token,
                chat_id=chat_id,
                run_cost_usd=total_cost_usd,
                spent_total_usd=spent_total_usd,
                remaining_estimated_usd=remaining_estimated_usd,
                ai_calls=ai_calls,
                prefilter_rejected=prefilter_rejected,
            )
        except Exception as e:
            print(f"WARNING: failed to send Telegram balance message: {repr(e)}")

    if quota_error:
        print("Analyzer stopped due to missing API quota. Add billing in platform.openai.com.")
        return


# -------------------------------------------------------
# Точка входа
# -------------------------------------------------------
if __name__ == "__main__":
    main()
