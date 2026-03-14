"""
analyzer.py

Назначение:
анализирует новые статьи из таблицы articles и сохраняет результат в article_analysis.

Логика работы:
1. Берём статьи из articles
2. Если статья уже анализировалась:
   - пропускаем повторный AI-анализ
   - при необходимости добавляем её в editor_queue
3. Если статья новая:
   - сначала запускаем дешёвый pre-filter без OpenAI
   - если статья явно нерелевантна, сразу записываем это в article_analysis
   - если статья потенциально интересна, отправляем её в OpenAI
4. Если AI считает статью релевантной и importance_score >= 6,
   добавляем её в editor_queue

Зачем нужен pre-filter:
- уменьшает число вызовов OpenAI
- снижает стоимость
- отбрасывает очевидный шум ещё до AI
"""

import json
import os
from typing import Any

from openai import OpenAI
from supabase import create_client


# -------------------------------------------------------
# Модель OpenAI
# -------------------------------------------------------
MODEL = "gpt-5-mini"


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

# Явный шум, который почти никогда не нужен
# ВАЖНО: entertainment здесь НЕ удаляется специально, по твоему запросу
HARD_REJECT_KEYWORDS = {
    "football", "soccer", "champions league", "premier league", "uefa",
    "basketball", "tennis tournament", "formula 1", "motogp",
    "movie star", "red carpet", "fashion week",
    "album release", "box office", "love island",
    "royal gossip", "tv show", "reality show",
    "transfer rumor", "match preview", "match report", "line-up",
    "livestream sports", "sports betting",
}

# Общие ключевые слова релевантности
PASS_KEYWORDS = {
    # Бельгия / регионы / города
    "belgium", "belgian", "brussels", "flanders", "wallonia", "antwerp", "ghent",
    "belgië", "brussel", "vlaanderen", "wallonië", "gent", "antwerpen",

    # миграция / статус / документы
    "visa", "residence permit", "permit", "asylum", "refugee", "refugees",
    "migrant", "migrants", "immigration", "integration", "expat", "foreign worker",
    "temporary protection", "residency", "residence card",

    # жильё / работа / деньги
    "housing", "rent", "rental", "landlord", "tenant", "mortgage",
    "salary", "wage", "employment", "job market", "unemployment",
    "tax", "taxes", "benefit", "benefits", "pension", "allowance",

    # быт
    "school", "education", "transport", "rail", "train", "tram", "bus",
    "healthcare", "hospital", "doctor", "medicine", "insurance",
    "safety", "police", "court", "law", "legal",

    # Украина / беженцы
    "ukrainian", "ukrainians", "ukraine",

    # Россия / русскоязычная аудитория
    "russia", "russian", "russians",
    "россия", "россияне", "русские", "русский",

    # entertainment оставляем как допустимую тему
    "entertainment", "festival", "concert", "cinema", "music", "cultural event",
    "culture", "event", "events",
}

# Сильные practical / общественные сигналы
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
def get_env(name: str) -> str:
    """
    Получает переменную окружения.
    Если переменная не задана — бросаем ошибку.
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


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


# -------------------------------------------------------
# Более строгий pre-filter перед OpenAI
# -------------------------------------------------------
def should_send_to_ai(article: dict[str, Any]) -> tuple[bool, str]:
    """
    Более строгий pre-filter перед OpenAI.

    Логика:
    1. Сразу режем явный мусор
    2. Требуем либо практическую тему, либо сочетание нескольких сильных сигналов
    3. Пограничные случаи без достаточного контекста режем
    """
    title = normalize_text(article.get("title"), 1000)
    summary = normalize_text(article.get("summary"), 4000)
    content = normalize_text(article.get("content"), 8000)

    combined = f"{title}\n{summary}\n{content}".strip()
    combined_lower = combined.lower()

    # Нет текста вообще — сразу reject
    if not title and not summary and not content:
        return False, "Нет заголовка, summary и content"

    # Явный шум — сразу reject
    if contains_any(combined_lower, HARD_REJECT_KEYWORDS):
        return False, "Явно нерелевантная тема (спорт и т.п.)"

    pass_matches = count_matches(combined_lower, PASS_KEYWORDS)
    high_signal_matches = count_matches(combined_lower, HIGH_SIGNAL_KEYWORDS)

    summary_len = len(summary)
    content_len = len(content)

    # -----------------------------
    # 1. Сильные practical темы
    # -----------------------------
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

    # -----------------------------
    # 2. Темы про Россию / русских / россиян
    # -----------------------------
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

    # -----------------------------
    # 3. Бельгийский контекст
    # -----------------------------
    belgium_keywords = {
        "belgium", "belgian", "brussels", "flanders", "wallonia", "antwerp", "ghent",
        "belgië", "brussel", "vlaanderen", "wallonië", "gent", "antwerpen",
    }
    belgium_matches = count_matches(combined_lower, belgium_keywords)

    if belgium_matches >= 1 and (practical_matches >= 1 or high_signal_matches >= 2):
        return True, "Бельгийский контекст + практическая или сильная тема"

    # -----------------------------
    # 4. Европа / соседи
    # -----------------------------
    europe_keywords = {
        "eu", "european union", "europe",
        "netherlands", "france", "germany", "luxembourg",
    }
    europe_matches = count_matches(combined_lower, europe_keywords)

    if europe_matches >= 1 and practical_matches >= 1 and (summary_len >= 80 or content_len >= 250):
        return True, "Европейский контекст с practical-углом"

    # -----------------------------
    # 5. Entertainment / culture
    # Оставляем проход, но только при наличии контекста
    # -----------------------------
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

    # -----------------------------
    # 6. Общий сильный случай
    # -----------------------------
    if high_signal_matches >= 3 and (summary_len >= 100 or content_len >= 300):
        return True, "Несколько сильных сигналов и содержательный текст"

    # -----------------------------
    # 7. Слишком короткие и слабые статьи режем
    # -----------------------------
    if summary_len < 60 and content_len < 180:
        return False, "Слишком мало содержательной информации"

    # -----------------------------
    # 8. Один слабый сигнал — недостаточно
    # -----------------------------
    if pass_matches <= 1 and high_signal_matches <= 1 and practical_matches == 0:
        return False, "Недостаточно сигналов релевантности"

    # -----------------------------
    # 9. Пограничные статьи режем жёстче
    # -----------------------------
    return False, "Пограничный случай без достаточных оснований для AI"


# -------------------------------------------------------
# Вызов OpenAI
# -------------------------------------------------------
def analyze_article(client: OpenAI, article: dict[str, Any]) -> dict[str, Any]:
    """
    Отправляет статью в OpenAI и получает структурированный анализ.
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

    return {
        "is_relevant": bool(data.get("is_relevant", False)),
        "category": normalize_text(data.get("category", "other"), 100),
        "importance_score": normalize_int(data.get("importance_score", 1)),
        "reason": normalize_text(data.get("reason", ""), 1000),
        "russian_summary": normalize_text(data.get("russian_summary", ""), 4000),
        "telegram_title": normalize_text(data.get("telegram_title", ""), 300),
        "telegram_text": normalize_text(data.get("telegram_text", ""), 4000),
    }


# -------------------------------------------------------
# Добавление статьи в очередь редактора
# -------------------------------------------------------
def add_to_editor_queue(sb, article_id: int) -> bool:
    """
    Добавляет статью в editor_queue, если её там ещё нет.

    Возвращает True, если запись была добавлена.
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
    """
    print("Starting analyzer...")

    sb = get_supabase()
    oa = get_openai()

    # Берём самые новые статьи за запуск
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

            analysis = analyze_article(oa, article)

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

    print(
        f"Done. processed={processed} skipped={skipped} "
        f"queued={queued} prefilter_rejected={prefilter_rejected} errors={errors}"
    )

    if quota_error:
        print("Analyzer stopped due to missing API quota. Add billing in platform.openai.com.")
        return


# -------------------------------------------------------
# Точка входа
# -------------------------------------------------------
if __name__ == "__main__":
    main()
