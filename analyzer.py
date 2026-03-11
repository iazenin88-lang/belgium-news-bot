import json
import os
from typing import Any

from openai import OpenAI
from supabase import create_client


MODEL = "gpt-5-mini"

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

Считать нерелевантными:
- обычные мировые новости без практической связи с жизнью в Бельгии
- спорт, шоу-бизнес, криминальные мелочи, если нет практической пользы
- локальные мелочи без заметного влияния на читателей

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


def get_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def get_supabase():
    return create_client(
        get_env("SUPABASE_URL"),
        get_env("SUPABASE_SERVICE_KEY"),
    )


def get_openai():
    return OpenAI(api_key=get_env("OPENAI_API_KEY"))


def pick_text(response: Any) -> str:
    # Responses API often exposes output_text directly
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
    if value is None:
        return ""
    text = str(value).strip()
    return text[:max_len]


def normalize_int(value: Any, default: int = 1) -> int:
    try:
        n = int(value)
    except Exception:
        return default
    return max(1, min(10, n))


def analyze_article(client: OpenAI, article: dict[str, Any]) -> dict[str, Any]:
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

    # If model wrapped JSON in ```json ... ```
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


def main():
    print("Starting analyzer...")

    sb = get_supabase()
    oa = get_openai()

    # Берём статьи
    result = (
        sb.table("articles")
        .select("*")
        .order("id", desc=False)
        .limit(20)
        .execute()
    )

    rows = result.data or []
    print(f"Loaded articles: {len(rows)}")

    processed = 0
    skipped = 0
    errors = 0

    for row in rows:
        article_id = row["id"]
        print(f"Processing article_id={article_id}")

        existing = (
            sb.table("article_analysis")
            .select("id")
            .eq("article_id", article_id)
            .limit(1)
            .execute()
        ).data

        if existing:
            print(f"Skip article_id={article_id}: already analyzed")
            skipped += 1
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
            analysis = analyze_article(oa, article)
            print(f"AI result for article_id={article_id}: relevant={analysis['is_relevant']}, category={analysis['category']}")

            insert_row = {
                "article_id": article_id,
                "is_relevant": analysis["is_relevant"],
                "category": analysis["category"],
                "importance_score": analysis["importance_score"],
                "reason": analysis["reason"],
                "russian_summary": analysis["russian_summary"],
                "telegram_title": analysis["telegram_title"],
                "telegram_text": analysis["telegram_text"],
            }

            sb.table("article_analysis").insert(insert_row).execute()
            processed += 1

        except Exception as e:
            print(f"ERROR article_id={article_id}: {repr(e)}")
            errors += 1

    print(f"Done. processed={processed} skipped={skipped} errors={errors}")

    # Не падаем всей job, чтобы увидеть логи
    if processed == 0 and errors > 0:
        raise RuntimeError("Analyzer finished with errors and processed 0 articles")


if __name__ == "__main__":
    main()
