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
  "telegram_text": "Готовый короткий текст для Telegram на русском без markdown."
}}
"""


def get_supabase():
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"],
    )


def get_openai():
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def pick_text(response: Any) -> str:
    # Универсально пытаемся достать текст из Responses API ответа
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    # запасной путь
    try:
        parts = []
        for item in response.output:
            if getattr(item, "type", None) != "message":
                continue
            for c in getattr(item, "content", []):
                if getattr(c, "type", None) == "output_text":
                    parts.append(c.text)
        return "\n".join(parts).strip()
    except Exception:
        return ""


def analyze_article(client: OpenAI, article: dict[str, Any]) -> dict[str, Any]:
    source_name = article.get("source_name") or "Unknown"
    title = article.get("title") or ""
    summary = article.get("summary") or ""
    content = article.get("content") or ""
    url = article.get("canonical_url") or article.get("original_url") or ""

    # не шлём слишком много текста
    if len(content) > 12000:
        content = content[:12000]

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
        text={"format": {"type": "json_object"}},
    )

    raw = pick_text(response)
    if not raw:
        raise ValueError("Empty model response")

    data = json.loads(raw)

    # базовая нормализация
    return {
        "is_relevant": bool(data.get("is_relevant", False)),
        "category": str(data.get("category", "other")),
        "importance_score": int(data.get("importance_score", 1)),
        "reason": str(data.get("reason", ""))[:1000],
        "russian_summary": str(data.get("russian_summary", ""))[:4000],
        "telegram_title": str(data.get("telegram_title", ""))[:300],
        "telegram_text": str(data.get("telegram_text", ""))[:4000],
    }


def main():
    sb = get_supabase()
    oa = get_openai()

    # Берём статьи, для которых ещё нет анализа
    result = (
        sb.table("articles")
        .select("id,title,summary,content,canonical_url,original_url,source_id,sources(name)")
        .limit(20)
        .order("id", desc=False)
        .execute()
    )

    rows = result.data or []

    processed = 0
    skipped = 0
    errors = 0

    for row in rows:
        article_id = row["id"]

        existing = (
            sb.table("article_analysis")
            .select("id")
            .eq("article_id", article_id)
            .limit(1)
            .execute()
        ).data

        if existing:
            skipped += 1
            continue

        source_obj = row.get("sources")
        if isinstance(source_obj, dict):
            source_name = source_obj.get("name")
        else:
            source_name = None

        article = {
            "id": article_id,
            "title": row.get("title"),
            "summary": row.get("summary"),
            "content": row.get("content"),
            "canonical_url": row.get("canonical_url"),
            "original_url": row.get("original_url"),
            "source_name": source_name,
        }

        try:
            analysis = analyze_article(oa, article)

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
            print(f"ERROR article_id={article_id}: {e}")
            errors += 1

    print(f"Done. processed={processed} skipped={skipped} errors={errors}")


if __name__ == "__main__":
    main()
