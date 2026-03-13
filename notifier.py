import html
import os
from typing import Any

import requests
from supabase import create_client


TELEGRAM_API_BASE = "https://api.telegram.org"


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


def telegram_send_message(bot_token: str, chat_id: str, text: str) -> dict[str, Any]:
    url = f"{TELEGRAM_API_BASE}/bot{bot_token}/sendMessage"

    response = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def build_message(article_id: int, analysis: dict[str, Any], article: dict[str, Any]) -> str:
    title = html.escape((analysis.get("telegram_title") or article.get("title") or "Без заголовка").strip())
    text = html.escape((analysis.get("telegram_text") or analysis.get("russian_summary") or "").strip())
    category = html.escape(str(analysis.get("category") or "other"))
    importance = analysis.get("importance_score") or 0
    url = (article.get("canonical_url") or article.get("original_url") or "").strip()

    parts = [
        "📰 <b>Кандидат в публикацию</b>",
        f"<b>{title}</b>",
        "",
        text,
        "",
        f"Категория: <b>{category}</b>",
        f"Важность: <b>{importance}</b>/10",
        f"Article ID: <code>{article_id}</code>",
    ]

    if url:
        safe_url = html.escape(url)
        parts.append(f'Источник: <a href="{safe_url}">ссылка</a>')

    return "\n".join(parts)


def main():
    print("Starting notifier...")

    sb = get_supabase()
    bot_token = get_env("TELEGRAM_BOT_TOKEN")
    chat_id = get_env("TELEGRAM_CHAT_ID")

    queue_rows = (
        sb.table("editor_queue")
        .select("*")
        .eq("status", "pending")
        .order("id", desc=False)
        .limit(5)
        .execute()
    ).data or []

    print(f"Loaded pending queue items: {len(queue_rows)}")

    sent = 0
    skipped = 0
    errors = 0

    for queue_row in queue_rows:
        article_id = queue_row["article_id"]
        queue_id = queue_row["id"]
        print(f"Processing queue_id={queue_id}, article_id={article_id}")

        analysis_rows = (
            sb.table("article_analysis")
            .select("*")
            .eq("article_id", article_id)
            .limit(1)
            .execute()
        ).data or []

        article_rows = (
            sb.table("articles")
            .select("*")
            .eq("id", article_id)
            .limit(1)
            .execute()
        ).data or []

        if not analysis_rows or not article_rows:
            print(f"Skip queue_id={queue_id}: missing article or analysis")
            skipped += 1
            continue

        analysis = analysis_rows[0]
        article = article_rows[0]

        try:
            message = build_message(article_id, analysis, article)
            telegram_send_message(bot_token, chat_id, message)

            sb.table("editor_queue").update({
                "status": "sent"
            }).eq("id", queue_id).execute()

            print(f"Sent queue_id={queue_id}")
            sent += 1

        except Exception as e:
            print(f"ERROR queue_id={queue_id}: {repr(e)}")
            errors += 1

    print(f"Done. sent={sent} skipped={skipped} errors={errors}")


if __name__ == "__main__":
    main()
