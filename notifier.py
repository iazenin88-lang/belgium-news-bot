"""
notifier.py

Назначение:
отправляет кандидатов на публикацию из editor_queue в Telegram редактору канала.

Что делает скрипт:
1. Подключается к Supabase
2. Берёт записи из editor_queue со статусом "pending"
3. Для каждой записи:
   - получает анализ из article_analysis
   - получает статью из articles
   - формирует сообщение
   - добавляет версионированные кнопки публикации / отклонения
   - отправляет сообщение в Telegram
4. После успешной отправки меняет статус записи на "sent"

Причину отклонения и комментарий обрабатывает Supabase Edge Function
telegram-webhook. Исправленный текст возвращается сюда со следующей revision.
"""

import html
import os
from datetime import datetime, timezone
from typing import Any

import requests
from supabase import create_client


# Базовый URL Telegram Bot API
TELEGRAM_API_BASE = "https://api.telegram.org"


# -------------------------------------------------------
# Получение переменной окружения
# -------------------------------------------------------
def get_env(name: str) -> str:
    """
    Возвращает значение переменной окружения.

    Если переменная не задана, завершаем работу с понятной ошибкой.
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


# -------------------------------------------------------
# Создание клиента Supabase
# -------------------------------------------------------
def get_supabase():
    """
    Создаёт клиент Supabase.
    Используются:
    - SUPABASE_URL
    - SUPABASE_SERVICE_KEY
    """
    return create_client(
        get_env("SUPABASE_URL"),
        get_env("SUPABASE_SERVICE_KEY"),
    )


# -------------------------------------------------------
# Отправка сообщения в Telegram
# -------------------------------------------------------
def telegram_send_message(
    bot_token: str,
    chat_id: str,
    text: str,
    reply_markup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Отправляет сообщение в Telegram.

    reply_markup используется для inline-кнопок.
    """
    url = f"{TELEGRAM_API_BASE}/bot{bot_token}/sendMessage"

    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    # Если переданы кнопки — добавляем их
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup

    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


# -------------------------------------------------------
# Формирование текста сообщения
# -------------------------------------------------------
def build_message(
    article_id: int,
    analysis: dict[str, Any],
    article: dict[str, Any],
    revision: int = 1,
) -> str:
    """
    Собирает текст кандидата в публикацию для Telegram.
    """
    title = html.escape(
        (analysis.get("telegram_title") or article.get("title") or "Без заголовка").strip()
    )

    text = html.escape(
        (analysis.get("telegram_text") or analysis.get("russian_summary") or "").strip()
    )

    category = html.escape(str(analysis.get("category") or "other"))
    importance = analysis.get("importance_score") or 0
    url = (article.get("canonical_url") or article.get("original_url") or "").strip()

    candidate_label = "📰 <b>Кандидат в публикацию</b>"
    if revision > 1:
        candidate_label = f"♻️ <b>Исправленная версия {revision}</b>"

    parts = [
        candidate_label,
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


# -------------------------------------------------------
# Формирование inline-кнопок
# -------------------------------------------------------
def build_reply_markup(queue_id: int, revision: int = 1) -> dict[str, Any]:
    """
    Создаёт inline-клавиатуру с кнопками публикации и отклонения.

    В callback_data передаём queue_id и revision, чтобы webhook не позволил
    опубликовать устаревший текст после AI-исправления.
    """
    return {
        "inline_keyboard": [
            [
                {
                    "text": "✅ Опубликовать",
                    "callback_data": f"publish:{queue_id}:{revision}",
                },
                {
                    "text": "❌ Не публиковать",
                    "callback_data": f"reject:{queue_id}:{revision}",
                },
            ]
        ]
    }


# -------------------------------------------------------
# Основная функция
# -------------------------------------------------------
def main():
    """
    Главная логика:
    - загружает pending записи из editor_queue
    - отправляет их в Telegram
    - меняет статус на sent
    """
    print("Starting notifier...")

    sb = get_supabase()
    bot_token = get_env("TELEGRAM_BOT_TOKEN")
    chat_id = get_env("TELEGRAM_CHAT_ID")

    # Берём максимум 5 pending новостей за один запуск
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
        queue_id = queue_row["id"]
        article_id = queue_row["article_id"]
        revision = int(queue_row.get("revision") or 1)

        print(f"Processing queue_id={queue_id}, article_id={article_id}")

        # Получаем анализ статьи
        analysis_rows = (
            sb.table("article_analysis")
            .select("*")
            .eq("article_id", article_id)
            .limit(1)
            .execute()
        ).data or []

        # Получаем саму статью
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
            message = build_message(article_id, analysis, article, revision)
            reply_markup = build_reply_markup(queue_id, revision)

            telegram_result = telegram_send_message(
                bot_token=bot_token,
                chat_id=chat_id,
                text=message,
                reply_markup=reply_markup,
            )

            sent_message = telegram_result.get("result") or {}
            sent_chat = sent_message.get("chat") or {}
            telegram_message_id = sent_message.get("message_id")
            telegram_chat_id = sent_chat.get("id", chat_id)

            if not telegram_message_id:
                raise RuntimeError("Telegram response has no message_id")

            # После отправки помечаем, что редактору уже показали
            sb.table("editor_queue").update({
                "status": "sent",
                "telegram_chat_id": telegram_chat_id,
                "telegram_message_id": telegram_message_id,
                "last_sent_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", queue_id).eq("status", "pending").eq(
                "revision", revision
            ).execute()

            print(f"Sent queue_id={queue_id}")
            sent += 1

        except Exception as e:
            print(f"ERROR queue_id={queue_id}: {repr(e)}")
            errors += 1

    print(f"Done. sent={sent} skipped={skipped} errors={errors}")


# -------------------------------------------------------
# Точка входа
# -------------------------------------------------------
if __name__ == "__main__":
    main()
