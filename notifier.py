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
PRIORITY_CATEGORIES = {"housing", "taxes", "migration", "work"}
NOTIFICATION_BATCH_SIZE = 5
PRIORITY_LOOKBACK = 100


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
# Отправка одного кандидата
# -------------------------------------------------------
def notify_queue_item(
    sb,
    bot_token: str,
    chat_id: str,
    queue_id: int,
) -> bool:
    """Send one pending candidate and claim it before calling Telegram.

    The claim prevents the immediate correction workflow and a scheduled
    notifier from sending the same revision twice.
    """
    queue_rows = (
        sb.table("editor_queue")
        .select("*")
        .eq("id", queue_id)
        .eq("status", "pending")
        .limit(1)
        .execute()
    ).data or []
    if not queue_rows:
        print(f"Skip queue_id={queue_id}: no longer pending")
        return False

    queue_row = queue_rows[0]
    article_id = int(queue_row["article_id"])
    revision = int(queue_row.get("revision") or 1)

    # supabase-py returns an array from UPDATE ... RETURNING.  Do not use
    # maybe_single(): current supabase-py versions do not expose it on this
    # filtered update builder.
    claimed_rows = (
        sb.table("editor_queue")
        .update({"status": "notifying"})
        .eq("id", queue_id)
        .eq("status", "pending")
        .eq("revision", revision)
        .select("id")
        .execute()
    ).data or []
    if not claimed_rows:
        print(f"Skip queue_id={queue_id}: claimed by another notifier")
        return False

    telegram_sent = False
    try:
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
            raise RuntimeError("Queue item has no article or analysis")

        message = build_message(article_id, analysis_rows[0], article_rows[0], revision)
        telegram_result = telegram_send_message(
            bot_token=bot_token,
            chat_id=chat_id,
            text=message,
            reply_markup=build_reply_markup(queue_id, revision),
        )

        sent_message = telegram_result.get("result") or {}
        sent_chat = sent_message.get("chat") or {}
        telegram_message_id = sent_message.get("message_id")
        telegram_chat_id = sent_chat.get("id", chat_id)
        if not telegram_message_id:
            raise RuntimeError("Telegram response has no message_id")
        telegram_sent = True

        finalized_rows = (
            sb.table("editor_queue")
            .update({
                "status": "sent",
                "telegram_chat_id": telegram_chat_id,
                "telegram_message_id": telegram_message_id,
                "last_sent_at": datetime.now(timezone.utc).isoformat(),
            })
            .eq("id", queue_id)
            .eq("status", "notifying")
            .eq("revision", revision)
            .select("id")
            .execute()
        ).data or []
        if not finalized_rows:
            raise RuntimeError("Telegram message sent but queue status was not finalized")

        print(f"Sent queue_id={queue_id} revision={revision}")
        return True
    except Exception:
        if not telegram_sent:
            sb.table("editor_queue").update({
                "status": "pending",
            }).eq("id", queue_id).eq("status", "notifying").eq(
                "revision", revision
            ).execute()
        raise


def prioritize_pending_queue_rows(
    oldest_rows: list[dict[str, Any]],
    recent_rows: list[dict[str, Any]],
    analysis_rows: list[dict[str, Any]],
    limit: int = NOTIFICATION_BATCH_SIZE,
) -> list[int]:
    """Give practical Belgian news a prompt slot while draining the old queue."""
    if not oldest_rows or limit <= 0:
        return []

    scores = {int(row["article_id"]): row for row in analysis_rows}
    oldest_id = int(oldest_rows[0]["id"])
    selected = [oldest_id]
    if limit == 1:
        return selected

    priority_rows = [
        row for row in recent_rows
        if (analysis := scores.get(int(row["article_id"])))
        and analysis.get("category") in PRIORITY_CATEGORIES
        and int(analysis.get("importance_score") or 0) >= 7
    ]
    priority_rows.sort(
        key=lambda row: (
            -int(scores[int(row["article_id"])]["importance_score"]),
            int(row["id"]),
        )
    )

    for row in priority_rows + oldest_rows:
        queue_id = int(row["id"])
        if queue_id not in selected:
            selected.append(queue_id)
        if len(selected) >= limit:
            break
    return selected


def select_pending_queue_ids(sb) -> list[int]:
    oldest_rows = (
        sb.table("editor_queue")
        .select("id,article_id")
        .eq("status", "pending")
        .order("id", desc=False)
        .limit(NOTIFICATION_BATCH_SIZE)
        .execute()
    ).data or []
    if not oldest_rows:
        return []

    try:
        recent_rows = (
            sb.table("editor_queue")
            .select("id,article_id")
            .eq("status", "pending")
            .order("id", desc=True)
            .limit(PRIORITY_LOOKBACK)
            .execute()
        ).data or []
        article_ids = list({int(row["article_id"]) for row in recent_rows})
        analysis_rows = (
            sb.table("article_analysis")
            .select("article_id,category,importance_score")
            .in_("article_id", article_ids)
            .execute()
        ).data or []
        return prioritize_pending_queue_rows(oldest_rows, recent_rows, analysis_rows)
    except Exception as error:
        print(f"Priority lookup failed; sending oldest items: {error!r}")
        return [int(row["id"]) for row in oldest_rows]


# -------------------------------------------------------
# Основная функция
# -------------------------------------------------------
def main():
    """
    Отправляет максимум пять pending-кандидатов, сначала важные практические новости.
    Вызов notify_queue_item()
    используется отдельным workflow для немедленного показа исправленной версии.
    """
    print("Starting notifier...")

    sb = get_supabase()
    bot_token = get_env("TELEGRAM_BOT_TOKEN")
    chat_id = get_env("TELEGRAM_CHAT_ID")

    queue_ids = select_pending_queue_ids(sb)
    print(f"Selected pending queue items: {queue_ids}")

    sent = 0
    skipped = 0
    errors = 0
    for queue_id in queue_ids:
        try:
            if notify_queue_item(sb, bot_token, chat_id, queue_id):
                sent += 1
            else:
                skipped += 1
        except Exception as error:
            print(f"ERROR queue_id={queue_id}: {repr(error)}")
            errors += 1

    print(f"Done. sent={sent} skipped={skipped} errors={errors}")


# -------------------------------------------------------
# Точка входа
# -------------------------------------------------------
if __name__ == "__main__":
    main()
