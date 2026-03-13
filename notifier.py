"""
notifier.py

Этот скрипт отправляет кандидатов на публикацию из таблицы editor_queue
в Telegram редактору канала.

Логика работы:

1. Подключаемся к Supabase
2. Берём статьи со статусом "pending"
3. Для каждой статьи:
      - получаем анализ (article_analysis)
      - получаем оригинальную статью (articles)
      - формируем сообщение
      - отправляем в Telegram
4. После отправки меняем статус очереди на "sent"
"""

import html
import os
from typing import Any

import requests
from supabase import create_client


# Базовый URL Telegram Bot API
TELEGRAM_API_BASE = "https://api.telegram.org"


# -------------------------------------------------------
# Получение переменных окружения
# -------------------------------------------------------
def get_env(name: str) -> str:
    """
    Получает переменную окружения.

    Если переменная отсутствует — завершаем программу,
    потому что без неё скрипт работать не сможет.
    """
    value = os.environ.get(name)

    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")

    return value


# -------------------------------------------------------
# Подключение к Supabase
# -------------------------------------------------------
def get_supabase():
    """
    Создаёт клиент Supabase.

    Используются переменные:
    SUPABASE_URL
    SUPABASE_SERVICE_KEY
    """

    return create_client(
        get_env("SUPABASE_URL"),
        get_env("SUPABASE_SERVICE_KEY"),
    )


# -------------------------------------------------------
# Отправка сообщения в Telegram
# -------------------------------------------------------
def telegram_send_message(bot_token: str, chat_id: str, text: str) -> dict[str, Any]:
    """
    Отправляет сообщение через Telegram Bot API.

    Используется метод:
    sendMessage
    """

    url = f"{TELEGRAM_API_BASE}/bot{bot_token}/sendMessage"

    response = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",  # поддержка форматирования
            "disable_web_page_preview": True,  # отключаем превью ссылок
        },
        timeout=30,
    )

    # если Telegram вернул ошибку — вызываем exception
    response.raise_for_status()

    return response.json()


# -------------------------------------------------------
# Формирование текста сообщения
# -------------------------------------------------------
def build_message(article_id: int, analysis: dict[str, Any], article: dict[str, Any]) -> str:
    """
    Собирает красивое сообщение для Telegram.

    Используются данные:
    - telegram_title
    - telegram_text
    - category
    - importance_score
    """

    # Заголовок
    title = html.escape(
        (analysis.get("telegram_title") or article.get("title") or "Без заголовка").strip()
    )

    # Основной текст
    text = html.escape(
        (analysis.get("telegram_text") or analysis.get("russian_summary") or "").strip()
    )

    # Категория
    category = html.escape(str(analysis.get("category") or "other"))

    # Важность
    importance = analysis.get("importance_score") or 0

    # URL статьи
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

    # если есть ссылка — добавляем
    if url:
        safe_url = html.escape(url)
        parts.append(f'Источник: <a href="{safe_url}">ссылка</a>')

    return "\n".join(parts)


# -------------------------------------------------------
# Основная логика скрипта
# -------------------------------------------------------
def main():
    """
    Основная функция программы.

    Выполняет:
    - загрузку очереди
    - отправку сообщений
    - обновление статуса
    """

    print("Starting notifier...")

    # подключаемся к базе
    sb = get_supabase()

    # получаем Telegram настройки
    bot_token = get_env("TELEGRAM_BOT_TOKEN")
    chat_id = get_env("TELEGRAM_CHAT_ID")

    # загружаем статьи из очереди редактора
    queue_rows = (
        sb.table("editor_queue")
        .select("*")
        .eq("status", "pending")
        .order("id", desc=False)
        .limit(5)  # отправляем максимум 5 за запуск
        .execute()
    ).data or []

    print(f"Loaded pending queue items: {len(queue_rows)}")

    sent = 0
    skipped = 0
    errors = 0

    # обрабатываем каждую запись
    for queue_row in queue_rows:

        article_id = queue_row["article_id"]
        queue_id = queue_row["id"]

        print(f"Processing queue_id={queue_id},
