"""
balance_report.py

Утренний отчёт по расходам OpenAI.

Отправляет в Telegram:
- сколько было потрачено вчера
- сколько потрачено всего
- какой оценочный остаток на момент отправки
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import os
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


def telegram_send_message(bot_token: str, chat_id: str, text: str) -> None:
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


def main():
    sb = get_supabase()
    bot_token = get_env("TELEGRAM_BOT_TOKEN")
    chat_id = get_env("TELEGRAM_CHAT_ID")

    now = datetime.now(timezone.utc)
    today_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    yesterday_start = today_start - timedelta(days=1)

    runs = (
        sb.table("ai_runs")
        .select("*")
        .gte("started_at", yesterday_start.isoformat())
        .lt("started_at", today_start.isoformat())
        .execute()
    ).data or []

    spent_yesterday = Decimal("0")
    for run in runs:
        spent_yesterday += Decimal(str(run.get("cost_usd", 0)))

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
    spent_total = Decimal(str(balance["spent_total_usd"]))
    remaining = starting_balance - spent_total

    text = (
        "📅 OpenAI daily report\n\n"
        f"Spent yesterday: ${spent_yesterday:.6f}\n"
        f"Spent total: ${spent_total:.6f}\n"
        f"Estimated remaining: ${remaining:.6f}"
    )

    telegram_send_message(bot_token, chat_id, text)


if __name__ == "__main__":
    main()
