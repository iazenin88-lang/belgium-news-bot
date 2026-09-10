"""Process one editor correction and return the revised candidate to Telegram.

The normal analyzer remains the recovery path for pending corrections. This
small runner is invoked by a workflow_dispatch request created immediately
after the editor submits a text-correction comment.
"""

from __future__ import annotations

import argparse
from decimal import Decimal


def parse_feedback_id(value: str) -> int:
    """Validate the workflow input before it reaches the database."""
    try:
        feedback_id = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("feedback_id must be a positive integer") from error
    if feedback_id <= 0:
        raise ValueError("feedback_id must be a positive integer")
    return feedback_id


def run_correction(feedback_id: int) -> bool:
    """Correct one pending request, account for it, and notify the editor."""
    # Imports are deferred so the small input validator remains testable
    # without installing the production API clients.
    from analyzer import (
        create_ai_run,
        finish_ai_run,
        get_openai,
        get_supabase,
        process_pending_corrections,
    )
    from notifier import get_env, notify_queue_item

    sb = get_supabase()
    client = get_openai()
    run_id = create_ai_run(sb, run_type="correction")
    stats = {
        "processed": 0,
        "failed": 0,
        "ai_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": Decimal("0"),
        "quota_error": False,
        "processed_items": [],
    }

    try:
        stats = process_pending_corrections(
            sb,
            client,
            max_items=1,
            feedback_id=feedback_id,
        )
    finally:
        finish_ai_run(
            sb,
            run_id,
            processed=0,
            skipped=0,
            queued=0,
            prefilter_rejected=0,
            ai_calls=int(stats["ai_calls"]),
            input_tokens=int(stats["input_tokens"]),
            output_tokens=int(stats["output_tokens"]),
            cost_usd=Decimal(stats["cost_usd"]),
            corrections_processed=int(stats["processed"]),
            corrections_failed=int(stats["failed"]),
        )

    if stats["failed"] or stats["quota_error"]:
        raise RuntimeError(f"Correction failed for feedback_id={feedback_id}")
    if not stats["processed_items"]:
        # A retried workflow can legitimately find an already-applied request.
        print(f"No pending correction remains for feedback_id={feedback_id}")
        return False

    item = stats["processed_items"][0]
    queue_id = int(item["queue_id"])
    sent = notify_queue_item(
        sb,
        get_env("TELEGRAM_BOT_TOKEN"),
        get_env("TELEGRAM_CHAT_ID"),
        queue_id,
    )
    if not sent:
        queue_rows = (
            sb.table("editor_queue")
            .select("status,revision")
            .eq("id", queue_id)
            .limit(1)
            .execute()
        ).data or []
        if not queue_rows or queue_rows[0].get("status") != "sent":
            raise RuntimeError(f"Corrected queue item {queue_id} was not sent")
        print(f"Corrected queue item {queue_id} was already sent")
        return False

    print(
        f"Correction delivered to Telegram: feedback_id={feedback_id} "
        f"queue_id={queue_id} revision={item['revision']}"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback-id", required=True)
    args = parser.parse_args()
    run_correction(parse_feedback_id(args.feedback_id))


if __name__ == "__main__":
    main()
