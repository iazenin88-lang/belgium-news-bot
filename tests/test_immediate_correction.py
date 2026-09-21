from pathlib import Path
import unittest

from correction_pipeline import (
    load_pending_delivery_for_applied_feedback,
    parse_feedback_id,
)


ROOT = Path(__file__).resolve().parents[1]


class ImmediateCorrectionTests(unittest.TestCase):
    def test_feedback_id_validation(self):
        self.assertEqual(parse_feedback_id("42"), 42)
        with self.assertRaises(ValueError):
            parse_feedback_id("0")
        with self.assertRaises(ValueError):
            parse_feedback_id("not-a-number")

    def test_applied_correction_can_retry_pending_telegram_delivery(self):
        class Response:
            def __init__(self, data):
                self.data = data

        class Query:
            def __init__(self, table):
                self.table = table
                self.filters = {}

            def select(self, _fields):
                return self

            def eq(self, field, value):
                self.filters[field] = value
                return self

            def limit(self, _value):
                return self

            def execute(self):
                if self.table == "editorial_feedback":
                    row = {"id": 64, "queue_id": 959, "status": "applied"}
                else:
                    row = {"id": 959, "revision": 2, "status": "pending"}
                if all(row.get(key) == value for key, value in self.filters.items()):
                    return Response([row])
                return Response([])

        class Supabase:
            def table(self, name):
                return Query(name)

        self.assertEqual(
            load_pending_delivery_for_applied_feedback(Supabase(), 64),
            {"feedback_id": 64, "queue_id": 959, "revision": 2},
        )

    def test_workflow_runs_only_the_requested_correction(self):
        workflow = (ROOT / ".github/workflows/editorial_correction.yml").read_text()
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("feedback_id:", workflow)
        self.assertIn("python correction_pipeline.py --feedback-id", workflow)
        self.assertNotIn("python collector.py", workflow)

    def test_dispatcher_uses_vault_and_is_idempotent(self):
        migration = (
            ROOT
            / "supabase/migrations/20260909162000_immediate_editor_correction.sql"
        ).read_text()
        self.assertIn("vault.decrypted_secrets", migration)
        self.assertIn("editorial_correction.yml/dispatches", migration)
        self.assertIn("pg_advisory_xact_lock", migration)
        self.assertIn("public.dispatch_editorial_correction", migration)

    def test_webhook_dispatches_after_comment_is_saved(self):
        webhook = (ROOT / "supabase/functions/telegram-webhook/index.ts").read_text()
        submit_at = webhook.index('"submit_editorial_feedback"')
        dispatch_at = webhook.index('"dispatch_editorial_correction"')
        self.assertGreater(dispatch_at, submit_at)
        self.assertIn("Исправленная версия придёт сюда сразу", webhook)

    def test_notifier_claims_before_sending(self):
        notifier = (ROOT / "notifier.py").read_text()
        self.assertIn('"status": "notifying"', notifier)
        self.assertIn('.eq("status", "pending")', notifier)
        self.assertIn("Telegram message sent but queue status was not finalized", notifier)

    def test_editor_can_open_the_next_candidate_without_scrolling(self):
        webhook = (ROOT / "supabase/functions/telegram-webhook/index.ts").read_text()
        self.assertIn('{ kind: "next"; queueId: number; revision?: number }', webhook)
        self.assertIn('callback_data: `next:${queueId}:${revision}`', webhook)
        self.assertIn("countRemainingCandidates", webhook)
        self.assertIn("Осталось согласовать", webhook)
        self.assertIn(
            "const remainingAfterCurrent = Math.max(remainingCount - 1, 0)",
            webhook,
        )
        self.assertEqual(webhook.count("remainingAfterCurrent,"), 2)
        self.assertIn('select("id", { count: "exact", head: true })', webhook)
        self.assertIn('.gt("telegram_message_id", cursorMessageId)', webhook)
        self.assertIn("wrap to the nearest one above", webhook)
        self.assertIn('.lt("telegram_message_id", cursorMessageId)', webhook)
        self.assertIn("wrappedCandidate", webhook)
        self.assertIn("activateCallbackMessage", webhook)
        self.assertIn("Открыто в другой карточке", webhook)
        self.assertIn("Открыто ниже", webhook)
        self.assertIn('status: "notifying"', webhook)
        self.assertIn('^\\/next', webhook)
        self.assertIn("Следующая новость отправлена", webhook)


if __name__ == "__main__":
    unittest.main()
