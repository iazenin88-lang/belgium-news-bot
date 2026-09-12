from pathlib import Path
import unittest

from correction_pipeline import parse_feedback_id


ROOT = Path(__file__).resolve().parents[1]


class ImmediateCorrectionTests(unittest.TestCase):
    def test_feedback_id_validation(self):
        self.assertEqual(parse_feedback_id("42"), 42)
        with self.assertRaises(ValueError):
            parse_feedback_id("0")
        with self.assertRaises(ValueError):
            parse_feedback_id("not-a-number")

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
        self.assertNotIn(".maybe_single()", notifier)
        self.assertIn("Notifier failed for", notifier)
        self.assertIn("Telegram message sent but queue status was not finalized", notifier)


if __name__ == "__main__":
    unittest.main()
