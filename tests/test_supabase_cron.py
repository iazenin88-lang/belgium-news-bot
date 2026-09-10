from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = next(
    ROOT.glob("supabase/migrations/*_supabase_cron_scheduler.sql")
).read_text(encoding="utf-8")
WORKFLOW = (ROOT / ".github/workflows/full_pipeline.yml").read_text(encoding="utf-8")


class SupabaseCronConfigurationTests(unittest.TestCase):
    def test_dispatcher_uses_vault_and_brussels_gate(self) -> None:
        self.assertIn("vault.decrypted_secrets", MIGRATION)
        self.assertIn("github_actions_token", MIGRATION)
        self.assertIn("Europe/Brussels", MIGRATION)
        self.assertIn("time '09:07'", MIGRATION)
        self.assertIn("time '00:37'", MIGRATION)

    def test_job_is_idempotent_and_uses_expected_slots(self) -> None:
        self.assertIn("'full-news-pipeline-dispatch'", MIGRATION)
        self.assertRegex(MIGRATION, r"'7,37 7-23 \* \* \*'")
        self.assertIn("full_pipeline.yml/dispatches", MIGRATION)
        self.assertIn("jsonb_build_object('ref', 'main')", MIGRATION)

    def test_github_workflow_keeps_manual_dispatch_only(self) -> None:
        self.assertIn("workflow_dispatch:", WORKFLOW)
        self.assertNotRegex(WORKFLOW, r"(?m)^\s+schedule:")


if __name__ == "__main__":
    unittest.main()
