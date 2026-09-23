import unittest

from run_report import build_run_report


class RunReportTests(unittest.TestCase):
    def test_shows_article_funnel_in_russian(self):
        report = build_run_report(
            collected=20,
            prefilter_rejected=2,
            ai_rejected=17,
            passed_to_editor=1,
            run_cost_usd="0.053856",
            spent_total_usd="12.778692",
            remaining_estimated_usd="2.274331",
        )

        self.assertIn("Собрано новых статей: 20", report)
        self.assertIn("Отсеяно префильтром: 2", report)
        self.assertIn("Отсеяно AI: 17", report)
        self.assertIn("Передано в редакторский чат: 1", report)
        self.assertNotIn("AI calls", report)

    def test_only_shows_optional_operational_details_when_present(self):
        report = build_run_report(
            collected=4,
            prefilter_rejected=1,
            ai_rejected=1,
            passed_to_editor=1,
            errors=1,
            corrections_processed=2,
            corrections_failed=1,
            run_cost_usd="0.01",
            spent_total_usd="1.01",
            remaining_estimated_usd="8.99",
        )

        self.assertIn("Ошибок обработки: 1", report)
        self.assertIn("Исправлений применено: 2", report)
        self.assertIn("Исправлений ожидают повтора: 1", report)


if __name__ == "__main__":
    unittest.main()
