import unittest

from semantic_dedup import (
    format_event_history,
    make_event_history_entry,
    reconcile_duplicate_decision,
)


class SemanticDedupTests(unittest.TestCase):
    def test_history_contains_source_title_summary_and_id(self):
        context = format_event_history([{
            "article_id": 52543,
            "status": "approved",
            "source_name": "HLN",
            "title": "Climate activist arrested in Germany",
            "summary": "Explosives were found near a power station.",
        }])

        self.assertIn("article_id=52543", context)
        self.assertIn("source=HLN", context)
        self.assertIn("Explosives", context)

    def test_history_is_bounded(self):
        context = format_event_history([{
            "article_id": index + 1,
            "status": "approved",
            "source_name": "source",
            "title": "A" * 500,
            "summary": "B" * 1000,
        } for index in range(100)], max_chars=800)

        self.assertLessEqual(len(context), 800)

    def test_valid_duplicate_without_update_is_blocked(self):
        analysis, blocked, note = reconcile_duplicate_decision({
            "is_duplicate_event": True,
            "duplicate_of_article_id": 52543,
            "duplicate_reason": "Same arrest and same explosives",
            "is_material_update": False,
        }, {52543, 52634})

        self.assertTrue(blocked)
        self.assertEqual(note, "Duplicate event without a material update")
        self.assertEqual(analysis["duplicate_of_article_id"], 52543)

    def test_material_update_is_allowed(self):
        analysis, blocked, _ = reconcile_duplicate_decision({
            "is_duplicate_event": True,
            "duplicate_of_article_id": "52543",
            "duplicate_reason": "Same event, but a court decision is new",
            "is_material_update": True,
        }, {52543})

        self.assertFalse(blocked)
        self.assertTrue(analysis["is_duplicate_event"])
        self.assertTrue(analysis["is_material_update"])

    def test_unknown_article_id_fails_open(self):
        analysis, blocked, note = reconcile_duplicate_decision({
            "is_duplicate_event": True,
            "duplicate_of_article_id": 99999,
            "duplicate_reason": "malformed",
            "is_material_update": False,
        }, {52543})

        self.assertFalse(blocked)
        self.assertIn("outside event history", note)
        self.assertFalse(analysis["is_duplicate_event"])
        self.assertIsNone(analysis["duplicate_of_article_id"])

    def test_in_memory_entry_keeps_current_run_coverage(self):
        entry = make_event_history_entry(
            100,
            {
                "source_name": "VRT NWS",
                "title": "A story",
                "summary": "A summary",
            },
            {"russian_summary": "Русский пересказ"},
        )

        self.assertEqual(entry["article_id"], 100)
        self.assertEqual(entry["source_name"], "VRT NWS")
        self.assertEqual(entry["russian_summary"], "Русский пересказ")


if __name__ == "__main__":
    unittest.main()
