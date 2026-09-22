import unittest

from editorial_feedback import enforce_ordinary_person_relevance_guard


class OrdinaryPersonRelevanceGuardTests(unittest.TestCase):
    def test_blocks_athlete_incident_reframed_as_safety(self):
        relevant, reason = enforce_ordinary_person_relevance_guard({
            "is_relevant": True,
            "category": "safety",
            "importance_score": 6,
            "is_public_figure_personal_incident": True,
            "passes_ordinary_person_test": False,
            "reason": "Belgian cyclist was hospitalized during training.",
        })

        self.assertFalse(relevant)
        self.assertIn("тест неизвестного человека", reason)

    def test_keeps_story_with_independent_public_consequence(self):
        relevant, reason = enforce_ordinary_person_relevance_guard({
            "is_relevant": True,
            "is_public_figure_personal_incident": True,
            "passes_ordinary_person_test": True,
            "reason": "The incident led to a national road-safety rule change.",
        })

        self.assertTrue(relevant)
        self.assertIn("road-safety", reason)

    def test_requires_guard_fields(self):
        with self.assertRaises(ValueError):
            enforce_ordinary_person_relevance_guard({"is_relevant": True})


if __name__ == "__main__":
    unittest.main()
