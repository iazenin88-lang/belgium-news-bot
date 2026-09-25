import unittest
from pathlib import Path

from editorial_memory import (
    EMBEDDING_PROFILE,
    article_embedding_text,
    embedding_text_hash,
    format_semantic_memory,
    score_semantic_neighbors,
    semantic_rescue_recommended,
)

class EditorialMemoryTests(unittest.TestCase):
    def test_main_uses_shadow_memory_to_rescue_nano_rejections(self):
        source = (
            Path(__file__).resolve().parents[1] / "analyzer.py"
        ).read_text(encoding="utf-8")
        main_source = source[source.index("def main():"):]
        semantic_call = main_source.index(
            "semantic_memory_context, semantic_score = load_semantic_memory_context"
        )
        nano_decision = main_source.index(
            'nano_decision = triage_result.get("decision", "uncertain")'
        )
        semantic_rescue = main_source.index(
            "and semantic_rescue_recommended(semantic_score)"
        )
        self.assertLess(semantic_call, nano_decision)
        self.assertLess(nano_decision, semantic_rescue)

    def test_embedding_text_is_stable_and_uses_source_content(self):
        article = {
            "title": "  Macron asks EU  ",
            "summary": "Fuel-price proposal",
            "content": "Possible consequences.",
        }
        text = article_embedding_text(article)
        self.assertEqual(
            text,
            "TITLE: Macron asks EU\n"
            "SUMMARY: Fuel-price proposal\n"
            "CONTENT: Possible consequences.",
        )
        self.assertEqual(embedding_text_hash(text), embedding_text_hash(text))
        self.assertIn("editorial-v1", EMBEDDING_PROFILE)

    def test_shadow_prediction_uses_balanced_label_scores(self):
        rows = [
            {"feedback_type": "topic_mismatch", "similarity": 0.92},
            {"feedback_type": "topic_mismatch", "similarity": 0.89},
            {"feedback_type": "approved", "similarity": 0.61},
        ]
        score = score_semantic_neighbors(rows)
        self.assertEqual(score["prediction"], "topic_mismatch")
        self.assertGreater(score["rejection_score"], score["approval_score"])

    def test_approval_prediction_can_only_rescue_for_ai_review(self):
        self.assertTrue(semantic_rescue_recommended({
            "prediction": "approved",
            "confidence": 0.08,
            "approval_examples": 3,
        }))
        self.assertFalse(semantic_rescue_recommended({
            "prediction": "topic_mismatch",
            "confidence": 0.40,
            "approval_examples": 3,
        }))

    def test_domestic_source_gate_precedes_keyword_heuristics(self):
        source = (
            Path(__file__).resolve().parents[1] / "analyzer.py"
        ).read_text(encoding="utf-8")
        domestic_gate = source.index("if is_belgian_domestic_source(source_name)")
        learned_rejection = source.index("if learned_decision is False")
        keyword_heuristics = source.index("pass_matches = count_matches")
        self.assertLess(domestic_gate, learned_rejection)
        self.assertLess(domestic_gate, keyword_heuristics)

    def test_location_neutrality_is_explicit_in_relevance_prompt(self):
        source = (
            Path(__file__).resolve().parents[1] / "analyzer.py"
        ).read_text(encoding="utf-8")
        self.assertIn("ГЕОГРАФИЧЕСКАЯ НЕЙТРАЛЬНОСТЬ", source)
        self.assertIn("editorial_interest_score", source)

    def test_close_scores_remain_uncertain(self):
        rows = [
            {"feedback_type": "topic_mismatch", "similarity": 0.81},
            {"feedback_type": "approved", "similarity": 0.80},
        ]
        self.assertEqual(
            score_semantic_neighbors(rows)["prediction"],
            "uncertain",
        )

    def test_context_contains_both_labels_and_similarity(self):
        context = format_semantic_memory([
            {
                "feedback_type": "approved",
                "similarity": 0.84,
                "source_title": "Belgian rent rules",
            },
            {
                "feedback_type": "topic_mismatch",
                "similarity": 0.88,
                "source_title": "French fuel-price proposal",
                "editor_comment": "Global news",
            },
        ])
        self.assertIn("ОПУБЛИКОВАНО", context)
        self.assertIn("ОТКЛОНЕНО ПО ТЕМАТИКЕ", context)
        self.assertIn("similarity=0.880", context)
        self.assertIn("Global news", context)


if __name__ == "__main__":
    unittest.main()
