import unittest

from editorial_memory import (
    EMBEDDING_PROFILE,
    article_embedding_text,
    embedding_text_hash,
    format_semantic_memory,
    score_semantic_neighbors,
)


class EditorialMemoryTests(unittest.TestCase):
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
