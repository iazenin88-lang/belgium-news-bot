import json
import unittest

from prefilter_learning import (
    complete_policy_coverage,
    evaluate_policy,
    parse_policy_proposal,
    policy_prefilter_decision,
    proposal_is_safe,
    format_training_examples,
    remove_unsafe_negative_terms,
)


class PrefilterLearningTests(unittest.TestCase):
    def test_parser_accepts_only_bounded_term_lists(self):
        policy = parse_policy_proposal(json.dumps({
            "summary": "Narrow local sport",
            "rationale": "Repeated editor decisions",
            "positive_terms": ["rent indexation", "Rent Indexation"],
            "negative_terms": ["local football"],
        }))
        self.assertEqual(policy["positive_terms"], ["rent indexation"])
        self.assertEqual(policy["negative_terms"], ["local football"])

    def test_positive_signal_overrides_negative_signal(self):
        decision, _ = policy_prefilter_decision(
            "New rent rule after a local football event",
            {
                "positive_terms": ["rent rule"],
                "negative_terms": ["local football"],
            },
        )
        self.assertTrue(decision)

    def test_text_corrections_do_not_affect_relevance_metrics(self):
        rows = [
            {
                "status": "applied",
                "feedback_type": "approved",
                "source_title": "Belgian rent indexation",
            },
            {
                "status": "applied",
                "feedback_type": "topic_mismatch",
                "source_title": "Local football",
            },
            {
                "status": "applied",
                "feedback_type": "text_correction",
                "source_title": "Local football",
            },
        ]
        metrics = evaluate_policy(rows, {
            "positive_terms": ["rent"],
            "negative_terms": ["football"],
        })
        self.assertEqual(metrics["approvals"], 1)
        self.assertEqual(metrics["topic_declines"], 1)
        self.assertTrue(proposal_is_safe(metrics))

    def test_proposal_losing_approved_news_is_blocked(self):
        metrics = evaluate_policy([{
            "status": "applied",
            "feedback_type": "approved",
            "source_title": "Belgian school rules",
        }, {
            "status": "applied",
            "feedback_type": "topic_mismatch",
            "source_title": "School football",
        }], {
            "positive_terms": [],
            "negative_terms": ["school"],
        })
        self.assertFalse(proposal_is_safe(metrics))

    def test_training_examples_are_balanced_before_size_limit(self):
        rows = [
            {
                "status": "applied",
                "feedback_type": "approved",
                "source_title": f"approved {index}",
            }
            for index in range(20)
        ] + [
            {
                "status": "applied",
                "feedback_type": "topic_mismatch",
                "source_title": f"declined {index}",
            }
            for index in range(20)
        ]
        prompt = format_training_examples(rows, limit=700)
        self.assertIn("ОПУБЛИКОВАНО", prompt)
        self.assertIn("НЕ ПОДХОДИТ ТЕМАТИКА", prompt)
        self.assertIn("approved 0", prompt)
        self.assertIn("declined 0", prompt)

    def test_negative_terms_seen_in_approvals_are_removed(self):
        policy = remove_unsafe_negative_terms([{
            "status": "applied",
            "feedback_type": "approved",
            "source_title": "Belgian school reform",
        }], {
            "summary": "test",
            "rationale": "test",
            "positive_terms": [],
            "negative_terms": ["school", "football"],
        })
        self.assertEqual(policy["negative_terms"], ["football"])

    def test_policy_coverage_is_completed_with_safe_source_terms(self):
        rows = [{
            "status": "applied",
            "feedback_type": "approved",
            "source_title": "Belgian rent rules for families",
        }]
        rows.extend({
            "status": "applied",
            "feedback_type": "topic_mismatch",
            "source_title": f"International football championship match {index}",
        } for index in range(10))
        policy = complete_policy_coverage(rows, {
            "summary": "Reduce sport",
            "rationale": "Repeated sport declines",
            "positive_terms": ["rent rules", "international"],
            "negative_terms": [],
        })
        metrics = evaluate_policy(rows, policy)
        self.assertGreaterEqual(metrics["approval_retention"], 0.95)
        self.assertGreaterEqual(metrics["decline_rejection"], 0.20)
        self.assertNotIn("international", policy["positive_terms"])
        self.assertTrue(policy["negative_terms"])

    def test_coverage_terms_never_appear_in_approved_sources(self):
        rows = [{
            "status": "applied",
            "feedback_type": "approved",
            "source_title": "Belgian school reform",
        }, {
            "status": "applied",
            "feedback_type": "topic_mismatch",
            "source_title": "Local school football match",
        }]
        policy = complete_policy_coverage(rows, {
            "summary": "Reduce sport",
            "rationale": "Sport decline",
            "positive_terms": [],
            "negative_terms": [],
        })
        self.assertNotIn("school", policy["negative_terms"])
        metrics = evaluate_policy(rows, policy)
        self.assertEqual(metrics["approval_retention"], 1.0)


if __name__ == "__main__":
    unittest.main()
