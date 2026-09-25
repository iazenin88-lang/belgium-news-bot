import json
import unittest

from nano_triage import (
    NANO_TRIAGE_BATCH_SIZE,
    NANO_TRIAGE_SYSTEM_PROMPT,
    article_batches,
    build_nano_triage_prompt,
    calc_nano_cost_usd,
    parse_nano_triage,
)


def make_articles(count):
    return [
        {
            "article_id": index,
            "source_name": "VRT NWS",
            "title": f"Article {index}",
            "summary": "Summary",
            "content": "Content",
        }
        for index in range(1, count + 1)
    ]


class NanoTriageTests(unittest.TestCase):
    def test_batches_every_article_without_a_daily_quota(self):
        batches = article_batches(make_articles(45))

        self.assertEqual([len(batch) for batch in batches], [20, 20, 5])
        self.assertEqual(
            [article["article_id"] for batch in batches for article in batch],
            list(range(1, 46)),
        )
        self.assertEqual(NANO_TRIAGE_BATCH_SIZE, 20)

    def test_all_articles_can_pass(self):
        decisions = parse_nano_triage(json.dumps({
            "decisions": [
                {"article_id": article_id, "decision": "pass", "reason": "good"}
                for article_id in range(1, 21)
            ]
        }), range(1, 21))

        self.assertEqual(len(decisions), 20)
        self.assertTrue(all(row["decision"] == "pass" for row in decisions.values()))

    def test_all_articles_can_be_rejected(self):
        decisions = parse_nano_triage(json.dumps({
            "decisions": [
                {
                    "article_id": article_id,
                    "decision": "reject",
                    "reason": "routine sport",
                }
                for article_id in range(1, 21)
            ]
        }), range(1, 21))

        self.assertEqual(len(decisions), 20)
        self.assertTrue(
            all(row["decision"] == "reject" for row in decisions.values())
        )

    def test_missing_decision_fails_open_and_invented_id_is_ignored(self):
        decisions = parse_nano_triage(json.dumps({
            "decisions": [
                {"article_id": 1, "decision": "reject", "reason": "no"},
                {"article_id": 999, "decision": "reject", "reason": "invented"},
            ]
        }), [1, 2])

        self.assertEqual(decisions[1]["decision"], "reject")
        self.assertEqual(decisions[2]["decision"], "uncertain")
        self.assertNotIn(999, decisions)

    def test_prompt_contains_every_id_and_explicit_independent_rule(self):
        prompt = build_nano_triage_prompt(make_articles(3), "Recent decisions")

        self.assertIn('"article_id": 1', prompt)
        self.assertIn('"article_id": 2', prompt)
        self.assertIn('"article_id": 3', prompt)
        self.assertIn("НЕЗАВИСИМО", NANO_TRIAGE_SYSTEM_PROMPT)
        self.assertIn("нет квоты", NANO_TRIAGE_SYSTEM_PROMPT)

    def test_nano_cost_uses_nano_and_cached_rates(self):
        cost = calc_nano_cost_usd(
            input_tokens=10_000,
            cached_input_tokens=8_000,
            output_tokens=1_000,
        )

        self.assertEqual(str(cost), "0.000540")


if __name__ == "__main__":
    unittest.main()
