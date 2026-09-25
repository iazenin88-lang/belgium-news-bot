import json
import importlib.util
import sys
import types
import unittest
from types import SimpleNamespace


if "openai" not in sys.modules and importlib.util.find_spec("openai") is None:
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub

if "supabase" not in sys.modules and importlib.util.find_spec("supabase") is None:
    supabase_stub = types.ModuleType("supabase")
    supabase_stub.create_client = lambda *_args, **_kwargs: None
    sys.modules["supabase"] = supabase_stub

from analyzer import (
    calc_cost_usd,
    extract_usage_tokens,
    triage_articles_with_nano,
)


class OpenAICostTests(unittest.TestCase):
    def test_cached_input_uses_discounted_rate(self):
        cost = calc_cost_usd(
            input_tokens=10_000,
            cached_input_tokens=8_000,
            output_tokens=1_000,
        )

        self.assertEqual(str(cost), "0.002700")

    def test_extracts_cached_input_from_response_usage(self):
        response = SimpleNamespace(usage=SimpleNamespace(
            input_tokens=10_000,
            input_tokens_details=SimpleNamespace(cached_tokens=8_000),
            output_tokens=1_000,
        ))

        self.assertEqual(
            extract_usage_tokens(response),
            (10_000, 8_000, 1_000),
        )

    def test_nano_triage_uses_one_request_per_twenty_articles(self):
        class FakeResponses:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                prompt = kwargs["input"][1]["content"]
                article_ids = [
                    article_id
                    for article_id in range(1, 22)
                    if f'"article_id": {article_id}' in prompt
                ]
                return SimpleNamespace(
                    output_text=json.dumps({
                        "decisions": [
                            {
                                "article_id": article_id,
                                "decision": "pass",
                                "reason": "good",
                            }
                            for article_id in article_ids
                        ]
                    }),
                    usage=SimpleNamespace(
                        input_tokens=1_000,
                        input_tokens_details=SimpleNamespace(cached_tokens=0),
                        output_tokens=100,
                    ),
                )

        responses = FakeResponses()
        client = SimpleNamespace(responses=responses)
        articles = [
            {
                "article_id": article_id,
                "source_name": "VRT NWS",
                "title": f"Article {article_id}",
                "summary": "Summary",
                "content": "Content",
            }
            for article_id in range(1, 22)
        ]

        decisions, input_tokens, output_tokens, cost, calls = (
            triage_articles_with_nano(client, articles)
        )

        self.assertEqual(calls, 2)
        self.assertEqual(len(responses.calls), 2)
        self.assertEqual(len(decisions), 21)
        self.assertEqual(input_tokens, 2_000)
        self.assertEqual(output_tokens, 200)
        self.assertEqual(str(cost), "0.000180")
        self.assertTrue(all(row["decision"] == "pass" for row in decisions.values()))
        self.assertTrue(all(
            call["model"] == "gpt-5-nano" and
            call["reasoning"] == {"effort": "minimal"}
            for call in responses.calls
        ))

    def test_nano_request_failure_forwards_the_whole_batch_to_mini(self):
        class FailingResponses:
            @staticmethod
            def create(**_kwargs):
                raise RuntimeError("temporary Nano failure")

        client = SimpleNamespace(responses=FailingResponses())
        articles = [
            {
                "article_id": article_id,
                "source_name": "HLN",
                "title": f"Article {article_id}",
                "summary": "Summary",
                "content": "Content",
            }
            for article_id in range(1, 21)
        ]

        decisions, input_tokens, output_tokens, cost, calls = (
            triage_articles_with_nano(client, articles)
        )

        self.assertEqual(len(decisions), 20)
        self.assertTrue(
            all(row["decision"] == "uncertain" for row in decisions.values())
        )
        self.assertEqual(
            (input_tokens, output_tokens, str(cost), calls),
            (0, 0, "0", 0),
        )


if __name__ == "__main__":
    unittest.main()
