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

from analyzer import calc_cost_usd, extract_usage_tokens


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


if __name__ == "__main__":
    unittest.main()
