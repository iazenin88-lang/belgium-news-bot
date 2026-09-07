import sys
import types
import unittest


if "requests" not in sys.modules:
    sys.modules["requests"] = types.ModuleType("requests")
if "supabase" not in sys.modules:
    supabase_stub = types.ModuleType("supabase")
    supabase_stub.create_client = lambda *_args, **_kwargs: None
    sys.modules["supabase"] = supabase_stub

from notifier import build_message, build_reply_markup


class NotifierRevisionTests(unittest.TestCase):
    def test_callbacks_include_queue_revision(self):
        markup = build_reply_markup(queue_id=42, revision=3)
        buttons = markup["inline_keyboard"][0]

        self.assertEqual(buttons[0]["callback_data"], "publish:42:3")
        self.assertEqual(buttons[1]["callback_data"], "reject:42:3")

    def test_corrected_candidate_is_labelled(self):
        message = build_message(
            article_id=7,
            analysis={
                "telegram_title": "Исправленный заголовок",
                "telegram_text": "Исправленный текст",
                "category": "housing",
                "importance_score": 8,
            },
            article={"canonical_url": "https://example.com/news"},
            revision=2,
        )

        self.assertIn("Исправленная версия 2", message)
        self.assertIn("Исправленный заголовок", message)


if __name__ == "__main__":
    unittest.main()
