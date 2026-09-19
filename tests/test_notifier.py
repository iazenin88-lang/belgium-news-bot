import sys
import types
import unittest


if "requests" not in sys.modules:
    sys.modules["requests"] = types.ModuleType("requests")
if "supabase" not in sys.modules:
    supabase_stub = types.ModuleType("supabase")
    supabase_stub.create_client = lambda *_args, **_kwargs: None
    sys.modules["supabase"] = supabase_stub

import notifier
from notifier import build_message, build_reply_markup, notify_queue_item


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    def __init__(self, state, table_name):
        self.state = state
        self.table_name = table_name
        self.filters = {}
        self.update_values = None

    def select(self, _fields):
        return self

    def update(self, values):
        self.update_values = values
        return self

    def eq(self, field, value):
        self.filters[field] = value
        return self

    def limit(self, _value):
        return self

    def execute(self):
        if self.update_values is not None:
            queue = self.state["queue"]
            if all(queue.get(k) == v for k, v in self.filters.items()):
                queue.update(self.update_values)
                return FakeResponse([{"id": queue["id"]}])
            return FakeResponse([])
        if self.table_name == "editor_queue":
            queue = self.state["queue"]
            if all(queue.get(k) == v for k, v in self.filters.items()):
                return FakeResponse([queue])
            return FakeResponse([])
        return FakeResponse(self.state.get(self.table_name, []))


class FakeSupabase:
    def __init__(self):
        self.state = {
            "queue": {"id": 959, "article_id": 53353, "revision": 2, "status": "pending"},
            "article_analysis": [{
                "telegram_title": "Исправленный заголовок",
                "telegram_text": "Исправленный текст",
                "category": "other",
                "importance_score": 7,
            }],
            "articles": [{"canonical_url": "https://example.com/news"}],
        }

    def table(self, name):
        return FakeQuery(self.state, name)


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

    def test_notify_queue_item_uses_supported_update_result(self):
        sb = FakeSupabase()
        original_sender = notifier.telegram_send_message
        try:
            notifier.telegram_send_message = lambda **_kwargs: {
                "result": {"message_id": 1234, "chat": {"id": 247841918}}
            }
            self.assertTrue(notify_queue_item(sb, "token", "247841918", 959))
        finally:
            notifier.telegram_send_message = original_sender

        self.assertEqual(sb.state["queue"]["status"], "sent")


if __name__ == "__main__":
    unittest.main()
