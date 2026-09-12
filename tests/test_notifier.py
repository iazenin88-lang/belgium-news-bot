import sys
import types
import unittest


if "requests" not in sys.modules:
    sys.modules["requests"] = types.ModuleType("requests")
if "supabase" not in sys.modules:
    supabase_stub = types.ModuleType("supabase")
    supabase_stub.create_client = lambda *_args, **_kwargs: None
    sys.modules["supabase"] = supabase_stub

from notifier import build_message, build_reply_markup, notify_queue_item


class _Response:
    def __init__(self, data):
        self.data = data


class _FakeBuilder:
    """Small Supabase builder double with list-only update responses."""

    def __init__(self, client, table_name):
        self.client = client
        self.table_name = table_name
        self.operation = None
        self.payload = None
        self.filters = {}

    def select(self, _columns):
        if self.operation is None:
            self.operation = "select"
        return self

    def update(self, payload):
        self.operation = "update"
        self.payload = payload
        return self

    def eq(self, column, value):
        self.filters[column] = value
        return self

    def limit(self, _value):
        return self

    def execute(self):
        self.client.calls.append((self.table_name, self.operation, self.payload, self.filters))
        if self.table_name == "editor_queue" and self.operation == "select":
            return _Response([{"id": 42, "article_id": 7, "revision": 1}])
        if self.table_name == "editor_queue" and self.payload == {"status": "notifying"}:
            return _Response([{"id": 42}])
        if (
            self.table_name == "editor_queue"
            and isinstance(self.payload, dict)
            and self.payload.get("status") == "sent"
        ):
            return _Response([{"id": 42}])
        if self.table_name == "article_analysis":
            return _Response([{"telegram_title": "Заголовок", "telegram_text": "Текст"}])
        if self.table_name == "articles":
            return _Response([{"canonical_url": "https://example.com/news"}])
        raise AssertionError(f"Unexpected Supabase operation: {self.table_name} {self.operation}")


class _FakeSupabase:
    def __init__(self):
        self.calls = []

    def table(self, table_name):
        return _FakeBuilder(self, table_name)


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

    def test_update_claim_and_finalize_use_list_responses(self):
        import notifier

        client = _FakeSupabase()
        original_sender = notifier.telegram_send_message
        notifier.telegram_send_message = lambda **_kwargs: {
            "result": {"message_id": 123, "chat": {"id": 456}}
        }
        try:
            self.assertTrue(notify_queue_item(client, "token", "chat", 42))
        finally:
            notifier.telegram_send_message = original_sender

        updates = [call for call in client.calls if call[1] == "update"]
        self.assertEqual([call[2] for call in updates], [
            {"status": "notifying"},
            {
                "status": "sent",
                "telegram_chat_id": 456,
                "telegram_message_id": 123,
                "last_sent_at": updates[1][2]["last_sent_at"],
            },
        ])


if __name__ == "__main__":
    unittest.main()
