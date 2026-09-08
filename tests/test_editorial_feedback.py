import json
import unittest

from editorial_feedback import (
    build_correction_prompt,
    build_editorial_policy_context,
    parse_correction_response,
    validate_publication_length,
)


class EditorialPolicyContextTests(unittest.TestCase):
    def test_only_topic_rejections_train_relevance(self):
        rows = [
            {
                "feedback_type": "other_rejection",
                "status": "applied",
                "draft_title": "Слишком старая новость",
                "editor_comment": "Уже обсуждали",
            },
            {
                "feedback_type": "topic_mismatch",
                "status": "applied",
                "source_title": "Local football score",
                "draft_title": "Итоги матча",
                "draft_text": "Команда выиграла матч.",
                "editor_comment": "Локальный спорт без связи с жизнью в Бельгии",
            },
        ]

        context = build_editorial_policy_context(rows)

        self.assertIn("Итоги матча", context)
        self.assertIn("Локальный спорт", context)
        self.assertNotIn("Слишком старая", context)

    def test_approvals_are_positive_calibration(self):
        rows = [
            {
                "feedback_type": "topic_mismatch",
                "status": "applied",
                "draft_title": "Мировая новость",
                "editor_comment": "Нет связи с Бельгией",
            },
            {
                "feedback_type": "approved",
                "status": "applied",
                "draft_title": "Новые правила аренды",
                "draft_text": "Изменения затронут арендаторов в Бельгии.",
            },
        ]

        context = build_editorial_policy_context(rows)

        self.assertIn("ОТКЛОНЕНО ПО ТЕМАТИКЕ", context)
        self.assertIn("ОПУБЛИКОВАНО", context)
        self.assertIn("единичный отказ не запрещает", context)

    def test_context_is_bounded(self):
        rows = [
            {
                "feedback_type": "topic_mismatch",
                "status": "applied",
                "draft_title": f"Материал {index}",
                "draft_text": "x" * 1000,
                "editor_comment": "y" * 1000,
            }
            for index in range(30)
        ]

        context = build_editorial_policy_context(rows, max_chars=1600)

        self.assertLessEqual(len(context), 1600)
        self.assertIn("Материал 0", context)

    def test_recent_approvals_do_not_hide_an_older_topic_rejection(self):
        rows = [
            {
                "feedback_type": "approved",
                "status": "applied",
                "draft_title": f"Принято {index}",
            }
            for index in range(20)
        ]
        rows.append({
            "feedback_type": "topic_mismatch",
            "status": "applied",
            "draft_title": "Не наша тема",
            "editor_comment": "Нет практической связи с Бельгией",
        })

        context = build_editorial_policy_context(rows)

        self.assertIn("Не наша тема", context)
        self.assertIn("ОПУБЛИКОВАНО", context)

    def test_text_correction_trains_future_style_without_topic_rejection(self):
        context = build_editorial_policy_context([{
            "feedback_type": "text_correction",
            "status": "applied",
            "editor_comment": "Слишком длинно и слово написано неправильно",
            "draft_title": "Длинный заголовок",
            "draft_text": "Старый длинный текст",
            "revised_title": "Короткий заголовок",
            "revised_text": "Короткий исправленный текст.",
        }])

        self.assertIn("ИСПРАВЛЕНИЕ СТИЛЯ", context)
        self.assertIn("Слишком длинно", context)
        self.assertIn("После исправления", context)
        self.assertIn("только как обязательные правила стиля", context)

    def test_length_limit_accepts_short_and_rejects_long_publication(self):
        validate_publication_length("Короткий заголовок", "Короткий текст новости.")
        with self.assertRaises(ValueError):
            validate_publication_length("Заголовок", "слово " * 71)

    def test_topic_history_cannot_crowd_out_style_memory(self):
        rows = [{
            "feedback_type": "topic_mismatch",
            "status": "applied",
            "draft_title": f"Тема {index}",
            "editor_comment": "Не подходит",
        } for index in range(12)]
        rows.append({
            "feedback_type": "text_correction",
            "status": "applied",
            "editor_comment": "Писать короче",
            "draft_text": "Длинный текст",
            "revised_text": "Короткий текст",
        })

        context = build_editorial_policy_context(rows)
        self.assertIn("Писать короче", context)


class CorrectionPromptTests(unittest.TestCase):
    def test_prompt_contains_editor_comment_and_source(self):
        prompt = build_correction_prompt(
            {
                "source_name": "HLN",
                "title": "Bronkop",
                "summary": "Brontekst",
                "content": "Volledige tekst",
                "canonical_url": "https://example.com/news",
            },
            {
                "category": "housing",
                "telegram_title": "Черновик",
                "telegram_text": "Текст черновика",
            },
            "Исправить число и падеж",
        )

        self.assertIn("Исправить число и падеж", prompt)
        self.assertIn("Bronkop", prompt)
        self.assertIn("Текст черновика", prompt)

    def test_parser_accepts_json_fence(self):
        raw = "```json\n" + json.dumps(
            {"telegram_title": "Новый заголовок", "telegram_text": "Новый текст"},
            ensure_ascii=False,
        ) + "\n```"

        title, text = parse_correction_response(raw)

        self.assertEqual(title, "Новый заголовок")
        self.assertEqual(text, "Новый текст")

    def test_parser_rejects_empty_text(self):
        with self.assertRaises(ValueError):
            parse_correction_response('{"telegram_title":"Заголовок","telegram_text":""}')


if __name__ == "__main__":
    unittest.main()
