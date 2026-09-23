"""Formatting for the per-run Telegram pipeline report."""


def build_run_report(
    *,
    collected: int,
    prefilter_rejected: int,
    ai_rejected: int,
    passed_to_editor: int,
    run_cost_usd,
    spent_total_usd,
    remaining_estimated_usd,
    errors: int = 0,
    corrections_processed: int = 0,
    corrections_failed: int = 0,
) -> str:
    """Build a compact article funnel, followed by secondary cost details."""
    lines = [
        "📊 Итоги обработки новостей",
        "",
        f"🆕 Собрано новых статей: {collected}",
        f"🗑 Отсеяно префильтром: {prefilter_rejected}",
        f"🤖 Отсеяно AI: {ai_rejected}",
        f"✅ Передано в редакторский чат: {passed_to_editor}",
    ]

    if errors:
        lines.append(f"⚠️ Ошибок обработки: {errors}")

    if corrections_processed or corrections_failed:
        lines.extend([
            "",
            f"✏️ Исправлений применено: {corrections_processed}",
            f"⚠️ Исправлений ожидают повтора: {corrections_failed}",
        ])

    lines.extend([
        "",
        "💰 OpenAI",
        f"Этот запуск: ${run_cost_usd}",
        f"Потрачено всего: ${spent_total_usd}",
        f"Остаток: ${remaining_estimated_usd}",
    ])
    return "\n".join(lines)
