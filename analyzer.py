"""
analyzer.py

Назначение:
анализирует новые статьи из таблицы articles и сохраняет результат в article_analysis.

Дополнительно:
- обрабатывает комментарии редактора и повторно ставит исправленные версии в очередь
- учитывает тематические отклонения при следующем AI-отборе
- считает стоимость OpenAI за текущий прогон
- пишет статистику в ai_runs
- обновляет ai_balance
- после прогона отправляет в Telegram сообщение о расходах

Важно:
- стоимость считается по заданным в коде тарифам
- остаток баланса является оценочным:
    remaining = starting_balance_usd - spent_total_usd
"""

import json
import os
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import requests
from openai import OpenAI
from supabase import create_client

from editorial_feedback import (
    CORRECTION_SYSTEM_PROMPT,
    build_correction_prompt,
    build_editorial_policy_context,
    parse_correction_response,
    validate_publication_length,
)
from prefilter_learning import (
    PROPOSAL_SYSTEM_PROMPT,
    evaluate_policy,
    format_training_examples,
    parse_policy_proposal,
    policy_prefilter_decision,
    proposal_is_safe,
)
from semantic_dedup import (
    EVENT_HISTORY_LIMIT,
    EVENT_HISTORY_LOOKBACK_DAYS,
    format_event_history,
    make_event_history_entry,
    parse_article_id,
    reconcile_duplicate_decision,
)


# -------------------------------------------------------
# Модель OpenAI
# -------------------------------------------------------
MODEL = "gpt-5-mini"


# -------------------------------------------------------
# Тарифы модели
#
# ВАЖНО:
# при необходимости поменяй вручную под актуальные цены.
# Значения ниже задаются в долларах за 1 миллион токенов.
# -------------------------------------------------------
INPUT_COST_PER_1M = Decimal("0.250000")
OUTPUT_COST_PER_1M = Decimal("2.000000")


# -------------------------------------------------------
# Telegram
# -------------------------------------------------------
TELEGRAM_API_BASE = "https://api.telegram.org"


# -------------------------------------------------------
# Системный prompt для AI
# -------------------------------------------------------
SYSTEM_PROMPT = """
Ты анализируешь новостные статьи для Telegram-канала на русском языке
для русскоязычных жителей Бельгии.

Нужно определить:
1. интересна ли новость этой аудитории
2. категория новости
3. общественная важность по шкале 1-10
4. практическая польза по шкале 1-10
5. краткая причина
6. короткий пересказ на русском
7. короткий заголовок для Telegram
8. текст поста для Telegram

Считать релевантными в первую очередь:
- миграция, визы, ВНЖ, убежище, украинские беженцы
- все содержательные новости о жилье и аренде в Бельгии: цены и индексация,
  договоры и гарантии, права арендаторов и обязанности владельцев,
  доступность жилья, социальное жильё и коммунальные правила
- правила краткосрочной аренды через Airbnb и другие платформы, а также
  инициативы ЕС, если они могут повлиять на предложение жилья, цены,
  права жильцов или полномочия бельгийских городов
- работа, зарплаты, налоги, пособия
- транспорт, школы, медицина, безопасность
- стоимость жизни и способы разумно сократить регулярные расходы в Бельгии
- сравнения тарифов и существенные акции на мобильную связь, интернет,
  электричество, газ, страхование и банковские услуги
- права потребителей, изменения цен, тарифов и условий договоров
- изменения законов и правил в Бельгии и ЕС, влияющие на жизнь в Бельгии
- важные новости о Брюсселе, Фландрии, Валлонии
- новости соседних стран, если они реально влияют на жизнь людей в Бельгии
- новости про Россию, россиян, русских, если они могут быть значимы
  для русскоязычной аудитории в Бельгии
- отдельные темы развлечений, если они реально заметны для русскоязычной аудитории в Бельгии
  или имеют практический/общественный контекст

Считать нерелевантными:
- обычные мировые новости без практической связи с жизнью в Бельгии
- спорт, криминальные мелочи, если нет практической пользы
- локальные мелочи без заметного влияния на читателей
- развлекательные новости низкой значимости без общественного или практического смысла
- реклама одного бренда без сравнения, ясных условий или заметной выгоды
- скидки на отдельные необязательные товары и партнёрские рекламные подборки

Оценивай два разных свойства новости:
- importance_score — общественная значимость, масштаб последствий и число затронутых людей
- practical_value_score — насколько информация помогает читателю прямо сейчас:
  сэкономить заметную сумму, выбрать услугу, избежать лишних расходов,
  выполнить обязательное действие или уложиться в срок

Не занижай practical_value_score только потому, что новость не связана с законом
или политикой. Конкретное сравнение нескольких предложений на бельгийском рынке
со значимой экономией обычно заслуживает 7-8. Практический материал для более
узкой группы — 6. Рекламный текст без сравнения и проверяемых условий — не выше 4.

Содержательная новость об аренде или доступности жилья в Бельгии обычно имеет
practical_value_score не ниже 6. То же относится к конкретному предложению ЕС,
которое может изменить правила аренды или полномочия бельгийских городов,
даже если оно ещё не принято. В таком случае ясно укажи, что это предложение,
а не действующий закон, и используй категорию housing.

Считай новость релевантной, если хотя бы общественная важность или практическая
польза составляет 6 и материал действительно относится к жизни в Бельгии.

Проверяй, не описывает ли статья то же самое реальное событие, которое уже
есть в блоке «УЖЕ ПРЕДЛОЖЕННЫЕ ИЛИ ОДОБРЕННЫЕ МАТЕРИАЛЫ». Событие считается
повтором, если совпадают конкретный инцидент, люди/организации, место и
действие, даже когда статья пришла с другого сайта и заголовок сформулирован
иначе. Одной общей темы (например, «аренда» или «безопасность») недостаточно.

Если это тот же инцидент, но появились существенные новые факты, новый этап
расследования, решение властей или заметные последствия, укажи
is_duplicate_event=true и is_material_update=true: такой материал можно
предложить редактору. Если существенного обновления нет, укажи
is_duplicate_event=true и is_material_update=false: материал не нужно снова
предлагать. Не считай материал повтором, если не уверен.

Категории используй только из списка:
migration, housing, work, taxes, transport, education, healthcare, social, politics, safety, europe, other

Верни строго JSON без пояснений.
"""

USER_TEMPLATE = """
Проанализируй статью.

Источник: {source_name}
Заголовок: {title}
Summary: {summary}
Content: {content}
URL: {url}

{editorial_policy_context}

СВЕРКА С НЕДАВНИМ ПОКРЫТИЕМ СОБЫТИЙ
{event_history_context}

Сравнивай статью с этим списком только по конкретному событию, а не по общей
теме. Если есть совпадение, укажи точный article_id из списка.

Верни JSON такого вида:
{{
  "is_relevant": true,
  "category": "migration",
  "importance_score": 8,
  "practical_value_score": 7,
  "reason": "Коротко почему новость важна",
  "russian_summary": "Короткий пересказ на русском, 2-4 предложения.",
  "telegram_title": "Короткий заголовок",
  "telegram_text": "Готовый короткий текст для Telegram без markdown.",
  "is_duplicate_event": false,
  "duplicate_of_article_id": null,
  "duplicate_reason": "",
  "is_material_update": false
}}
"""


# -------------------------------------------------------
# Отдельная редактура готового текста перед editor_queue
# -------------------------------------------------------
EDITOR_SYSTEM_PROMPT = """
Ты — выпускающий редактор русскоязычного новостного Telegram-канала.

Тебе дают исходную статью и уже подготовленный черновик публикации.
Твоя задача — не оценивать и не отклонять материал, а вернуть полностью
отредактированный вариант, готовый для показа редактору канала.

Обязательные требования:
- используй естественный современный литературный русский язык;
- исправляй орфографию, грамматику, пунктуацию и неудачный порядок слов;
- устраняй буквальные переводы, кальки, канцелярит, искусственно образованные
  и несуществующие слова, в том числе формы вроде «хвасты» и «сгенерация»;
- каждое предложение должно быть понятным и логически связанным с соседними;
- при первом упоминании человека укажи полное имя и кратко объясни, кто он;
- при первом упоминании малоизвестной партии или организации поясни страну
  и её роль, если это следует из источника;
- заголовок должен быть понятен без предварительного знания новости;
- проверяй, что категория соответствует главной теме статьи;
- не добавляй факты, причинно-следственные связи и политические выводы,
  которых нет в исходной статье;
- не придумывай связь с Бельгией, миграцией или жизнью читателей;
- не добавляй шаблонный вывод о значении новости для Бельгии. Упоминай
  последствия только тогда, когда они конкретны и подтверждаются источником;
- если фрагмент черновика искажён или не подтверждается источником, восстанови
  его по источнику либо удали. Ничего не додумывай;
- сохрани точный смысл цитат, чисел, дат и названий;
- текст должен быть кратким, связным и без Markdown-разметки.
- заголовок и основной текст вместе — не более 70 слов, оптимально 45–60 слов;
- после короткого заголовка дай один абзац из 2–3 коротких предложений;
- весь материал должен читаться за 15–20 секунд: подробности читатель получит
  по ссылке на оригинал.

Перед ответом перечитай заголовок и каждое предложение как корректор.
Верни строго JSON без пояснений и без списка внесённых изменений.
"""

EDITOR_USER_TEMPLATE = """
Отредактируй публикацию, сверяя её с исходной статьёй.

ИСХОДНАЯ СТАТЬЯ
Источник: {source_name}
Заголовок: {source_title}
Summary: {source_summary}
Content: {source_content}
URL: {url}

РЕДАКЦИОННАЯ ПАМЯТЬ
{editorial_policy_context}

ЧЕРНОВИК ПУБЛИКАЦИИ
Категория: {draft_category}
Заголовок: {draft_title}
Текст: {draft_text}

Верни JSON такого вида:
{{
  "category": "politics",
  "telegram_title": "Полностью отредактированный заголовок",
  "telegram_text": "Полностью отредактированный текст"
}}
"""

ALLOWED_CATEGORIES = {
    "migration", "housing", "work", "taxes", "transport", "education",
    "healthcare", "social", "politics", "safety", "europe", "other",
}


# -------------------------------------------------------
# Ключевые слова для pre-filter
# -------------------------------------------------------
HARD_REJECT_KEYWORDS = {
    "football", "soccer", "champions league", "premier league", "uefa",
    "basketball", "tennis tournament", "formula 1", "motogp",
    "movie star", "red carpet", "fashion week",
    "album release", "box office", "love island",
    "royal gossip", "tv show", "reality show",
    "transfer rumor", "match preview", "match report", "line-up",
    "livestream sports", "sports betting",
}

HOUSING_RENTAL_KEYWORDS = {
    # English
    "housing", "affordable housing", "housing shortage", "rent", "rents",
    "rental", "renting", "landlord", "tenant", "tenant rights", "lease",
    "leases", "security deposit",
    "rent indexation", "rent cap", "short-term rental", "holiday rental",
    "rental platform", "social housing", "mortgage", "mortgages", "airbnb",

    # Nederlands
    "huur", "verhuur", "huren", "huurder", "huurders", "verhuurder",
    "verhuurders", "huurprijs", "huurprijzen", "huurcontract",
    "huurcontracten", "huurovereenkomst", "huurovereenkomsten",
    "huurwaarborg", "huurwaarborgen", "huurindexatie", "huurwoning",
    "huurwoningen", "woninghuur", "woning", "woningen", "woningmarkt",
    "betaalbaar wonen", "wooncrisis",
    "sociale woning", "kortetermijnverhuur", "vakantieverhuur",
    "verhuurplatform", "verhuurplatformen", "vastgoed", "vastgoedmarkt",

    # Français
    "logement", "logements", "logement abordable", "crise du logement",
    "loyer", "loyers", "location", "locations", "locataire", "locataires",
    "bailleur", "bailleurs", "bail", "baux",
    "garantie locative", "indexation du loyer", "plafonnement des loyers",
    "logement social", "location de courte durée", "location touristique",
    "plateforme de location",
}

MULTILINGUAL_POLICY_KEYWORDS = {
    "wetsvoorstel", "wetgeving", "regelgeving", "regulering", "huurregels",
    "proposition de loi", "projet de loi", "législation", "réglementation",
    "régulation", "règles de location",
}

PASS_KEYWORDS = {
    "belgium", "belgian", "brussels", "flanders", "wallonia", "antwerp", "ghent",
    "belgië", "brussel", "vlaanderen", "wallonië", "gent", "antwerpen",

    "visa", "residence permit", "permit", "asylum", "refugee", "refugees",
    "migrant", "migrants", "immigration", "integration", "expat", "foreign worker",
    "temporary protection", "residency", "residence card",

    *HOUSING_RENTAL_KEYWORDS,
    "salary", "wage", "employment", "job market", "unemployment",
    "tax", "taxes", "benefit", "benefits", "pension", "allowance",

    "school", "education", "transport", "rail", "train", "tram", "bus",
    "healthcare", "hospital", "doctor", "medicine", "insurance",
    "safety", "police", "court", "law", "legal",

    "ukrainian", "ukrainians", "ukraine",

    "russia", "russian", "russians",
    "россия", "россияне", "русские", "русский",

    "entertainment", "festival", "concert", "cinema", "music", "cultural event",
    "culture", "event", "events",
}

HIGH_SIGNAL_KEYWORDS = {
    "new law", "law", "rules", "policy", "ban", "decision", "court",
    *MULTILINGUAL_POLICY_KEYWORDS,
    "tax", "taxes", "visa", "permit", "pension", "housing", "rent",
    "transport", "strike", "school", "benefits", "immigration",
    "refugee", "refugees", "asylum", "healthcare", "insurance",
    "border", "customs", "residence", "residency", "work permit",
    "salary", "wage", "unemployment", "education", "police", "safety",
}

CONSUMER_SERVICE_KEYWORDS = {
    "telecom", "mobile plan", "mobile subscription", "mobile operator",
    "broadband", "internet provider", "fixed internet", "phone contract",
    "energy contract", "electricity contract", "gas contract",
    "insurance premium", "insurer", "bank account", "bank fee",

    "telecomoperator", "mobiel abonnement", "gsm-abonnement",
    "internetabonnement", "vast internet", "energiecontract",
    "elektriciteitscontract", "gascontract", "verzekering", "verzekeraar",
    "bankrekening", "bankkosten",

    "télécom", "forfait mobile", "abonnement mobile", "abonnement internet",
    "fournisseur d'internet", "contrat d'énergie", "contrat d'électricité",
    "assurance", "frais bancaires",
}

SAVINGS_KEYWORDS = {
    "discount", "discounts", "promotion", "saving", "savings", "cheaper",
    "price comparison", "compare prices", "switch provider", "first year",
    "cashback",

    "korting", "kortingen", "promotie", "aanbieding", "voordeel",
    "besparen", "goedkoper", "prijsvergelijking", "prijzen vergelijken",
    "overstappen", "eerste jaar",

    "réduction", "remise", "promotion", "économie", "moins cher",
    "comparateur", "changer de fournisseur", "première année",
}


# -------------------------------------------------------
# Вспомогательные функции
# -------------------------------------------------------
def get_env(name: str, required: bool = True) -> str:
    """
    Получает переменную окружения.
    """
    value = os.environ.get(name)
    if required and not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value or ""


def get_supabase():
    """
    Создаёт клиент Supabase.
    """
    return create_client(
        get_env("SUPABASE_URL"),
        get_env("SUPABASE_SERVICE_KEY"),
    )


def get_openai():
    """
    Создаёт клиент OpenAI.
    """
    return OpenAI(api_key=get_env("OPENAI_API_KEY"))


def pick_text(response: Any) -> str:
    """
    Достаёт текст из ответа OpenAI Responses API.
    """
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text.strip()

    parts: list[str] = []

    try:
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue

            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    text = getattr(c, "text", "")
                    if text:
                        parts.append(text)
    except Exception:
        pass

    return "\n".join(parts).strip()


def normalize_text(value: Any, max_len: int = 4000) -> str:
    """
    Приводит значение к строке и обрезает по длине.
    """
    if value is None:
        return ""
    return str(value).strip()[:max_len]


def normalize_int(value: Any, default: int = 1) -> int:
    """
    Безопасно приводит значение к int и ограничивает диапазон 1..10.
    """
    try:
        n = int(value)
    except Exception:
        return default

    return max(1, min(10, n))


def contains_any(text: str, keywords: set[str]) -> bool:
    """
    Проверяет, содержит ли текст хотя бы одно ключевое слово.
    """
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)


def count_matches(text: str, keywords: set[str]) -> int:
    """
    Подсчитывает, сколько ключевых слов встретилось в тексте.
    """
    text_lower = text.lower()
    return sum(1 for keyword in keywords if keyword in text_lower)


def count_whole_term_matches(text: str, keywords: set[str]) -> int:
    """
    Считает отдельные слова и фразы, не принимая, например, rent в current
    за упоминание аренды.
    """
    text_lower = text.lower()
    return sum(
        1
        for keyword in keywords
        if re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text_lower)
    )


def quantize_money(value: Decimal) -> Decimal:
    """
    Округляет денежное значение до 6 знаков после запятой.
    """
    return value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


def now_iso() -> str:
    """
    Возвращает текущее время в ISO UTC.
    """
    return datetime.now(timezone.utc).isoformat()


# -------------------------------------------------------
# Telegram helper
# -------------------------------------------------------
def telegram_send_message(bot_token: str, chat_id: str, text: str) -> None:
    """
    Отправляет сообщение в Telegram.
    """
    url = f"{TELEGRAM_API_BASE}/bot{bot_token}/sendMessage"

    response = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
        },
        timeout=30,
    )
    response.raise_for_status()


# -------------------------------------------------------
# Более строгий pre-filter перед OpenAI
# -------------------------------------------------------
def should_send_to_ai(
    article: dict[str, Any],
    *,
    learning_mode: bool = False,
    learned_policy: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """
    Более строгий pre-filter перед OpenAI.
    """
    title = normalize_text(article.get("title"), 1000)
    summary = normalize_text(article.get("summary"), 4000)
    content = normalize_text(article.get("content"), 8000)

    combined = f"{title}\n{summary}\n{content}".strip()
    combined_lower = combined.lower()

    if not title and not summary and not content:
        return False, "Нет заголовка, summary и content"

    learned_decision, learned_reason = policy_prefilter_decision(
        combined_lower, learned_policy
    )
    if learned_decision is not None:
        return learned_decision, learned_reason

    if contains_any(combined_lower, HARD_REJECT_KEYWORDS):
        return False, "Явно нерелевантная тема (спорт и т.п.)"

    if learning_mode:
        return True, "Learning mode: пограничный материал передан AI"

    pass_matches = count_matches(combined_lower, PASS_KEYWORDS)
    high_signal_matches = count_matches(combined_lower, HIGH_SIGNAL_KEYWORDS)
    consumer_service_matches = count_matches(combined_lower, CONSUMER_SERVICE_KEYWORDS)
    savings_matches = count_matches(combined_lower, SAVINGS_KEYWORDS)
    housing_rental_matches = count_whole_term_matches(
        combined_lower,
        HOUSING_RENTAL_KEYWORDS,
    )

    summary_len = len(summary)
    content_len = len(content)

    practical_keywords = {
        "visa", "permit", "residence permit", "asylum", "refugee", "migrant",
        "salary", "wage", "employment",
        "tax", "taxes", "pension", "benefit", "benefits",
        "school", "education", "transport", "train", "tram", "bus",
        "healthcare", "hospital", "insurance", "doctor",
        "police", "court", "law", "legal", "safety",
        "temporary protection", "work permit",
    }

    practical_matches = (
        count_matches(combined_lower, practical_keywords)
        + housing_rental_matches
    )

    if (
        housing_rental_matches >= 1
        and (len(title) >= 25 or summary_len >= 60 or content_len >= 180)
    ):
        return True, "Жильё или аренда — приоритетная тема для аудитории"

    if (
        consumer_service_matches >= 1
        and savings_matches >= 1
        and (summary_len >= 80 or content_len >= 250)
    ):
        return True, "Потребительская услуга + конкретная экономия или сравнение"

    if practical_matches >= 1 and (summary_len >= 80 or content_len >= 250):
        return True, "Есть practical-тема и достаточно содержательный текст"

    russia_keywords = {
        "russia", "russian", "russians",
        "россия", "россияне", "русские", "русский",
    }
    russia_matches = count_matches(combined_lower, russia_keywords)

    if russia_matches >= 1:
        if practical_matches >= 1:
            return True, "Россия/русские + practical-контекст"
        if high_signal_matches >= 2 and (summary_len >= 80 or content_len >= 250):
            return True, "Россия/русские + сильный новостной контекст"

    belgium_keywords = {
        "belgium", "belgian", "brussels", "flanders", "wallonia", "antwerp", "ghent",
        "belgië", "brussel", "vlaanderen", "wallonië", "gent", "antwerpen",
    }
    belgium_matches = count_matches(combined_lower, belgium_keywords)

    if belgium_matches >= 1 and (practical_matches >= 1 or high_signal_matches >= 2):
        return True, "Бельгийский контекст + практическая или сильная тема"

    europe_keywords = {
        "eu", "european union", "europe",
        "netherlands", "france", "germany", "luxembourg",
    }
    europe_matches = count_matches(combined_lower, europe_keywords)

    if europe_matches >= 1 and practical_matches >= 1 and (summary_len >= 80 or content_len >= 250):
        return True, "Европейский контекст с practical-углом"

    entertainment_keywords = {
        "entertainment", "festival", "concert", "cinema", "music",
        "cultural event", "culture", "event", "events",
    }
    entertainment_matches = count_matches(combined_lower, entertainment_keywords)

    if entertainment_matches >= 1:
        if belgium_matches >= 1 and (summary_len >= 80 or content_len >= 250):
            return True, "Развлекательная тема с бельгийским контекстом"
        if high_signal_matches >= 2 and (summary_len >= 100 or content_len >= 300):
            return True, "Развлекательная тема с сильным общественным контекстом"

    if high_signal_matches >= 3 and (summary_len >= 100 or content_len >= 300):
        return True, "Несколько сильных сигналов и содержательный текст"

    if summary_len < 60 and content_len < 180:
        return False, "Слишком мало содержательной информации"

    if pass_matches <= 1 and high_signal_matches <= 1 and practical_matches == 0:
        return False, "Недостаточно сигналов релевантности"

    return False, "Пограничный случай без достаточных оснований для AI"


# -------------------------------------------------------
# Считаем стоимость OpenAI-вызова
# -------------------------------------------------------
def extract_usage_tokens(response: Any) -> tuple[int, int]:
    """
    Пытается достать input/output tokens из ответа OpenAI.
    """
    input_tokens = 0
    output_tokens = 0

    usage = getattr(response, "usage", None)
    if usage:
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    return input_tokens, output_tokens


def calc_cost_usd(input_tokens: int, output_tokens: int) -> Decimal:
    """
    Считает стоимость запроса в долларах.
    """
    input_cost = (Decimal(input_tokens) / Decimal("1000000")) * INPUT_COST_PER_1M
    output_cost = (Decimal(output_tokens) / Decimal("1000000")) * OUTPUT_COST_PER_1M
    return quantize_money(input_cost + output_cost)


# -------------------------------------------------------
# Вызов OpenAI
# -------------------------------------------------------
def analyze_article(
    client: OpenAI,
    article: dict[str, Any],
    editorial_policy_context: str = "",
    event_history: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], int, int, Decimal]:
    """
    Отправляет статью в OpenAI и получает:
    - анализ
    - input tokens
    - output tokens
    - стоимость запроса
    """
    source_name = normalize_text(article.get("source_name"), 200) or "news"
    title = normalize_text(article.get("title"), 1000)
    summary = normalize_text(article.get("summary"), 4000)
    content = normalize_text(article.get("content"), 12000)
    url = normalize_text(article.get("canonical_url") or article.get("original_url"), 1000)

    user_prompt = USER_TEMPLATE.format(
        source_name=source_name,
        title=title,
        summary=summary,
        content=content,
        url=url,
        editorial_policy_context=editorial_policy_context,
        event_history_context=format_event_history(event_history),
    )

    response = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = pick_text(response)
    if not raw:
        raise ValueError("Empty model response")

    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    data = json.loads(raw)

    input_tokens, output_tokens = extract_usage_tokens(response)
    cost_usd = calc_cost_usd(input_tokens, output_tokens)

    public_importance_score = normalize_int(data.get("importance_score", 1))
    practical_value_score = normalize_int(data.get("practical_value_score", 1))
    editorial_score = max(public_importance_score, practical_value_score)

    analysis = {
        "is_relevant": bool(data.get("is_relevant", False)),
        "category": normalize_text(data.get("category", "other"), 100),
        "importance_score": editorial_score,
        "public_importance_score": public_importance_score,
        "practical_value_score": practical_value_score,
        "reason": normalize_text(data.get("reason", ""), 1000),
        "russian_summary": normalize_text(data.get("russian_summary", ""), 4000),
        "telegram_title": normalize_text(data.get("telegram_title", ""), 300),
        "telegram_text": normalize_text(data.get("telegram_text", ""), 4000),
        "is_duplicate_event": data.get("is_duplicate_event", False),
        "duplicate_of_article_id": parse_article_id(
            data.get("duplicate_of_article_id")
        ),
        "duplicate_reason": normalize_text(
            data.get("duplicate_reason", ""),
            1000,
        ),
        "is_material_update": data.get("is_material_update", False),
    }

    return analysis, input_tokens, output_tokens, cost_usd


# -------------------------------------------------------
# Независимая редактура текста перед отправкой редактору
# -------------------------------------------------------
def review_telegram_text(
    client: OpenAI,
    article: dict[str, Any],
    analysis: dict[str, Any],
    editorial_policy_context: str = "",
    max_attempts: int = 2,
) -> tuple[dict[str, Any] | None, int, int, Decimal, int, str]:
    """
    Проверяет и переписывает готовый Telegram-текст на хорошем русском языке.

    Редактор всегда возвращает исправленный текст и не решает, публиковать
    материал или нет. При ошибке формата делает ещё одну попытку.
    """
    source_name = normalize_text(article.get("source_name"), 200) or "news"
    source_title = normalize_text(article.get("title"), 1000)
    source_summary = normalize_text(article.get("summary"), 4000)
    source_content = normalize_text(article.get("content"), 12000)
    url = normalize_text(article.get("canonical_url") or article.get("original_url"), 1000)

    user_prompt = EDITOR_USER_TEMPLATE.format(
        source_name=source_name,
        source_title=source_title,
        source_summary=source_summary,
        source_content=source_content,
        url=url,
        editorial_policy_context=editorial_policy_context,
        draft_category=normalize_text(analysis.get("category"), 100),
        draft_title=normalize_text(analysis.get("telegram_title"), 300),
        draft_text=normalize_text(analysis.get("telegram_text"), 4000),
    )

    total_input_tokens = 0
    total_output_tokens = 0
    calls = 0
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        attempt_prompt = user_prompt
        if attempt > 1:
            attempt_prompt += (
                "\nПредыдущий ответ не прошёл техническую проверку. "
                "Верни непустые поля и строго корректный JSON без Markdown."
            )

        response = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": EDITOR_SYSTEM_PROMPT},
                {"role": "user", "content": attempt_prompt},
            ],
        )

        calls += 1
        input_tokens, output_tokens = extract_usage_tokens(response)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        try:
            raw = pick_text(response)
            if not raw:
                raise ValueError("Empty editorial response")

            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            data = json.loads(raw)
            category = normalize_text(data.get("category"), 100)
            telegram_title = normalize_text(data.get("telegram_title"), 300)
            telegram_text = normalize_text(data.get("telegram_text"), 4000)

            if not telegram_title:
                raise ValueError("Editorial response has empty telegram_title")
            if not telegram_text:
                raise ValueError("Editorial response has empty telegram_text")

            validate_publication_length(telegram_title, telegram_text)

            if category not in ALLOWED_CATEGORIES:
                category = normalize_text(analysis.get("category"), 100)
            if category not in ALLOWED_CATEGORIES:
                category = "other"

            reviewed_analysis = dict(analysis)
            reviewed_analysis.update({
                "category": category,
                "telegram_title": telegram_title,
                "telegram_text": telegram_text,
            })

            total_cost_usd = calc_cost_usd(total_input_tokens, total_output_tokens)
            print(f"Editorial review completed in {attempt} attempt(s)")
            return (
                reviewed_analysis,
                total_input_tokens,
                total_output_tokens,
                total_cost_usd,
                calls,
                "",
            )

        except (ValueError, TypeError, json.JSONDecodeError) as e:
            last_error = repr(e)
            print(f"WARNING: editorial review attempt {attempt} failed: {last_error}")

    total_cost_usd = calc_cost_usd(total_input_tokens, total_output_tokens)
    return (
        None,
        total_input_tokens,
        total_output_tokens,
        total_cost_usd,
        calls,
        last_error or "Unknown editorial review error",
    )


# -------------------------------------------------------
# Обратная связь редактора
# -------------------------------------------------------
def load_editorial_policy_context(sb) -> str:
    """Загружает недавние решения редактора для тематической калибровки AI."""
    rows = (
        sb.table("editorial_feedback")
        .select(
            "feedback_type,status,editor_comment,source_title,source_summary,"
            "draft_title,draft_text,revised_title,revised_text,created_at"
        )
        .eq("status", "applied")
        .in_("feedback_type", ["approved", "topic_mismatch", "text_correction"])
        .order("created_at", desc=True)
        .limit(40)
        .execute()
    ).data or []

    context = build_editorial_policy_context(rows)
    negative_count = sum(
        1 for row in rows if row.get("feedback_type") == "topic_mismatch"
    )
    positive_count = sum(
        1 for row in rows if row.get("feedback_type") == "approved"
    )
    correction_count = sum(
        1 for row in rows if row.get("feedback_type") == "text_correction"
    )
    print(
        "Loaded editorial policy context: "
        f"topic_rejections={negative_count} approvals={positive_count} "
        f"text_corrections={correction_count} "
        f"chars={len(context)}"
    )
    return context


EVENT_COVERAGE_STATUSES = [
    # Pending/sent rows prevent two copies of the same event from appearing in
    # one editor session. Approved/published rows are the durable coverage
    # memory. Rejected rows are intentionally excluded: an editor may reject a
    # story for style or another reason without saying the event was covered.
    "pending",
    "notifying",
    "sent",
    "awaiting_feedback",
    "correction_pending",
    "publishing",
    "approved",
    "published",
]


def load_recent_event_history(
    sb,
    *,
    limit: int = EVENT_HISTORY_LIMIT,
    lookback_days: int = EVENT_HISTORY_LOOKBACK_DAYS,
) -> list[dict[str, Any]]:
    """Load compact cross-source coverage context for the AI prompt.

    Queue status is the source of truth for whether an article has already been
    proposed or approved. We deliberately fetch only a bounded recent window;
    this keeps prompt size and OpenAI cost predictable while covering the period
    in which duplicate wire stories normally arrive.
    """
    since = (
        datetime.now(timezone.utc) - timedelta(days=lookback_days)
    ).isoformat()
    queue_rows = (
        sb.table("editor_queue")
        .select("article_id,status,created_at")
        .in_("status", EVENT_COVERAGE_STATUSES)
        .gte("created_at", since)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    ).data or []

    article_ids = [
        article_id
        for article_id in (
            parse_article_id(row.get("article_id")) for row in queue_rows
        )
        if article_id is not None
    ]
    if not article_ids:
        return []

    article_rows = (
        sb.table("articles")
        .select("id,source_id,title,summary,published_at,canonical_url,original_url")
        .in_("id", article_ids)
        .execute()
    ).data or []
    analysis_rows = (
        sb.table("article_analysis")
        .select("article_id,russian_summary,telegram_title,telegram_text,category")
        .in_("article_id", article_ids)
        .execute()
    ).data or []

    source_ids = {
        int(row["source_id"])
        for row in article_rows
        if row.get("source_id") is not None
    }
    source_rows = []
    if source_ids:
        source_rows = (
            sb.table("sources")
            .select("id,name")
            .in_("id", list(source_ids))
            .execute()
        ).data or []

    article_by_id = {
        int(row["id"]): row
        for row in article_rows
        if row.get("id") is not None
    }
    analysis_by_article_id = {
        int(row["article_id"]): row
        for row in analysis_rows
        if row.get("article_id") is not None
    }
    source_by_id = {
        int(row["id"]): row.get("name") or f"source-{row['id']}"
        for row in source_rows
        if row.get("id") is not None
    }

    history: list[dict[str, Any]] = []
    for queue_row in queue_rows:
        article_id = parse_article_id(queue_row.get("article_id"))
        if article_id is None:
            continue
        article_row = article_by_id.get(article_id)
        if not article_row:
            continue
        analysis_row = analysis_by_article_id.get(article_id) or {}
        source_id = article_row.get("source_id")
        history.append({
            "article_id": article_id,
            "status": queue_row.get("status") or "unknown",
            "source_name": source_by_id.get(
                int(source_id), f"source-{source_id}"
            ) if source_id is not None else "news",
            "title": article_row.get("title") or "",
            "summary": article_row.get("summary") or "",
            "russian_summary": analysis_row.get("russian_summary") or "",
            "telegram_title": analysis_row.get("telegram_title") or "",
            "telegram_text": analysis_row.get("telegram_text") or "",
            "published_at": article_row.get("published_at") or "",
        })
    return history[:limit]


def revise_telegram_text_from_feedback(
    client: OpenAI,
    article: dict[str, Any],
    analysis: dict[str, Any],
    editor_comment: str,
    max_attempts: int = 2,
) -> tuple[dict[str, Any] | None, int, int, Decimal, int, str]:
    """Исправляет уже показанный черновик по конкретному комментарию редактора."""
    user_prompt = build_correction_prompt(article, analysis, editor_comment)
    total_input_tokens = 0
    total_output_tokens = 0
    calls = 0
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        attempt_prompt = user_prompt
        if attempt > 1:
            attempt_prompt += (
                "\nПредыдущий ответ не прошёл техническую проверку. "
                "Верни непустые поля и строго корректный JSON без Markdown."
            )

        response = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
                {"role": "user", "content": attempt_prompt},
            ],
        )
        calls += 1
        input_tokens, output_tokens = extract_usage_tokens(response)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        try:
            title, text = parse_correction_response(pick_text(response))
            revised_analysis = dict(analysis)
            revised_analysis.update({
                "telegram_title": title,
                "telegram_text": text,
            })
            return (
                revised_analysis,
                total_input_tokens,
                total_output_tokens,
                calc_cost_usd(total_input_tokens, total_output_tokens),
                calls,
                "",
            )
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            last_error = repr(error)
            print(
                f"WARNING: feedback correction attempt {attempt} failed: {last_error}"
            )

    return (
        None,
        total_input_tokens,
        total_output_tokens,
        calc_cost_usd(total_input_tokens, total_output_tokens),
        calls,
        last_error or "Unknown feedback correction error",
    )


def process_pending_corrections(
    sb,
    client: OpenAI,
    max_items: int = 5,
    feedback_id: int | None = None,
) -> dict[str, Any]:
    """Обрабатывает correction requests, optionally limited to one feedback id.

    The dedicated editor-correction workflow uses feedback_id so that one
    Telegram comment is handled immediately without re-running collection and
    analysis. The regular analyzer continues to process the queue in batches
    as a recovery path.
    """
    stats: dict[str, Any] = {
        "processed": 0,
        "failed": 0,
        "ai_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": Decimal("0"),
        "quota_error": False,
        "processed_items": [],
    }

    # Возвращаем в очередь запрос, который остался processing после аварийного
    # завершения предыдущего runner. Сравнение по времени не затрагивает живой вызов.
    stale_before = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    stale_query = sb.table("editorial_feedback").update({
        "status": "pending_processing",
        "processing_started_at": None,
        "updated_at": now_iso(),
        "error": "Recovered after stale processing lease",
    }).eq("feedback_type", "text_correction").eq("status", "processing").lt(
        "processing_started_at", stale_before
    ).lt("attempts", 3)
    if feedback_id is not None:
        stale_query = stale_query.eq("id", feedback_id)
    stale_query.execute()

    rows_query = (
        sb.table("editorial_feedback")
        .select("*")
        .eq("feedback_type", "text_correction")
        .eq("status", "pending_processing")
        .lt("attempts", 3)
        .order("created_at", desc=False)
        .limit(max_items)
    )
    if feedback_id is not None:
        rows_query = rows_query.eq("id", feedback_id)
    rows = (rows_query.execute()).data or []
    print(f"Loaded pending editor corrections: {len(rows)}")

    for pending_row in rows:
        current_feedback_id = int(pending_row["id"])
        claim_result = sb.rpc(
            "claim_editorial_correction",
            {"p_feedback_id": current_feedback_id},
        ).execute()
        claimed_rows = claim_result.data or []
        if not claimed_rows:
            print(f"Correction {current_feedback_id} was claimed by another runner")
            continue

        feedback = claimed_rows[0]
        attempts = int(feedback.get("attempts") or 1)
        queue_id = int(feedback["queue_id"])
        article_id = int(feedback["article_id"])
        expected_revision = int(feedback["queue_revision"])

        try:
            queue_rows = (
                sb.table("editor_queue")
                .select("id,article_id,status,revision")
                .eq("id", queue_id)
                .eq("article_id", article_id)
                .eq("status", "correction_pending")
                .eq("revision", expected_revision)
                .limit(1)
                .execute()
            ).data or []
            article_rows = (
                sb.table("articles")
                .select("*")
                .eq("id", article_id)
                .limit(1)
                .execute()
            ).data or []
            analysis_rows = (
                sb.table("article_analysis")
                .select("*")
                .eq("article_id", article_id)
                .limit(1)
                .execute()
            ).data or []

            if not queue_rows or not article_rows or not analysis_rows:
                raise RuntimeError("Correction references missing or stale queue data")

            source_row = article_rows[0]
            article = {
                "source_name": "news",
                "title": source_row.get("title"),
                "summary": source_row.get("summary"),
                "content": source_row.get("content"),
                "canonical_url": source_row.get("canonical_url"),
                "original_url": source_row.get("original_url"),
            }
            analysis = analysis_rows[0]

            (
                revised,
                input_tokens,
                output_tokens,
                cost_usd,
                calls,
                correction_error,
            ) = revise_telegram_text_from_feedback(
                client,
                article,
                analysis,
                normalize_text(feedback.get("editor_comment"), 2000),
            )

            stats["ai_calls"] += calls
            stats["input_tokens"] += input_tokens
            stats["output_tokens"] += output_tokens
            stats["cost_usd"] = quantize_money(stats["cost_usd"] + cost_usd)

            if revised is None:
                raise ValueError(correction_error)

            apply_result = sb.rpc(
                "apply_editorial_correction",
                {
                    "p_feedback_id": current_feedback_id,
                    "p_expected_revision": expected_revision,
                    "p_title": revised["telegram_title"],
                    "p_text": revised["telegram_text"],
                    "p_model": MODEL,
                },
            ).execute()
            if apply_result.data is None:
                raise RuntimeError("Correction transaction returned no revision")

            new_revision = int(apply_result.data)
            stats["processed"] += 1
            stats["processed_items"].append({
                "feedback_id": current_feedback_id,
                "queue_id": queue_id,
                "article_id": article_id,
                "revision": new_revision,
            })
            print(
                f"Applied editor correction feedback_id={current_feedback_id} "
                f"queue_id={queue_id} new_revision={new_revision}"
            )

        except Exception as error:
            error_text = repr(error)[:4000]
            print(f"ERROR correction feedback_id={current_feedback_id}: {error_text}")

            if "insufficient_quota" in error_text or "RateLimitError" in error_text:
                sb.table("editorial_feedback").update({
                    "status": "pending_processing",
                    "attempts": max(0, attempts - 1),
                    "processing_started_at": None,
                    "updated_at": now_iso(),
                    "error": error_text,
                }).eq("id", current_feedback_id).eq("status", "processing").execute()
                stats["quota_error"] = True
                break

            next_status = "pending_processing" if attempts < 3 else "failed"
            sb.table("editorial_feedback").update({
                "status": next_status,
                "processing_started_at": None,
                "updated_at": now_iso(),
                "error": error_text,
            }).eq("id", current_feedback_id).eq("status", "processing").execute()

            if next_status == "failed":
                sb.table("editor_queue").update({
                    "status": "correction_failed",
                }).eq("id", queue_id).eq("status", "correction_pending").eq(
                    "revision", expected_revision
                ).execute()
            stats["failed"] += 1

    return stats


# -------------------------------------------------------
# Добавление статьи в очередь редактора
# -------------------------------------------------------
def add_to_editor_queue(sb, article_id: int) -> bool:
    """
    Добавляет статью в editor_queue, если её там ещё нет.
    """
    existing = (
        sb.table("editor_queue")
        .select("id")
        .eq("article_id", article_id)
        .limit(1)
        .execute()
    ).data

    if existing:
        print(f"Queue skip for article_id={article_id}: already in editor_queue")
        return False

    sb.table("editor_queue").insert({
        "article_id": article_id,
        "status": "pending",
    }).execute()

    print(f"Added article_id={article_id} to editor_queue")
    return True


# -------------------------------------------------------
# Быстрая запись нерелевантной статьи без OpenAI
# -------------------------------------------------------
def save_prefilter_rejection(sb, article_id: int, reason: str) -> None:
    """
    Сохраняет результат pre-filter как нерелевантную статью без вызова AI.
    """
    sb.table("article_analysis").insert({
        "article_id": article_id,
        "is_relevant": False,
        "category": "other",
        "importance_score": 1,
        "reason": f"Pre-filter: {reason}",
        "russian_summary": "",
        "telegram_title": "",
        "telegram_text": "",
    }).execute()

    print(f"Saved pre-filter rejection for article_id={article_id}: {reason}")


# -------------------------------------------------------
# Работа с ai_runs / ai_balance
# -------------------------------------------------------
def create_ai_run(sb, run_type: str = "analyze") -> int:
    """
    Создаёт запись нового прогона и возвращает run_id.
    """
    result = sb.table("ai_runs").insert({
        "run_type": run_type,
        "started_at": now_iso(),
    }).execute()

    row = result.data[0]
    return row["id"]


def finish_ai_run(
    sb,
    run_id: int,
    processed: int,
    skipped: int,
    queued: int,
    prefilter_rejected: int,
    ai_calls: int,
    input_tokens: int,
    output_tokens: int,
    cost_usd: Decimal,
    corrections_processed: int = 0,
    corrections_failed: int = 0,
) -> tuple[Decimal, Decimal, Decimal]:
    """
    Завершает прогон, пишет статистику и обновляет ai_balance.

    Возвращает:
    - starting_balance_usd
    - spent_total_usd
    - remaining_estimated_usd
    """
    sb.table("ai_runs").update({
        "finished_at": now_iso(),
        "processed_count": processed,
        "skipped_count": skipped,
        "queued_count": queued,
        "prefilter_rejected_count": prefilter_rejected,
        "ai_calls_count": ai_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": str(cost_usd),
        "corrections_processed_count": corrections_processed,
        "corrections_failed_count": corrections_failed,
    }).eq("id", run_id).execute()

    balance_rows = (
        sb.table("ai_balance")
        .select("*")
        .eq("id", 1)
        .limit(1)
        .execute()
    ).data or []

    if not balance_rows:
        raise RuntimeError("ai_balance row with id=1 not found")

    balance = balance_rows[0]
    starting_balance = Decimal(str(balance["starting_balance_usd"]))
    spent_total_old = Decimal(str(balance["spent_total_usd"]))
    spent_total_new = quantize_money(spent_total_old + cost_usd)
    remaining = quantize_money(starting_balance - spent_total_new)

    sb.table("ai_balance").update({
        "spent_total_usd": str(spent_total_new),
        "updated_at": now_iso(),
    }).eq("id", 1).execute()

    return starting_balance, spent_total_new, remaining


def send_run_balance_message(
    bot_token: str,
    chat_id: str,
    run_cost_usd: Decimal,
    spent_total_usd: Decimal,
    remaining_estimated_usd: Decimal,
    ai_calls: int,
    prefilter_rejected: int,
    corrections_processed: int = 0,
    corrections_failed: int = 0,
) -> None:
    """
    Отправляет Telegram-сообщение после прогона analyzer.
    """
    text = (
        "📊 OpenAI balance update\n\n"
        f"Last run cost: ${run_cost_usd}\n"
        f"Spent total: ${spent_total_usd}\n"
        f"Estimated remaining: ${remaining_estimated_usd}\n\n"
        f"AI calls this run: {ai_calls}\n"
        f"Pre-filter rejected: {prefilter_rejected}\n"
        f"Editor corrections applied: {corrections_processed}\n"
        f"Editor corrections failed/retrying: {corrections_failed}"
    )

    telegram_send_message(bot_token, chat_id, text)


def load_prefilter_state(sb) -> tuple[bool, dict[str, Any] | None, dict[str, Any]]:
    """Load exploration flag and the last editor-approved policy."""
    settings_rows = (
        sb.table("prefilter_settings").select("*").eq("id", 1).limit(1).execute()
    ).data or []
    if not settings_rows:
        return False, None, {}
    settings = settings_rows[0]
    policy = None
    policy_id = settings.get("current_policy_id")
    if policy_id:
        rows = (
            sb.table("prefilter_policy_proposals")
            .select("policy")
            .eq("id", policy_id)
            .eq("status", "active")
            .limit(1)
            .execute()
        ).data or []
        if rows:
            policy = rows[0].get("policy")
    return bool(settings.get("learning_mode")), policy, settings


def maybe_create_prefilter_proposal(
    sb,
    client: OpenAI,
    settings: dict[str, Any],
    bot_token: str,
    chat_id: str,
) -> tuple[int, int, Decimal]:
    """Create and announce one safe proposal after enough labelled decisions."""
    if not settings:
        return 0, 0, Decimal("0")
    pending = (
        sb.table("prefilter_policy_proposals")
        .select("id")
        .eq("status", "pending")
        .limit(1)
        .execute()
    ).data or []
    if pending:
        return 0, 0, Decimal("0")

    rows = (
        sb.table("editorial_feedback")
        .select(
            "feedback_type,status,editor_comment,source_title,source_summary,"
            "draft_title,draft_text,created_at"
        )
        .eq("status", "applied")
        .in_("feedback_type", ["approved", "topic_mismatch"])
        .order("created_at", desc=True)
        .limit(200)
        .execute()
    ).data or []
    approvals = sum(row.get("feedback_type") == "approved" for row in rows)
    declines = sum(row.get("feedback_type") == "topic_mismatch" for row in rows)
    decisions = approvals + declines
    minimum = int(settings.get("minimum_training_decisions") or 50)
    previous = int(settings.get("decisions_at_last_proposal") or 0)
    if decisions - previous < minimum or approvals < 10 or declines < 10:
        return 0, 0, Decimal("0")

    response = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": PROPOSAL_SYSTEM_PROMPT},
            {"role": "user", "content": format_training_examples(rows)},
        ],
    )
    input_tokens, output_tokens = extract_usage_tokens(response)
    cost = calc_cost_usd(input_tokens, output_tokens)
    policy = parse_policy_proposal(pick_text(response))
    metrics = evaluate_policy(rows, policy)
    if not proposal_is_safe(metrics):
        print(f"Prefilter proposal rejected by replay: {metrics}")
        return input_tokens, output_tokens, cost

    inserted = sb.table("prefilter_policy_proposals").insert({
        "summary": policy["summary"],
        "rationale": policy["rationale"],
        "policy": {
            "positive_terms": policy["positive_terms"],
            "negative_terms": policy["negative_terms"],
        },
        "metrics": metrics,
        "training_decisions": decisions,
        "training_approvals": approvals,
        "training_topic_declines": declines,
        "model": MODEL,
    }).select("id").single().execute().data
    proposal_id = int(inserted["id"])

    retention = round(float(metrics["approval_retention"]) * 100)
    rejected = round(float(metrics["decline_rejection"]) * 100)
    response_message = requests.post(
        f"{TELEGRAM_API_BASE}/bot{bot_token}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": (
                "🧠 Новое предложение для prefilter\n\n"
                f"{policy['summary']}\n\n"
                f"Решений: {decisions} (✅ {approvals}, ❌ {declines})\n"
                f"Сохранено прошлых approvals: {retention}%\n"
                f"Отсекается прошлых topic declines: {rejected}%\n\n"
                "Правила не изменятся без вашего подтверждения."
            ),
            "reply_markup": {"inline_keyboard": [
                [{"text": "🔎 Детали", "callback_data": f"pf:details:{proposal_id}"}],
                [
                    {"text": "✅ Активировать", "callback_data": f"pf:activate:{proposal_id}"},
                    {"text": "❌ Отклонить", "callback_data": f"pf:reject:{proposal_id}"},
                ],
            ]},
        },
        timeout=30,
    )
    response_message.raise_for_status()
    message = response_message.json().get("result", {})
    sb.table("prefilter_policy_proposals").update({
        "telegram_chat_id": int(message.get("chat", {}).get("id", chat_id)),
        "telegram_message_id": int(message["message_id"]),
    }).eq("id", proposal_id).execute()
    print(f"Sent prefilter proposal id={proposal_id}")
    return input_tokens, output_tokens, cost


# -------------------------------------------------------
# Основная функция
# -------------------------------------------------------
def main():
    """
    Главная логика:
    - берёт статьи
    - не анализирует уже обработанные повторно
    - применяет pre-filter
    - вызывает OpenAI только для кандидатов
    - добавляет релевантные статьи в editor_queue
    - считает стоимость OpenAI и отправляет баланс в Telegram
    """
    print("Starting analyzer...")

    sb = get_supabase()
    oa = get_openai()

    run_id = create_ai_run(sb)
    editorial_policy_context = load_editorial_policy_context(sb)
    learning_mode, learned_policy, prefilter_settings = load_prefilter_state(sb)
    print(
        f"Prefilter state: learning_mode={learning_mode} "
        f"active_policy={bool(learned_policy)}"
    )
    correction_stats = process_pending_corrections(sb, oa)

    result = (
        sb.table("articles")
        .select("*")
        .order("id", desc=True)
        .limit(20)
        .execute()
    )

    rows = result.data or []
    print(f"Loaded articles: {len(rows)}")

    try:
        event_history = load_recent_event_history(sb)
    except Exception as error:
        # Deduplication must never stop collection/analysis.  A missing history
        # is fail-open for this run and is visible in logs for investigation.
        print(f"WARNING: event history unavailable; dedup skipped: {repr(error)}")
        event_history = []
    print(f"Loaded event coverage history: {len(event_history)} item(s)")

    try:
        source_rows = (
            sb.table("sources")
            .select("id,name")
            .execute()
        ).data or []
        source_names = {
            int(source["id"]): source.get("name") or f"source-{source['id']}"
            for source in source_rows
            if source.get("id") is not None
        }
    except Exception as error:
        print(f"WARNING: source names unavailable: {repr(error)}")
        source_names = {}

    processed = 0
    skipped = 0
    queued = 0
    prefilter_rejected = 0
    ai_calls = int(correction_stats["ai_calls"])
    total_input_tokens = int(correction_stats["input_tokens"])
    total_output_tokens = int(correction_stats["output_tokens"])
    total_cost_usd = Decimal(correction_stats["cost_usd"])
    corrections_processed = int(correction_stats["processed"])
    corrections_failed = int(correction_stats["failed"])
    errors = 0
    quota_error = bool(correction_stats["quota_error"])

    for row in rows:
        if quota_error:
            break

        article_id = row["id"]
        print(f"Processing article_id={article_id}")

        existing = (
            sb.table("article_analysis")
            .select("id,is_relevant,importance_score")
            .eq("article_id", article_id)
            .limit(1)
            .execute()
        ).data

        if existing:
            print(f"Skip article_id={article_id}: already analyzed")
            skipped += 1

            existing_row = existing[0]
            if existing_row.get("is_relevant") and (existing_row.get("importance_score") or 0) >= 6:
                if add_to_editor_queue(sb, article_id):
                    queued += 1
                    event_history.insert(0, make_event_history_entry(
                        article_id,
                        {
                            "source_name": source_names.get(
                                int(row.get("source_id")), "news"
                            ) if row.get("source_id") is not None else "news",
                            "title": row.get("title"),
                            "summary": row.get("summary"),
                            "published_at": row.get("published_at"),
                        },
                        existing_row,
                    ))
                    del event_history[EVENT_HISTORY_LIMIT:]

            continue

        article = {
            "source_name": source_names.get(
                int(row.get("source_id")), "news"
            ) if row.get("source_id") is not None else "news",
            "title": row.get("title"),
            "summary": row.get("summary"),
            "content": row.get("content"),
            "canonical_url": row.get("canonical_url"),
            "original_url": row.get("original_url"),
            "published_at": row.get("published_at"),
        }

        try:
            send_to_ai, prefilter_reason = should_send_to_ai(
                article,
                learning_mode=learning_mode,
                learned_policy=learned_policy,
            )

            if not send_to_ai:
                save_prefilter_rejection(sb, article_id, prefilter_reason)
                prefilter_rejected += 1
                processed += 1
                continue

            print(f"Sending article_id={article_id} to AI: {prefilter_reason}")

            analysis, input_tokens, output_tokens, cost_usd = analyze_article(
                oa,
                article,
                editorial_policy_context,
                event_history,
            )

            ai_calls += 1
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_cost_usd = quantize_money(total_cost_usd + cost_usd)

            history_ids = {
                article_id_from_history
                for article_id_from_history in (
                    parse_article_id(history_row.get("article_id"))
                    for history_row in event_history
                )
                if article_id_from_history is not None
            }
            analysis, duplicate_blocked, dedup_note = reconcile_duplicate_decision(
                analysis,
                history_ids,
            )
            if dedup_note:
                print(
                    f"article_id={article_id} duplicate decision note: {dedup_note}"
                )
            if duplicate_blocked:
                analysis["is_relevant"] = False
                analysis["reason"] = (
                    "Повтор уже покрытого события: "
                    f"{analysis.get('duplicate_reason') or 'без существенного обновления'}"
                )
                print(
                    f"Blocked cross-source duplicate article_id={article_id} "
                    f"of article_id={analysis['duplicate_of_article_id']}"
                )

            # Только кандидаты для редакторского чата проходят обязательную
            # независимую редактуру. Сырой AI-текст в очередь не попадает.
            if (
                not duplicate_blocked
                and analysis["is_relevant"]
                and analysis["importance_score"] >= 6
            ):
                print(f"Reviewing Telegram text for article_id={article_id}")
                (
                    reviewed_analysis,
                    review_input_tokens,
                    review_output_tokens,
                    review_cost_usd,
                    review_calls,
                    review_error,
                ) = review_telegram_text(
                    oa, article, analysis, editorial_policy_context
                )

                ai_calls += review_calls
                total_input_tokens += review_input_tokens
                total_output_tokens += review_output_tokens
                total_cost_usd = quantize_money(total_cost_usd + review_cost_usd)

                if reviewed_analysis is None:
                    raise ValueError(
                        f"Editorial review failed for article_id={article_id}: {review_error}"
                    )

                analysis = reviewed_analysis

            sb.table("article_analysis").insert({
                "article_id": article_id,
                "is_relevant": analysis["is_relevant"],
                "category": analysis["category"],
                "importance_score": analysis["importance_score"],
                "reason": analysis["reason"],
                "russian_summary": analysis["russian_summary"],
                "telegram_title": analysis["telegram_title"],
                "telegram_text": analysis["telegram_text"],
                "is_duplicate_event": analysis["is_duplicate_event"],
                "duplicate_of_article_id": analysis["duplicate_of_article_id"],
                "duplicate_reason": analysis["duplicate_reason"],
                "is_material_update": analysis["is_material_update"],
            }).execute()

            processed += 1

            print(
                f"Saved analysis for article_id={article_id}: "
                f"relevant={analysis['is_relevant']}, "
                f"importance={analysis['importance_score']}, "
                f"public_importance={analysis.get('public_importance_score', 1)}, "
                f"practical_value={analysis.get('practical_value_score', 1)}, "
                f"category={analysis['category']}"
            )

            if analysis["is_relevant"] and analysis["importance_score"] >= 6:
                if add_to_editor_queue(sb, article_id):
                    queued += 1
                    event_history.insert(0, make_event_history_entry(
                        article_id,
                        article,
                        analysis,
                    ))
                    del event_history[EVENT_HISTORY_LIMIT:]

        except Exception as e:
            error_text = repr(e)
            print(f"ERROR article_id={article_id}: {error_text}")
            errors += 1

            if "insufficient_quota" in error_text or "RateLimitError" in error_text:
                print("Stopping because OpenAI API quota/billing is not available.")
                quota_error = True
                break

    bot_token = get_env("TELEGRAM_BOT_TOKEN", required=False)
    chat_id = get_env("TELEGRAM_CHAT_ID", required=False)
    if bot_token and chat_id and not quota_error:
        try:
            proposal_input, proposal_output, proposal_cost = (
                maybe_create_prefilter_proposal(
                    sb, oa, prefilter_settings, bot_token, chat_id
                )
            )
            if proposal_input or proposal_output:
                ai_calls += 1
                total_input_tokens += proposal_input
                total_output_tokens += proposal_output
                total_cost_usd = quantize_money(total_cost_usd + proposal_cost)
        except Exception as e:
            print(f"WARNING: prefilter proposal failed safely: {repr(e)}")

    starting_balance, spent_total_usd, remaining_estimated_usd = finish_ai_run(
        sb=sb,
        run_id=run_id,
        processed=processed,
        skipped=skipped,
        queued=queued,
        prefilter_rejected=prefilter_rejected,
        ai_calls=ai_calls,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        cost_usd=total_cost_usd,
        corrections_processed=corrections_processed,
        corrections_failed=corrections_failed,
    )

    print(
        f"Done. processed={processed} skipped={skipped} "
        f"queued={queued} prefilter_rejected={prefilter_rejected} "
        f"corrections_processed={corrections_processed} "
        f"corrections_failed={corrections_failed} "
        f"ai_calls={ai_calls} cost_usd={total_cost_usd} errors={errors}"
    )

    # Telegram-уведомление о балансе после каждого прогона
    if bot_token and chat_id:
        try:
            send_run_balance_message(
                bot_token=bot_token,
                chat_id=chat_id,
                run_cost_usd=total_cost_usd,
                spent_total_usd=spent_total_usd,
                remaining_estimated_usd=remaining_estimated_usd,
                ai_calls=ai_calls,
                prefilter_rejected=prefilter_rejected,
                corrections_processed=corrections_processed,
                corrections_failed=corrections_failed,
            )
        except Exception as e:
            print(f"WARNING: failed to send Telegram balance message: {repr(e)}")

    if quota_error:
        print("Analyzer stopped due to missing API quota. Add billing in platform.openai.com.")
        return


# -------------------------------------------------------
# Точка входа
# -------------------------------------------------------
if __name__ == "__main__":
    main()
