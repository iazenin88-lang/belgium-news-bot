"""Detect routine sports coverage before the editorial AI call.

The feeds are multilingual. This deliberately looks at the headline and summary,
where the article's subject is stated, instead of names buried in full text.
Items about concrete public consequences (law, safety, health, transport, etc.)
are left for the main relevance review.
"""

SPORTS_SUBJECT_TERMS = (
    # Dutch
    "wielrennen", "wielrenner", "wielrenners", "wielerwedstrijd", "wielerploeg",
    "voetbal", "voetballer", "wieler-", "sportwedstrijd", "sportnieuws",
    # French
    "cyclisme", "cycliste", "course cycliste", "football", "footballeur",
    "match de", "sportif", "sportive",
    # English
    "cycling", "cyclist", "football", "soccer", "basketball", "tennis",
    "sports team", "athlete", "world cup", "champions league",
    # Russian
    "велоспорт", "велогон", "велосипедист", "футбол", "футболист",
    "баскетбол", "теннис", "спортивный матч", "спортивная команда",
)

SPORTS_RESULT_TERMS = (
    # Dutch
    "medaillestand", "medaillespiegel", "medaille", "tijdrit", "tijdritten",
    "uitslag", "uitslagen", "wereldkampioen", "kampioen", "kampioenschap",
    "wedstrijd", "wedstrijden", "score", "gewonnen", "goud", "zilver", "brons",
    "doelpunt", "transfer",
    # French
    "médaille", "médailles", "classement", "contre-la-montre", "résultat",
    "résultats", "champion du monde", "championnat", "victoire", "score",
    "but", "transfert",
    # English
    "medal", "medals", "standings", "time trial", "results", "champion",
    "championship", "match report", "match preview", "score", "won", "wins",
    "victory", "goal", "transfer",
    # Russian
    "медал", "чемпион", "чемпионат", "результат", "матч", "счёт", "счет",
    "побед", "золото", "серебро", "бронза", "трансфер",
)

PUBLIC_CONSEQUENCE_TERMS = (
    # Dutch
    "wet", "wetgeving", "regelgeving", "verbod", "veiligheid", "ongeval",
    "gewond", "gezondheid", "vervoer", "staking", "fraude", "discriminatie",
    # French
    "loi", "législation", "réglementation", "interdiction", "sécurité",
    "accident", "blessé", "santé", "transport", "grève", "fraude",
    # English
    "law", "legislation", "regulation", "ban", "safety", "accident", "injured",
    "health", "transport", "strike", "fraud", "discrimination", "systemic",
    # Russian
    "закон", "законодательств", "правил", "запрет", "безопасност",
    "авари", "травм", "здоров", "транспорт", "забастов", "мошеннич",
    "дискриминац", "системн",
)


def _has_term(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(term in lowered for term in terms)


def is_routine_sports_coverage(title: str, summary: str) -> bool:
    """Return True for sports results/reporting, but defer public-interest stories."""
    headline = f"{title or ''}\n{summary or ''}".strip()
    if not headline or not _has_term(headline, SPORTS_SUBJECT_TERMS):
        return False
    if _has_term(headline, PUBLIC_CONSEQUENCE_TERMS):
        return False
    return _has_term(headline, SPORTS_RESULT_TERMS)
