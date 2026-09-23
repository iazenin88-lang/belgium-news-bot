import unittest

from sports_filter import is_routine_sports_coverage


class RoutineSportsCoverageTests(unittest.TestCase):
    def test_blocks_the_missed_dutch_cycling_medal_article(self):
        title = (
            "Met dank aan historisch goud Remco Evenepoel en sterke jongeren: "
            "België prijkt na tijdritten bovenaan in WK-medaillestand"
        )
        summary = (
            "Het WK wielrennen in Montréal is nog in volle gang. België prijkt "
            "bovenaan de voorlopige medaillespiegel na tijdritgoud en prestaties "
            "van junioren en beloften."
        )
        self.assertTrue(is_routine_sports_coverage(title, summary))

    def test_blocks_french_and_english_match_results(self):
        self.assertTrue(is_routine_sports_coverage(
            "Cyclisme: résultats et médailles du championnat",
            "Classement du championnat du monde.",
        ))
        self.assertTrue(is_routine_sports_coverage(
            "Cycling championship results",
            "The team won the final and leads the standings.",
        ))

    def test_blocks_russian_sports_results(self):
        self.assertTrue(is_routine_sports_coverage(
            "Велогонщик выиграл чемпионат мира",
            "Результаты гонки и медали сборной.",
        ))

    def test_keeps_safety_or_rule_change_stories_for_ai(self):
        self.assertFalse(is_routine_sports_coverage(
            "Voetbalwedstrijd leidt tot nieuwe veiligheidsregels",
            "De voetbalbond verandert de regelgeving na een ernstig ongeval.",
        ))
        self.assertFalse(is_routine_sports_coverage(
            "Cycling law changes",
            "New rules affect safety and transport around races.",
        ))

    def test_does_not_reject_generic_non_result_mention(self):
        self.assertFalse(is_routine_sports_coverage(
            "Belgian sports clubs face new funding rules",
            "The law changes how local clubs receive public funding.",
        ))


if __name__ == "__main__":
    unittest.main()
