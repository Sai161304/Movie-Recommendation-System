"""Unit tests that do not need an external dataset."""

import unittest

from src.text_cleaner import clean_text, drop_duplicate_movies, genre_list, parse_name_list
import pandas as pd


class TextCleanerTests(unittest.TestCase):
    def test_clean_text_lowers_and_strips_punctuation(self):
        self.assertEqual(clean_text("  The Dark Knight!!! "), "the dark knight")

    def test_missing_values_become_empty(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(float("nan")), "")

    def test_movielens_genres(self):
        parsed = parse_name_list("Action|Adventure|Sci-Fi")
        self.assertIn("action", parsed)
        self.assertIn("sci fi", parsed)

    def test_tmdb_json_genres(self):
        raw = '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
        parsed = parse_name_list(raw)
        self.assertEqual(genre_list(parsed), ["action", "adventure"])

    def test_duplicate_titles_keep_higher_vote_count(self):
        table = pd.DataFrame(
            {
                "title": ["Dune", "dune ", "Arrival"],
                "vote_count": [10, 50, 3],
                "genres": ["a", "b", "c"],
            }
        )
        cleaned = drop_duplicate_movies(table)
        self.assertEqual(len(cleaned), 2)
        dune_row = cleaned[cleaned["title"].str.lower().str.strip() == "dune"].iloc[0]
        self.assertEqual(int(dune_row["vote_count"]), 50)


if __name__ == "__main__":
    unittest.main()
