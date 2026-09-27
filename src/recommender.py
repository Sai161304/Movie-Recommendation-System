"""
Content-based movie recommender.

Idea in one sentence:
  Movies that share similar words in genres, plot, keywords, and other
  metadata are treated as similar.

How the math works (interview version):
1) Build one text blob per movie ("content soup").
2) TF-IDF turns that text into numbers.
   - TF = how often a word appears in this movie
   - IDF = how rare that word is across all movies
3) Cosine similarity measures the angle between two movie vectors.
   1.0 = same direction (very similar), 0.0 = unrelated.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.text_cleaner import clean_text, genre_list


@dataclass
class SearchResult:
    found: bool
    title: str
    message: str
    matches: list[str]


class MovieRecommender:
    def __init__(self, movies: pd.DataFrame, max_features: int = 5000):
        self.movies = movies.reset_index(drop=True)
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),  # single words and two-word phrases
            min_df=2,  # ignore extremely rare noise tokens when possible
        )
        self.tfidf_matrix = None
        self._title_lookup: dict[str, int] = {}

    def fit(self) -> "MovieRecommender":
        """Prepare TF-IDF vectors. Call this once after loading data."""
        content = self.movies.apply(self._content_soup, axis=1)
        if content.fillna("").str.strip().eq("").all():
            raise ValueError(
                "Every movie has empty text after cleaning. "
                "Check that your CSV has title/genres/director/cast columns."
            )
        # Some tiny datasets may have min_df=2 fail; fall back gracefully.
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(content)
        except ValueError:
            self.vectorizer.set_params(min_df=1)
            self.tfidf_matrix = self.vectorizer.fit_transform(content)

        self._title_lookup = {
            str(title).strip().lower(): index
            for index, title in enumerate(self.movies["title"].tolist())
        }
        return self

    def search(self, query: str, limit: int = 15) -> SearchResult:
        """Find a movie title even if the user types a partial or messy name."""
        query = (query or "").strip()
        if not query:
            return SearchResult(False, "", "Please type a movie name.", [])

        titles = self.movies["title"].tolist()
        query_lower = query.lower()

        exact = [title for title in titles if str(title).strip().lower() == query_lower]
        if exact:
            return SearchResult(True, exact[0], "", exact[:limit])

        contains = [title for title in titles if query_lower in str(title).lower()]
        if contains:
            return SearchResult(
                True,
                contains[0],
                f"No exact title match. Using closest name: {contains[0]}",
                contains[:limit],
            )

        close = difflib.get_close_matches(query, titles, n=limit, cutoff=0.45)
        if close:
            return SearchResult(
                True,
                close[0],
                f'No exact title match. Did you mean "{close[0]}"?',
                close,
            )

        return SearchResult(
            False,
            "",
            f'No movie found for "{query}". Try another title or clear the genre filter.',
            [],
        )

    def get_details(self, title: str) -> dict | None:
        index = self._index_for_title(title)
        if index is None:
            return None
        row = self.movies.iloc[index]
        return {column: row[column] for column in self.movies.columns}

    def recommend(
        self,
        title: str,
        n_recommendations: int = 8,
        genre_filter: str | None = None,
    ) -> pd.DataFrame:
        """
        Return similar movies.

        Similarity is computed for one movie at a time (not a giant NxN matrix)
        so a laptop can handle thousands of titles.
        """
        if self.tfidf_matrix is None:
            raise RuntimeError("Call fit() before recommend().")

        index = self._index_for_title(title)
        if index is None:
            return pd.DataFrame()

        movie_vector = self.tfidf_matrix[index]
        scores = cosine_similarity(movie_vector, self.tfidf_matrix).flatten()

        # Rank every other movie. The selected movie itself is score ~ 1.0
        ranked_indices = np.argsort(scores)[::-1]

        rows = []
        for other_index in ranked_indices:
            if other_index == index:
                continue
            row = self.movies.iloc[int(other_index)]
            if genre_filter and genre_filter != "All":
                movie_genres = {name.lower() for name in genre_list(row["genres"])}
                if genre_filter.lower() not in movie_genres:
                    continue
            rows.append(
                {
                    "title": row["title"],
                    "genres": row["genres"],
                    "director": row["director"],
                    "cast": row["cast"],
                    "overview": row["overview"],
                    "keywords": row["keywords"],
                    "vote_average": row["vote_average"],
                    "vote_count": row["vote_count"],
                    "release_date": row["release_date"],
                    "similarity": round(float(scores[other_index]), 4),
                }
            )
            if len(rows) >= n_recommendations:
                break

        return pd.DataFrame(rows)

    def available_genres(self) -> list[str]:
        unique = set()
        for value in self.movies["genres"].fillna(""):
            unique.update(genre_list(value))
        return sorted(unique)

    def titles_for_genre(self, genre: str | None) -> list[str]:
        titles = self.movies.copy()
        if genre and genre != "All":
            wanted = genre.lower()
            mask = titles["genres"].apply(
                lambda value: wanted in {name.lower() for name in genre_list(value)}
            )
            titles = titles[mask]
        return titles["title"].tolist()

    def _index_for_title(self, title: str) -> int | None:
        return self._title_lookup.get(str(title).strip().lower())

    @staticmethod
    def _content_soup(row: pd.Series) -> str:
        """
        Combine metadata into one document.

        Uses only genre, director, and cast for similarity.
        """
        genres = clean_text(row.get("genres", ""))
        director = clean_text(row.get("director", ""))
        cast = clean_text(row.get("cast", ""))

        parts = [
            genres,
            director,
            cast,
        ]
        return " ".join(part for part in parts if part)
