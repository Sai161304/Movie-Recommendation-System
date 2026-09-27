"""
Simple evaluation / demo for the content-based recommender.

This is NOT a full offline ranking experiment (we have no held-out
"correct" next movie). Instead it shows:

1) Example recommendations for well-known titles (if they exist).
2) A genre-overlap score: do recommended movies share genres with the seed?
3) Honest limitations of content-based filtering.

Run from the project root:

    python -m src.evaluator
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import DatasetNotFoundError, describe_dataset, load_movies
from src.recommender import MovieRecommender
from src.text_cleaner import genre_list


DEMO_TITLES = [
    "The Dark Knight",
    "Toy Story",
    "Avatar",
    "Inception",
    "The Godfather",
    "Titanic",
]


LIMITATIONS = """
Limitations of content-based filtering (say this in an interview)
-----------------------------------------------------------------
1. Cold-start for NEW movies is easy; cold-start for NEW users is not solved.
   The model only looks at item metadata, not at what people actually watch.
2. Over-specialization: if you like one superhero film, you may only see
   more superhero films. Discovery of different-but-enjoyable movies is weak.
3. Quality depends on metadata. Missing overviews/keywords make scores worse.
4. TF-IDF does not understand meaning. "bank" (money) and "bank" (river)
   look the same. It also cannot tell that two plots are similar if they
   use completely different words.
5. Popularity bias is not modeled unless we add it. A cult film and a
   blockbuster with the same words look equally "similar".
6. No ratings/collaborative signal: "users who liked X also liked Y"
   is a different family of methods (collaborative filtering / SVD).

A stronger production system usually *hybridizes* content-based and
collaborative filtering.
""".strip()


def genre_overlap(seed_genres: str, recommended_genres: str) -> float:
    """Jaccard overlap between two genre strings."""
    seed = set(genre_list(seed_genres))
    other = set(genre_list(recommended_genres))
    if not seed or not other:
        return 0.0
    return len(seed & other) / len(seed | other)


def run_demo(n_recommendations: int = 8) -> None:
    print("Loading dataset from data/ ...")
    try:
        movies = load_movies()
    except DatasetNotFoundError as error:
        print(error)
        sys.exit(1)

    info = describe_dataset(movies)
    print(f"Movies loaded : {info['n_movies']}")
    print(f"Dataset source: {info['source']}")
    print(f"Rows with overview: {info['overview_coverage']:.0%}")
    print(f"Rows with keywords: {info['keyword_coverage']:.0%}")
    print()

    recommender = MovieRecommender(movies).fit()
    print(f"TF-IDF vocabulary size: {len(recommender.vectorizer.get_feature_names_out())}")
    print()

    used_titles = []
    overlap_scores = []

    for query in DEMO_TITLES:
        result = recommender.search(query)
        if not result.found:
            continue
        used_titles.append(result.title)
        details = recommender.get_details(result.title)
        recs = recommender.recommend(result.title, n_recommendations=n_recommendations)
        if recs.empty:
            print(f"{result.title}: no recommendations produced.")
            continue

        movie_overlaps = [
            genre_overlap(str(details["genres"]), str(row["genres"]))
            for _, row in recs.iterrows()
        ]
        mean_overlap = sum(movie_overlaps) / len(movie_overlaps)
        overlap_scores.append(mean_overlap)

        print("=" * 72)
        print(f"Seed movie: {result.title}")
        if result.message:
            print(f"Note: {result.message}")
        print(f"Genres    : {details['genres'] or '(none)'}")
        print(f"Mean genre overlap with recommendations: {mean_overlap:.2f}  (1.0 = identical genres)")
        print("Recommendations:")
        for rank, row in recs.iterrows():
            print(
                f"  {rank + 1:2d}. {row['title']}  "
                f"(similarity={row['similarity']:.3f}, genres={row['genres'] or 'n/a'})"
            )
        print()

    if not used_titles:
        print("None of the demo titles were in this dataset.")
        print("Open the Streamlit app and try a title that appears in your CSV.")
    elif overlap_scores:
        print("-" * 72)
        print(
            f"Average genre overlap across demo movies: "
            f"{sum(overlap_scores) / len(overlap_scores):.2f}"
        )
        print("Higher overlap means recommendations stay in a similar genre neighborhood.")
        print("This is a sanity check, not proof that users would enjoy the list.")

    print()
    print(LIMITATIONS)


if __name__ == "__main__":
    run_demo()
