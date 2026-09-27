"""
Load movie CSVs from the data/ folder.

Supported layouts:
1) TMDB 5000: tmdb_5000_movies.csv [+ tmdb_5000_credits.csv]
2) MovieLens small: movies.csv + tags.csv + ratings.csv
3) A generic movies.csv with a title column
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.text_cleaner import (
    drop_duplicate_movies,
    fill_missing_text,
    genre_list,
    parse_director,
    parse_name_list,
    parse_top_cast,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the data/ folder has no usable movie CSV."""


def load_movies(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Return a cleaned table with a stable set of columns for the recommender.

    Always includes: title, genres, overview, keywords, tagline, director, cast,
    vote_average, vote_count, release_date, runtime, source
    """
    folder = Path(data_dir) if data_dir is not None else DATA_DIR
    movies = _load_raw_table(folder)
    movies = _standardize_columns(movies)
    movies = _clean_table(movies)
    if movies.empty:
        raise DatasetNotFoundError("The dataset was loaded, but no valid movie titles remained.")
    return movies


def describe_dataset(movies: pd.DataFrame) -> dict:
    """Small summary used by the UI and the evaluator."""
    has_overview = (movies["overview"].fillna("").str.strip() != "").mean()
    has_keywords = (movies["keywords"].fillna("").str.strip() != "").mean()
    return {
        "n_movies": int(len(movies)),
        "source": str(movies["source"].iloc[0]) if "source" in movies.columns and len(movies) else "unknown",
        "overview_coverage": float(has_overview),
        "keyword_coverage": float(has_keywords),
        "n_genres": int(len({name for value in movies["genres"] for name in genre_list(value)})),
    }


def _load_raw_table(folder: Path) -> pd.DataFrame:
    tmdb_movies = folder / "tmdb_5000_movies.csv"
    tmdb_credits = folder / "tmdb_5000_credits.csv"
    movielens_movies = folder / "movies.csv"

    if tmdb_movies.exists():
        movies = pd.read_csv(tmdb_movies)
        movies["source"] = "tmdb_5000"
        if tmdb_credits.exists():
            credits = pd.read_csv(tmdb_credits)
            movies = _merge_tmdb_credits(movies, credits)
        return movies

    if movielens_movies.exists():
        return _load_movielens(folder)

    raise DatasetNotFoundError(
        "No movie dataset found in the data/ folder.\n\n"
        "Recommended: download TMDB 5000 and copy tmdb_5000_movies.csv here.\n"
        "No-login option: run  python scripts/download_movielens.py\n"
        "Full steps: see data/README.md"
    )


def _merge_tmdb_credits(movies: pd.DataFrame, credits: pd.DataFrame) -> pd.DataFrame:
    credits = credits.copy()
    # Different dumps use id or movie_id
    if "movie_id" in credits.columns:
        credits = credits.rename(columns={"movie_id": "id"})
    keep = [column for column in ("id", "cast", "crew") if column in credits.columns]
    credits = credits[keep].drop_duplicates(subset=["id"])
    merged = movies.merge(credits, on="id", how="left", suffixes=("", "_credits"))
    return merged


def _load_movielens(folder: Path) -> pd.DataFrame:
    movies = pd.read_csv(folder / "movies.csv")
    movies["source"] = "movielens_small"

    tags_path = folder / "tags.csv"
    if tags_path.exists():
        tags = pd.read_csv(tags_path)
        if {"movieId", "tag"}.issubset(tags.columns):
            tag_text = (
                tags.dropna(subset=["tag"])
                .groupby("movieId")["tag"]
                .apply(lambda values: " ".join(str(tag) for tag in values.unique()))
                .reset_index(name="keywords")
            )
            movies = movies.merge(tag_text, on="movieId", how="left")

    ratings_path = folder / "ratings.csv"
    if ratings_path.exists():
        ratings = pd.read_csv(ratings_path)
        if {"movieId", "rating"}.issubset(ratings.columns):
            rating_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
            rating_stats = rating_stats.rename(columns={"mean": "vote_average", "count": "vote_count"})
            movies = movies.merge(rating_stats, on="movieId", how="left")

    return movies


def _standardize_columns(movies: pd.DataFrame) -> pd.DataFrame:
    renamed = movies.rename(
        columns={
            "movie_title": "title",
            "plot": "overview",
            "summary": "overview",
            "description": "overview",
            "tag": "keywords",
            "tags": "keywords",
            "director_name": "director",
        }
    )
    for column in (
        "title",
        "genres",
        "overview",
        "keywords",
        "tagline",
        "director",
        "cast",
        "crew",
        "vote_average",
        "vote_count",
        "release_date",
        "runtime",
        "source",
        "popularity",
    ):
        if column not in renamed.columns:
            renamed[column] = pd.NA
    return renamed


def _clean_table(movies: pd.DataFrame) -> pd.DataFrame:
    working = movies.copy()
    working["title"] = fill_missing_text(working["title"]).str.strip()
    working = drop_duplicate_movies(working)

    working["genres"] = working["genres"].apply(parse_name_list)
    working["keywords"] = working["keywords"].apply(parse_name_list)
    working["overview"] = fill_missing_text(working["overview"])
    working["tagline"] = fill_missing_text(working["tagline"])

    if working["director"].isna().all() and "crew" in working.columns:
        working["director"] = working["crew"].apply(parse_director)
    else:
        working["director"] = working["director"].apply(parse_name_list)

    if working["cast"].notna().any():
        working["cast"] = working["cast"].apply(parse_top_cast)
    else:
        working["cast"] = ""

    working["vote_average"] = pd.to_numeric(working["vote_average"], errors="coerce")
    working["vote_count"] = pd.to_numeric(working["vote_count"], errors="coerce").fillna(0).astype(int)
    working["runtime"] = pd.to_numeric(working["runtime"], errors="coerce")
    working["release_date"] = fill_missing_text(working["release_date"])
    working["source"] = fill_missing_text(working["source"]).replace("", "unknown")

    keep = [
        "title",
        "genres",
        "overview",
        "keywords",
        "tagline",
        "director",
        "cast",
        "vote_average",
        "vote_count",
        "release_date",
        "runtime",
        "source",
    ]
    return working[keep].reset_index(drop=True)
