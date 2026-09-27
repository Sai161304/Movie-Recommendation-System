"""
Clean messy movie text so TF-IDF sees consistent words.

Beginner notes:
- TF-IDF counts words. "Action" and "action" should be the same.
- Extra spaces, punctuation, and NaN values would hurt similarity scores.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

import pandas as pd


_NON_LETTERS = re.compile(r"[^a-z0-9\s]+")
_MULTI_SPACE = re.compile(r"\s+")


def clean_text(value: Any) -> str:
    """Turn almost anything into a lowercase bag of words."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null", "[]", "{}"}:
        return ""
    text = _NON_LETTERS.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def fill_missing_text(series: pd.Series) -> pd.Series:
    """Replace NaN / None with empty strings."""
    return series.fillna("").astype(str).replace({"nan": "", "None": "", "NaN": ""})


def parse_name_list(raw_value: Any) -> str:
    """
    Convert many genre/keyword formats into a comma-separated string.

    Handles:
    - MovieLens: "Action|Adventure|Comedy"
    - TMDB JSON: '[{"id": 28, "name": "Action"}, ...]'
    - already plain text: "action adventure"
    """
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return ""

    if isinstance(raw_value, list):
        names = [_extract_name(item) for item in raw_value]
        return ", ".join(name for name in names if name)

    text = str(raw_value).strip()
    if not text or text in {"[]", "{}", "nan", "None"}:
        return ""

    if text.startswith("[") or text.startswith("{"):
        parsed = _safe_parse_structure(text)
        if isinstance(parsed, list):
            names = [_extract_name(item) for item in parsed]
            return ", ".join(name for name in names if name)
        if isinstance(parsed, dict):
            return _extract_name(parsed)

    # MovieLens uses a pipe | between genres
    if "|" in text:
        parts = [clean_text(part) for part in text.split("|")]
        return ", ".join(part for part in parts if part)

    return clean_text(text)


def genre_list(genres: str | None) -> list[str]:
    """Split stored genres into individual labels."""
    text = str(genres or "").strip()
    if not text:
        return []
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [part.strip() for part in text.split() if part.strip()]


def parse_top_cast(raw_cast: Any, top_n: int = 3) -> str:
    """Take the first few actor names from a TMDB credits 'cast' JSON string."""
    people = _safe_parse_structure(raw_cast)
    if not isinstance(people, list):
        return ""
    names = []
    for person in people[:top_n]:
        name = _extract_name(person)
        if name:
            names.append(name)
    return " ".join(names)


def parse_director(raw_crew: Any) -> str:
    """Find the director name in a TMDB credits 'crew' JSON string."""
    people = _safe_parse_structure(raw_crew)
    if not isinstance(people, list):
        return ""
    directors = []
    for person in people:
        if not isinstance(person, dict):
            continue
        job = str(person.get("job", "")).strip().lower()
        if job == "director":
            name = _extract_name(person)
            if name:
                directors.append(name)
    return " ".join(directors)


def drop_duplicate_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one row per movie title.

    If popularity-like columns exist, keep the better-known copy.
    """
    working = movies.copy()
    working["title_key"] = working["title"].fillna("").astype(str).str.strip().str.lower()
    working = working[working["title_key"] != ""]

    sort_columns = [column for column in ("vote_count", "popularity", "rating_count") if column in working.columns]
    if sort_columns:
        working = working.sort_values(sort_columns, ascending=False)

    working = working.drop_duplicates(subset=["title_key"], keep="first")
    return working.drop(columns=["title_key"]).reset_index(drop=True)


def _extract_name(item: Any) -> str:
    if isinstance(item, dict):
        return clean_text(item.get("name", ""))
    return clean_text(item)


def _safe_parse_structure(raw_value: Any) -> Any:
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return None
    if isinstance(raw_value, (list, dict)):
        return raw_value
    text = str(raw_value).strip()
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError, TypeError, RecursionError):
            continue
    return None
