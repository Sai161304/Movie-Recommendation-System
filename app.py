"""
Streamlit front end for the content-based movie recommender.

Run from the project root:

    streamlit run app.py
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import DatasetNotFoundError, describe_dataset, load_movies
from src.recommender import MovieRecommender

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_README = PROJECT_ROOT / "data" / "README.md"

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.4rem; max-width: 1200px; }
            .hero {
                background: linear-gradient(135deg, #111827 0%, #1f2937 60%, #0f766e 100%);
                color: #f9fafb;
                padding: 1.4rem 1.6rem;
                border-radius: 16px;
                margin-bottom: 1rem;
            }
            .hero h1 { margin: 0 0 0.35rem 0; font-size: 1.8rem; }
            .hero p { margin: 0; opacity: 0.92; }
            .movie-card {
                border: 1px solid #e5e7eb;
                border-radius: 14px;
                padding: 0.9rem 1rem;
                background: #ffffff;
                height: 100%;
            }
            .muted { color: #6b7280; font-size: 0.92rem; }
            .score { color: #0f766e; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Building TF-IDF vectors...")
def get_recommender() -> MovieRecommender:
    movies = load_movies()
    return MovieRecommender(movies).fit()


def pretty_genres(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "Not listed"
    return ", ".join(part.strip().title() for part in text.split(",") if part.strip())


def format_rating(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "N/A"


def show_dataset_help() -> None:
    st.error("No movie dataset was found.")
    st.markdown(
        """
        The app does not invent movies and does not need an API key.
        Download a public dataset, then refresh this page.

        **Option A — TMDB 5000 (best for plot overviews and keywords)**  
        1. Download from [Kaggle: TMDB 5000 Movie Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata)  
        2. Copy `tmdb_5000_movies.csv` (and optionally `tmdb_5000_credits.csv`) into the `data/` folder.

        **Option B — MovieLens (no login)**  
        In a terminal, from this project folder:
        ```bash
        python scripts/download_movielens.py
        ```
        """
    )
    if DATA_README.exists():
        with st.expander("Full dataset and license notes"):
            st.markdown(DATA_README.read_text(encoding="utf-8"))


def render_movie_details(details: dict) -> None:
    st.subheader(details["title"])
    meta_bits = [
        pretty_genres(details.get("genres")),
        f"Rating {format_rating(details.get('vote_average'))}",
        f"{int(details.get('vote_count') or 0)} votes",
    ]
    if details.get("release_date"):
        meta_bits.insert(1, str(details["release_date"])[:10])
    st.caption(" · ".join(meta_bits))

    overview = str(details.get("overview") or "").strip()
    if overview:
        st.write(overview)
    else:
        st.info("No plot overview is available for this title in the current dataset.")

    extra_cols = st.columns(3)
    extra_cols[0].markdown(f"**Keywords**  \n{details.get('keywords') or 'Not listed'}")
    extra_cols[1].markdown(f"**Director**  \n{details.get('director') or 'Not listed'}")
    extra_cols[2].markdown(f"**Top cast**  \n{details.get('cast') or 'Not listed'}")


def render_recommendations(recs: pd.DataFrame, genre_filter: str) -> None:
    st.subheader("Recommended movies")
    if recs.empty:
        if genre_filter != "All":
            st.warning(
                f"No similar movies were found in the **{genre_filter.title()}** genre. "
                "Clear the genre filter or try another title."
            )
        else:
            st.warning("No recommendations could be produced for this title.")
        return

    st.caption("Ranked by cosine similarity of TF-IDF text vectors.")
    columns = st.columns(2)
    for index, row in recs.iterrows():
        with columns[index % 2]:
            overview = str(row.get("overview") or "").strip()
            preview = overview[:220] + ("..." if len(overview) > 220 else "")
            if not preview:
                preview = "No plot overview in this dataset."
            st.markdown(
                f"""
                <div class="movie-card">
                    <h4>{html.escape(str(row['title']))}</h4>
                    <p class="muted">{html.escape(pretty_genres(row['genres']))}</p>
                    <p>{html.escape(preview)}</p>
                    <p class="score">Similarity: {row['similarity']:.3f}
                    &nbsp;·&nbsp; Rating {format_rating(row['vote_average'])}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    inject_css()
    st.markdown(
        """
        <div class="hero">
            <h1>🎬 Movie Recommendation System</h1>
            <p>Content-based suggestions using genres, overviews, keywords, and other metadata.
            Powered by TF-IDF and cosine similarity. No paid APIs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        recommender = get_recommender()
    except DatasetNotFoundError:
        show_dataset_help()
        return
    except Exception as error:  # show a readable error instead of a blank page
        st.error("The recommender could not start.")
        st.exception(error)
        return

    info = describe_dataset(recommender.movies)
    genres = ["All"] + recommender.available_genres()

    with st.sidebar:
        st.header("Filters")
        genre_filter = st.selectbox(
            "Genre",
            options=genres,
            format_func=lambda value: "All genres" if value == "All" else value.title(),
        )
        n_recs = st.slider("Number of recommendations", min_value=5, max_value=15, value=8)
        st.divider()
        st.header("Dataset")
        st.write(f"**Source:** `{info['source']}`")
        st.write(f"**Movies:** {info['n_movies']}")
        st.write(f"**Overviews filled:** {info['overview_coverage']:.0%}")
        st.write(f"**Keywords filled:** {info['keyword_coverage']:.0%}")
        if info["source"] == "movielens_small":
            st.caption(
                "MovieLens has genres and user tags, but not plot overviews. "
                "Use TMDB 5000 for richer text (see data/README.md)."
            )
        elif info["source"] == "tmdb_5000":
            st.caption("This product uses TMDB data but is not endorsed or certified by TMDB.")
        with st.expander("How this model works"):
            st.markdown(
                """
                1. Each movie becomes one text document (the *content soup*).
                2. **TF-IDF** turns words into numbers.
                3. **Cosine similarity** finds the closest documents.
                4. The selected movie is removed from its own recommendation list.
                """
            )
        with st.expander("Limitations"):
            st.markdown(
                """
                Content-based filtering cannot see what *users* liked together.
                It can also over-recommend the same genre and it does not
                understand synonyms unless the words actually appear.
                """
            )

    catalog = recommender.titles_for_genre(genre_filter)
    if not catalog:
        st.warning(
            f'No movies were found for the genre "{genre_filter.title()}". '
            "Choose another genre."
        )
        return

    search_query = st.text_input(
        "Search for a movie",
        placeholder="Try: Inception, Toy Story, The Dark Knight...",
    )

    selected_title = None
    if search_query.strip():
        result = recommender.search(search_query)
        if not result.found:
            st.error(result.message)
            st.info("Tip: pick a title from the list below, or clear part of the search.")
        else:
            allowed_matches = [title for title in result.matches if title in set(catalog)]
            if not allowed_matches:
                st.error(
                    f'Found "{result.title}", but it is not in the current genre filter. '
                    "Set Genre to All genres, or search another title."
                )
            else:
                if result.message:
                    st.info(result.message)
                selected_title = st.selectbox("Matching titles", options=allowed_matches)
    else:
        selected_title = st.selectbox("Or choose a movie from the list", options=catalog)

    if not selected_title:
        st.warning("No movie is selected yet. Search or pick a title to see recommendations.")
        return

    details = recommender.get_details(selected_title)
    if details is None:
        st.error(f'Movie details were not found for "{selected_title}".')
        return

    render_movie_details(details)
    recs = recommender.recommend(
        selected_title,
        n_recommendations=n_recs,
        genre_filter=genre_filter,
    )
    st.markdown("---")
    render_recommendations(recs, genre_filter)


if __name__ == "__main__":
    main()
