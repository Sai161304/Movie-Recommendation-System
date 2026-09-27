# Movie Recommendation System

A beginner-friendly **content-based** movie recommender built with Python, Pandas, NumPy, Scikit-learn, and Streamlit.

The model does **not** use user-user collaborative filtering and does **not** call paid APIs. It recommends movies whose **text metadata** looks similar: genres, overview, keywords, tagline, director, and top cast when those fields exist.

This style of project is a common final-year CSE demo because you can explain the full pipeline: data cleaning → vectorization → similarity → UI.

---

## What you need from me (you run these)

I created the source files only. I did **not** install packages, did **not** download a dataset, and did **not** start Streamlit. Please run the commands below yourself.

### 1. Create a virtual environment (recommended)

**Windows PowerShell** (from this project folder):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Get a dataset (required)

There is no movie CSV in this repo on purpose (license + size). Pick one:

**Option A — TMDB 5000 (recommended, has overviews + keywords)**

1. Download: https://www.kaggle.com/tmdb/tmdb-movie-metadata
2. Copy `tmdb_5000_movies.csv` and, if you have it, `tmdb_5000_credits.csv` into `data/`
3. Non-commercial / attribution use. See `data/README.md`.

**Option B — MovieLens Latest Small (no login, official zip)**

```powershell
python scripts/download_movielens.py
```

Or download by hand: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip  
Then put `movies.csv`, `tags.csv`, and `ratings.csv` in `data/`.

MovieLens license is **research/education**, not commercial. Cite Harper & Konstan (2015). Details in `data/README.md`.

### 3. Run the app

```powershell
streamlit run app.py
```

Your browser should open a local URL (often `http://localhost:8501`).

### 4. Run the evaluation demo

```powershell
python -m src.evaluator
```

### 5. Optional: run small unit tests (no dataset needed)

```powershell
python -m unittest tests/test_text_cleaner.py
```

---

## Folder structure

```
Movie_Recommendation_System/
├── app.py                         Streamlit UI
├── requirements.txt               Python libraries
├── README.md                      This file
├── .gitignore
├── data/                          You place CSVs here (not committed)
│   ├── README.md                  How to obtain data + licenses
│   └── .gitkeep
├── scripts/
│   └── download_movielens.py      Official MovieLens downloader
├── src/
│   ├── __init__.py
│   ├── data_loader.py             Load TMDB or MovieLens CSVs
│   ├── text_cleaner.py            Missing values, duplicates, messy text
│   ├── recommender.py             TF-IDF + cosine similarity
│   └── evaluator.py               Demo + genre-overlap sanity check
└── tests/
    └── test_text_cleaner.py       Tiny tests for cleaning helpers
```

---

## What each file does

| File | Role |
|------|------|
| `app.py` | Search/select a movie, show details, 5–15 recommendations, genre filter, not-found messages |
| `src/data_loader.py` | Detects dataset type, merges extra files, standardizes columns |
| `src/text_cleaner.py` | Fills missing text, drops duplicate titles, parses JSON / pipe genres |
| `src/recommender.py` | Builds a “content soup”, fits TF-IDF, ranks by cosine similarity |
| `src/evaluator.py` | Prints sample lists and explains why content-based filtering is limited |
| `scripts/download_movielens.py` | Fetches GroupLens’ official zip only |
| `data/README.md` | Dataset obtain steps and license notes |

---

## How the recommendation algorithm works

1. **Content soup**  
   For each movie we concatenate cleaned genres (repeated for extra weight), overview, keywords, tagline, director, cast, and title.
2. **TF-IDF** (`sklearn.feature_extraction.text.TfidfVectorizer`)  
   - Term Frequency: word is important if it appears often in *this* movie.  
   - Inverse Document Frequency: word is *less* useful if it appears in almost every movie (“film”, “story”, …).  
   - Unigrams and bigrams (`ngram_range=(1, 2)`), English stop words removed.
3. **Cosine similarity**  
   For the selected movie vector `v`, we compute `cosine_similarity(v, all_movie_vectors)`.  
   We do **not** store a full N×N matrix. That keeps memory reasonable on a student laptop.
4. **Ranking**  
   Sort scores, drop the movie itself, optionally keep only one genre, return the top N.

This is **item-item content similarity**, not “users who liked X also liked Y”.

---

## How to test it like a user

After `streamlit run app.py`:

1. Confirm the sidebar shows a movie count greater than 0.
2. Search `Inception` or `Toy Story` (names depend on the dataset).
3. You should see details + at least 5 similar titles with a similarity score.
4. Type a nonsense name such as `zzzznotamovie`. You should get a clear **not found** message.
5. Pick a genre filter. Recommendations should stay in that genre, or you should see a warning if none remain.
6. Run `python -m src.evaluator` and read the printed limitations. That output is meant for viva/interviews.

I have **not** claimed these checks passed in this environment, because packages and data were not installed/downloaded here.

---

## Interview talking points

**Why content-based?**  
Works with only item metadata. A new movie can be recommended as soon as we have its text. We do not need a large user-rating matrix.

**Why TF-IDF instead of counting words?**  
Raw counts over-weight long plots and common words. IDF down-weights words that appear everywhere.

**Why cosine similarity instead of Euclidean distance?**  
Cosine cares about direction (the mix of words), not vector length. A long overview and a short overview can still be similar.

**What is the evaluation catch?**  
There is no single ground-truth “correct” recommendation. The demo uses **genre Jaccard overlap** as a *sanity check*, not as a user-satisfaction metric. Better research metrics (Precision@K, MAP, NDCG) need a rating or click log.

**Limitations (say these out loud)**  
- Over-specialization (more of the same genre)  
- No collaborative signal  
- Sensitive to missing/poor metadata  
- TF-IDF has no real language understanding (synonyms, word sense)  
- Does not model popularity or recency unless you add them

**Natural extension**  
Hybrid: content score + collaborative filtering (truncated SVD on the ratings matrix).

---

## Citation

If you use MovieLens:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015). https://doi.org/10.1145/2827872

If you use TMDB 5000, attribute [The Movie Database](https://www.themoviedb.org/) and follow their non-commercial terms. This product is not endorsed or certified by TMDB.
