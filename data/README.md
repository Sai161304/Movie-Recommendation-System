# Dataset instructions

This project does **not** ship movie data. You must download a public dataset and place the CSV files in this `data/` folder.

The code supports two formats. Use **Option A** if you want overviews and keywords. Use **Option B** if you want a fully official, no-login download.

---

## Option A (recommended): TMDB 5000 Movie Metadata

This is the usual academic dataset for **content-based** recommenders because it includes:

- `title`
- `overview` (plot summary)
- `genres`
- `keywords`
- `tagline`
- `vote_average`, `vote_count`, `release_date`, `runtime`
- optional `tmdb_5000_credits.csv` for director and top cast

### How to obtain it

1. Open the Kaggle dataset page: [TMDB 5000 Movie Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata)
2. Sign in to Kaggle (free).
3. Click **Download**.
4. Unzip the files.
5. Copy these files into this `data/` folder:
   - `tmdb_5000_movies.csv`
   - `tmdb_5000_credits.csv` (optional, but recommended)

### License and attribution (important)

- The records come from [The Movie Database (TMDB)](https://www.themoviedb.org/).
- TMDB data is intended for **personal / non-commercial** use, with attribution. Commercial use needs a separate agreement with TMDB.
- TMDB’s own API terms **do not allow redistributing a full dumped dataset**. That is why this repo does not include the CSV files.
- Kaggle listing: https://www.kaggle.com/tmdb/tmdb-movie-metadata
- TMDB terms: https://www.themoviedb.org/terms-of-use
- Suggested UI notice: *This product uses the TMDB API / TMDB data but is not endorsed or certified by TMDB.*

Use this option for a final-year project demo that talks about plot text, keywords, and metadata.

---

## Option B (no login): MovieLens Latest Small

Official files from GroupLens. No API key.

This option has **genres**, user **tags** (used as keywords), and **ratings**. It does **not** include plot overviews.

### How to obtain it

**Manual download**

1. Open: https://grouplens.org/datasets/movielens/
2. Download **ml-latest-small.zip**  
   Direct zip (official): https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
3. Unzip it.
4. Copy these files into this `data/` folder:
   - `movies.csv`
   - `tags.csv`
   - `ratings.csv`

**Or run the helper script** (from the project root, after installing requirements):

```bash
python scripts/download_movielens.py
```

### License and citation

MovieLens data may be used for **research / educational** purposes. You must:

- Not imply endorsement by the University of Minnesota or GroupLens
- Cite the dataset in reports/papers
- Not use it for commercial / revenue-bearing purposes without permission
- Redistribute only under the same license conditions

**Citation**

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015). https://doi.org/10.1145/2827872

License text: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html

---

## Expected columns (if you use your own CSV)

Minimum useful file: a CSV named `movies.csv` with at least:

| column     | required | notes                          |
|------------|----------|--------------------------------|
| title      | yes      | movie name                     |
| genres     | no       | `Action\|Comedy` or JSON list  |
| overview   | no       | plot text                      |
| keywords   | no       | words or JSON list             |
| tagline    | no       | short slogan                   |
| vote_average | no     | shown in the UI                |
| vote_count | no       | shown in the UI                |

The loader will fill missing text fields with empty strings so TF-IDF can still run.
