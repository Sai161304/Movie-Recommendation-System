"""
Download the official MovieLens Latest Small dataset into data/.

This does not create fake movies. It only fetches GroupLens' public zip.

Run from the project root:

    python scripts/download_movielens.py

License: research / education only. See data/README.md.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
NEEDED_FILES = ("movies.csv", "tags.csv", "ratings.csv")


def download_zip(destination: Path) -> None:
    print(f"Downloading:\n  {ZIP_URL}")
    try:
        with urlopen(ZIP_URL, timeout=60) as response:
            destination.write_bytes(response.read())
    except URLError as error:
        raise SystemExit(
            "Download failed. Check your internet connection, then either retry "
            "this script or download the zip by hand:\n"
            f"  {ZIP_URL}\n\n"
            f"Details: {error}"
        ) from error


def extract_csv_files(zip_path: Path) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        copied = []
        for needed in NEEDED_FILES:
            match = next((name for name in names if name.endswith(needed)), None)
            if match is None:
                continue
            target = DATA_DIR / Path(match).name
            target.write_bytes(archive.read(match))
            copied.append(target.name)
    if "movies.csv" not in copied:
        raise SystemExit("The zip downloaded, but movies.csv was not inside it.")
    print("Saved to data/:")
    for name in copied:
        print(f"  - {name}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "ml-latest-small.zip"
    download_zip(zip_path)
    extract_csv_files(zip_path)
    print("\nMovieLens files are ready.")
    print("This dataset has genres + user tags, but not plot overviews.")
    print("For overviews/keywords, use TMDB 5000 instead (see data/README.md).")
    print("\nNext:")
    print("  python -m src.evaluator")
    print("  streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
