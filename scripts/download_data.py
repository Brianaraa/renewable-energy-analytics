"""
scripts/download_data.py
=========================
Helper script to download the Global Power Plant Database from WRI.
Run once before starting the dashboard.

Usage:
    python scripts/download_data.py
"""

import sys
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_URL = (
    "https://datasets.wri.org/dataset/540dcf46-f287-47ac-985d-269b04bea4c6/"
    "resource/9c3bd566-28c3-4f2f-9e88-a9cdbbf05dea/download/"
    "global_power_plant_database_v_1_3.zip"
)
ZIP_FILE = RAW_DIR / "gppd.zip"
CSV_NAME = "global_power_plant_database.csv"

MAX_RETRIES = 3
TIMEOUT_SECONDS = 120


def _download_with_retry(url: str, dest: Path, retries: int = MAX_RETRIES) -> None:
    """Download a URL to a local file with retry logic and progress feedback."""
    for attempt in range(1, retries + 1):
        try:
            print(f"  Attempt {attempt}/{retries} …")
            urllib.request.urlretrieve(url, dest)
            return
        except urllib.error.URLError as e:
            print(f"  ❌ Network error: {e.reason}")
        except urllib.error.HTTPError as e:
            print(f"  ❌ HTTP error {e.code}: {e.reason}")
        except TimeoutError:
            print(f"  ❌ Timeout after {TIMEOUT_SECONDS}s")

        if attempt < retries:
            wait = 2 ** attempt
            print(f"  ⏳ Retrying in {wait}s …")
            time.sleep(wait)

    raise ConnectionError(
        f"Failed to download after {retries} attempts.\n"
        f"URL: {url}\n"
        "Please check your internet connection or download manually from:\n"
        "https://datasets.wri.org/dataset/globalpowerplantdatabase"
    )


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / CSV_NAME

    if target.exists():
        size_mb = target.stat().st_size / 1e6
        print(f"✅ Dataset already exists at {target} ({size_mb:.1f} MB)")
        return

    print(f"⬇️  Downloading dataset from WRI …")
    try:
        _download_with_retry(DATA_URL, ZIP_FILE)
    except ConnectionError as e:
        print(f"\n❌ Download failed:\n{e}")
        sys.exit(1)
    print(f"✅ Downloaded → {ZIP_FILE}")

    print("📦 Extracting …")
    try:
        with zipfile.ZipFile(ZIP_FILE, "r") as z:
            csv_found = False
            for name in z.namelist():
                if name.endswith(".csv"):
                    z.extract(name, RAW_DIR)
                    extracted = RAW_DIR / name
                    if extracted.name != CSV_NAME:
                        extracted.rename(RAW_DIR / CSV_NAME)
                    csv_found = True
                    break
            if not csv_found:
                print("❌ No CSV found inside ZIP archive.")
                sys.exit(1)
    except zipfile.BadZipFile:
        print("❌ Downloaded file is not a valid ZIP archive.")
        ZIP_FILE.unlink(missing_ok=True)
        sys.exit(1)

    ZIP_FILE.unlink(missing_ok=True)
    size_mb = (RAW_DIR / CSV_NAME).stat().st_size / 1e6
    print(f"✅ Dataset ready at {RAW_DIR / CSV_NAME} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    download()
