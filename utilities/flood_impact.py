import os
import math
import time
import requests
import pandas as pd
from tqdm import tqdm

API_BASE = "https://www.fema.gov/api/open/v2/FimaNfipClaims"
SELECTED_FIELDS = "reportedZipCode,yearOfLoss,floodEvent"
PAGE_SIZE = 10_000
MIN_YEAR = 2010
CACHE_PATH = "data/intermediate/flood_claims_per_zip.csv"


def _get_total_count(session: requests.Session) -> int:
    params = {
        "$top": 1,
        "$select": "reportedZipCode",
        "$filter": f"yearOfLoss ge {MIN_YEAR}",
        "$inlinecount": "allpages",
    }
    resp = session.get(API_BASE, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["metadata"]["count"]


def _fetch_page(session: requests.Session, skip: int, max_retries: int = 5) -> list[dict]:
    params = {
        "$top": PAGE_SIZE,
        "$skip": skip,
        "$filter": f"yearOfLoss ge {MIN_YEAR}",
        "$select": SELECTED_FIELDS,
        "$format": "json",
        "$metadata": "off",
    }
    for attempt in range(max_retries):
        try:
            resp = session.get(API_BASE, params=params, timeout=90)
            if resp.status_code in (429, 503, 502, 504):
                wait = int(resp.headers.get("Retry-After", 2 ** (attempt + 2)))
                tqdm.write(f"  HTTP {resp.status_code} at skip={skip}, attempt {attempt + 1}/{max_retries}. Waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json().get("FimaNfipClaims", [])
        except (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError):
            wait = 2 ** attempt
            tqdm.write(f"  Connection error at skip={skip}, attempt {attempt + 1}/{max_retries}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch page at skip={skip} after {max_retries} attempts")


def fetch_flood_data_by_zip() -> pd.DataFrame:
    """
    Downloads FEMA NFIP claims from MIN_YEAR onwards and returns the number of
    distinct flood events per ZIP code.
    """
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "US-Housing-Analysis/1.0"})

    print(f"Fetching total NFIP claim count (year >= {MIN_YEAR})...")
    total = _get_total_count(session)
    n_pages = math.ceil(total / PAGE_SIZE)
    print(f"  {total:,} records → {n_pages} pages of {PAGE_SIZE:,}")

    # zip_code -> set of distinct (yearOfLoss, floodEvent) tuples
    zip_events: dict[str, set] = {}

    for page in tqdm(range(n_pages), desc="Downloading NFIP pages"):
        records = _fetch_page(session, skip=page * PAGE_SIZE)
        if not records:
            break
        time.sleep(0.3)

        for r in records:
            z = r.get("reportedZipCode")
            if not z or z == "Currently Unavailable":
                continue
            z = str(z).strip().zfill(5)

            if z not in zip_events:
                zip_events[z] = set()

            event = r.get("floodEvent") or "unknown"
            year = r.get("yearOfLoss") or 0
            zip_events[z].add((year, event))

    rows = [
        {"zip_code": z, "flood_count": len(events)}
        for z, events in zip_events.items()
    ]
    return pd.DataFrame(rows)


def generate_csv() -> pd.DataFrame:
    os.makedirs("data/intermediate", exist_ok=True)

    if os.path.exists(CACHE_PATH):
        print(f"Flood data already cached, loading from {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH, dtype={"zip_code": str})

    df = fetch_flood_data_by_zip()
    df.to_csv(CACHE_PATH, index=False)
    print(f"\nFlood data saved: {len(df):,} ZIP codes → {CACHE_PATH}")
    print(df.sort_values("flood_count", ascending=False).head(10).to_string(index=False))
    return df


if __name__ == "__main__":
    generate_csv()
