import requests
import pandas as pd
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uszipcode import SearchEngine
from tqdm import tqdm

tqdm.pandas()


def add_zip_codes(df):
    """Map (Latitude, Longitude) to the nearest US ZIP code (within 30 miles)."""
    search = SearchEngine()

    def lookup(row):
        lat, lon = row["Latitude"], row["Longitude"]
        if pd.isna(lat) or pd.isna(lon):
            return None
        result = search.by_coordinates(lat=lat, lng=lon, radius=30, returns=1)
        return result[0].zipcode if result else None

    print("Mapping coordinates to ZIP codes...")
    df["Zip_Code"] = df.progress_apply(lookup, axis=1)
    return df

def fetch_wildfire_data():
    # NIFC WFIGS Incident Locations API Endpoint
    url = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Incident_Locations/FeatureServer/0/query"

    # Restrict to fires discovered in the last 20 years
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=365 * 20)).strftime("%Y-%m-%d")

    # Parameters to request GeoJSON format and filter the results
    # We filter for fires larger than 0.5 sqkm (~123 acres) to exclude minor brush fires
    # OBJECTID ordering is required for stable pagination across requests
    page_size = 2000
    params = {
        "where": f"IncidentSize > 123 AND FireDiscoveryDateTime >= DATE '{cutoff_date}'",
        "outFields": "IncidentName,IncidentSize,FireDiscoveryDateTime",
        "f": "geojson",
        "returnGeometry": "true",
        "orderByFields": "OBJECTID",
        "resultRecordCount": page_size,
    }

    print("Fetching data from NIFC API...")
    fire_records = []
    offset = 0
    while True:
        page_params = {**params, "resultOffset": offset}
        response = requests.get(url, params=page_params)
        response.raise_for_status()
        data = response.json()
        features = data.get("features", [])
        if not features:
            break

        for feature in features:
            properties = feature.get("properties", {})
            geometry = feature.get("geometry", {})

            # ArcGIS REST API returns dates as Unix timestamps (milliseconds)
            discovery_time = properties.get("FireDiscoveryDateTime")
            if discovery_time:
                discovery_time = pd.to_datetime(discovery_time, unit='ms')

            # Extract coordinates (Longitude, Latitude)
            coordinates = geometry.get("coordinates", [None, None])
            if coordinates and len(coordinates) == 2:
                lon, lat = coordinates
            else:
                lon, lat = None, None

            fire_records.append({
                "Fire_Name": properties.get("IncidentName"),
                "Acres_Burned": properties.get("IncidentSize"),
                "Discovery_Date": discovery_time,
                "Longitude": lon,
                "Latitude": lat
            })

        print(f"  fetched {len(fire_records)} records so far...")
        if not data.get("properties", {}).get("exceededTransferLimit") and len(features) < page_size:
            break
        offset += len(features)

    # Convert the structured list directly into a Pandas DataFrame
    df = pd.DataFrame(fire_records)
    
    # Sort by the largest fires
    df = df.sort_values(by="Acres_Burned", ascending=False).reset_index(drop=True)

    print(f"Successfully loaded {len(df)} wildfire records.")
    return df

if __name__ == "__main__":
    wildfire_df = fetch_wildfire_data()
    wildfire_df = add_zip_codes(wildfire_df)
    print("\nSample of Extracted Wildfire Data:")
    print(wildfire_df.head())

    output_dir = Path("data/intermediate")
    output_dir.mkdir(parents=True, exist_ok=True)

    incidents_path = output_dir / "wildfire_incidents.csv"
    wildfire_df.to_csv(incidents_path, index=False)
    print(f"\nWrote {len(wildfire_df)} records to {incidents_path}")

    hits_per_zip = (
        wildfire_df.dropna(subset=["Zip_Code"])
        .groupby("Zip_Code")
        .size()
        .reset_index(name="Wildfire_Hits")
        .sort_values("Wildfire_Hits", ascending=False)
    )
    hits_path = output_dir / "wildfire_hits_per_zip.csv"
    hits_per_zip.to_csv(hits_path, index=False)
    print(f"Wrote {len(hits_per_zip)} ZIP counts to {hits_path}")