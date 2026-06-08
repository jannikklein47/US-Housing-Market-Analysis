import pandas as pd
import datetime
import numpy as np
import requests
from io import StringIO
from uszipcode import SearchEngine
from tqdm import tqdm
import os

tqdm.pandas()

def get_noaa_tornado_data(n_years: int = 15):
    """
    Downloads historical tornado data from the NOAA Storm Prediction Center
    and filters it for the last `n_years`.
    """
    current_year = datetime.datetime.now().year

    # Use requests (which uses certifi) so SSL works on macOS Python installs.
    df = None
    for year in range(current_year, 2020, -1):
        url = f"https://www.spc.noaa.gov/wcm/data/1950-{year}_actual_tornadoes.csv"
        print(f"Trying to fetch data from: {url}")
        try:
            resp = requests.get(url, timeout=30)
        except requests.RequestException as e:
            print(f"  Network error: {e}")
            continue
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code}")
            continue
        df = pd.read_csv(StringIO(resp.text))
        print(f"Successfully loaded dataset up to {year}!")
        latest_year_in_data = year
        break

    if df is None:
        raise RuntimeError("Could not fetch the tornado dataset from NOAA.")
        
    # Filter for the last 'n' years
    target_start_year = latest_year_in_data - n_years
    recent_tornadoes = df[df['yr'] > target_start_year].copy()
    print(f"\nFiltered down to {len(recent_tornadoes)} tornadoes from {target_start_year + 1} to {latest_year_in_data}.")
    
    return recent_tornadoes

def analyze_path_and_zips(df):
    """
    Analyzes the path of each tornado, interpolates points between the start 
    and end coordinates, and extracts all affected ZIP codes.
    """
    print("\n--- ANALYZING TORNADO PATHS ---")
    print("Mapping start, end, and path coordinates to ZIP codes...")
    print("(This will take a few minutes as it calculates multiple points per tornado path)")
    
    search = SearchEngine()
    
    def get_zip_from_coords(lat, lon):
        if pd.isna(lat) or pd.isna(lon) or lat == 0.0 or lon == 0.0:
            return None
        # radius=30 miles, returns=1 gets the closest single match
        result = search.by_coordinates(lat=lat, lng=lon, radius=30, returns=1)
        if result:
            return result[0].zipcode
        return None

    def get_affected_zips(row):
        slat, slon = row['slat'], row['slon']
        elat, elon = row['elat'], row['elon']
        
        affected_zips = set()
        
        # 1. Get the starting ZIP code
        start_zip = get_zip_from_coords(slat, slon)
        if start_zip:
            affected_zips.add(start_zip)
            
        # 2. If there is a valid end coordinate, interpolate the path
        if not (pd.isna(elat) or pd.isna(elon) or elat == 0.0 or elon == 0.0):
            # Generate 5 coordinates evenly spaced between start and end
            lats = np.linspace(slat, elat, num=5)
            lons = np.linspace(slon, elon, num=5)
            
            for lat, lon in zip(lats, lons):
                path_zip = get_zip_from_coords(lat, lon)
                if path_zip:
                    affected_zips.add(path_zip)
                    
        # Return as a comma-separated string for easy viewing in a CSV
        if not affected_zips:
            return "Unknown"
        return ", ".join(list(affected_zips))

    # Apply the path calculation to every row
    df['all_affected_zips'] = df.progress_apply(get_affected_zips, axis=1)
    
    # Explode the comma-separated ZIPs to count individual ZIP code hits
    exploded_zips = df.assign(single_zip=df['all_affected_zips'].str.split(', ')).explode('single_zip')
    
    print("\n--- MOST AFFECTED ZIP CODES (INCLUDING ENTIRE PATH) ---")
    zip_counts = exploded_zips['single_zip'].value_counts().reset_index()
    zip_counts.columns = ['ZIP_Code', 'Tornado_Hits']
    valid_zips = zip_counts[zip_counts['ZIP_Code'] != 'Unknown']
    
    print(valid_zips.head(15).to_string(index=False))

    return df, valid_zips

# Maps NOAA SPC's terse column codes to human-readable names.
# Loss/crop-loss units changed over time in NOAA's encoding; left as raw values.
NOAA_COLUMN_RENAMES = {
    "om": "tornado_id",
    "yr": "year",
    "mo": "month",
    "dy": "day",
    "date": "date",
    "time": "time",
    "tz": "time_zone",
    "st": "state",
    "stf": "state_fips",
    "stn": "state_tornado_number",
    "mag": "magnitude_ef_scale",
    "inj": "injuries",
    "fat": "fatalities",
    "loss": "property_loss_raw",
    "closs": "crop_loss_raw",
    "slat": "start_lat",
    "slon": "start_lon",
    "elat": "end_lat",
    "elon": "end_lon",
    "len": "path_length_miles",
    "wid": "path_width_yards",
    "ns": "num_states_affected",
    "sn": "state_segment_flag",
    "sg": "track_segment_flag",
    "f1": "county_fips_1",
    "f2": "county_fips_2",
    "f3": "county_fips_3",
    "f4": "county_fips_4",
    "fc": "magnitude_estimated_flag",
    "edat": "end_date",
    "etime": "end_time",
}

def generate_csv():
    os.makedirs("data/intermediate", exist_ok=True)
    recent_data = get_noaa_tornado_data(n_years=5)

    # Process the paths
    final_data, zip_hit_counts = analyze_path_and_zips(recent_data)

    # Per-tornado data: one row per tornado, with all_affected_zips as a comma-separated list
    final_data = final_data.rename(columns=NOAA_COLUMN_RENAMES)
    final_data.to_csv("data/intermediate/recent_tornado_paths_with_zips.csv", index=False)
    print("\nPer-tornado path data saved to 'recent_tornado_paths_with_zips.csv'")

    # Per-ZIP data: one row per ZIP, with the count of tornadoes that crossed it
    zip_hit_counts.to_csv("data/intermediate/tornado_hits_per_zip.csv", index=False)
    print(f"Per-ZIP hit counts ({len(zip_hit_counts)} ZIPs) saved to 'tornado_hits_per_zip.csv'")

if __name__ == "__main__":
    generate_csv()