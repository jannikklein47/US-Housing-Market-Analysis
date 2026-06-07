import pandas as pd
import datetime
import urllib.error
import numpy as np
from uszipcode import SearchEngine
from tqdm import tqdm

tqdm.pandas()

def get_noaa_tornado_data(n_years: int = 5):
    """
    Downloads historical tornado data from the NOAA Storm Prediction Center
    and filters it for the last `n_years`.
    """
    current_year = datetime.datetime.now().year
    
    # The URL updates annually; try the last few years to find the most recent dataset
    df = None
    for year in range(current_year, 2020, -1):
        url = f"https://www.spc.noaa.gov/wcm/data/1950-{year}_actual_tornadoes.csv"
        try:
            print(f"Trying to fetch data from: {url}")
            df = pd.read_csv(url)
            print(f"Successfully loaded dataset up to {year}!")
            latest_year_in_data = year
            break
        except (urllib.error.HTTPError, urllib.error.URLError, Exception):
            continue
            
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
    
    return df

if __name__ == "__main__":
    # Example: Get data for the last 5 years to keep processing time reasonable
    recent_data = get_noaa_tornado_data(n_years=5)
    
    # Process the paths
    final_data = analyze_path_and_zips(recent_data)
    
    # Save the complete results to a local CSV for your own use
    # You will now have an 'all_affected_zips' column with every ZIP the tornado crossed
    final_data.to_csv("recent_tornado_paths_with_zips.csv", index=False)
    print("\nFull path data saved to 'recent_tornado_paths_with_zips.csv'")