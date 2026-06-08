import requests
import pandas as pd
from datetime import datetime
from collections import Counter
from uszipcode import SearchEngine
import os

# ==================== CONFIGURATION ====================
TIMESPAN_YEARS = 20   # Standard value: 20 years
MIN_MAGNITUDE = 3.0    # Earthquakes smaller than 3.0 are rarely felt
OUTPUT_FILE = "data/intermediate/zip_code_earthquakes.csv"
# =======================================================

def fetch_zip_earthquake_data():
    current_year = datetime.now().year
    start_year = current_year - TIMESPAN_YEARS
    
    print(f"Initializing US ZIP Code search engine...")
    search = SearchEngine()
    zip_counts = Counter()
    
    print(f"Fetching earthquake data from {start_year} to {current_year}...")
    
    # We loop year-by-year because the USGS API caps single requests at 20,000 events
    for year in range(start_year, current_year + 1):
        start_time = f"{year}-01-01"
        end_time = f"{year}-12-31"
        
        # USGS API endpoint for earthquake events
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        
        # Bounding box roughly covering the United States (including AK and HI)
        params = {
            "format": "geojson",
            "starttime": start_time,
            "endtime": end_time,
            "minmagnitude": MIN_MAGNITUDE,
            "minlatitude": 18.0,
            "maxlatitude": 72.0,
            "minlongitude": -180.0,
            "maxlongitude": -65.0
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code != 200:
                print(f"  [Warning] Failed to fetch data for year {year} (Status: {response.status_code})")
                continue
                
            data = response.json()
            events = data.get("features", [])
            print(f"  Year {year}: Processing {len(events)} events...")
            
            for event in events:
                coords = event["geometry"]["coordinates"]
                lon, lat = coords[0], coords[1]
                
                # Find if this coordinate falls within or near (radius in miles) a US ZIP code
                # A 15-mile radius accounts for regional shaking impacts
                results = search.by_coordinates(lat, lon, radius=15)
                if results:
                    # Attribute the event to the closest matching ZIP code
                    closest_zip = results[0].zipcode
                    zip_counts[closest_zip] += 1
                    
        except Exception as e:
            print(f"  [Error] Could not process year {year}: {e}")
            continue

    # Convert the results into a structured Pandas DataFrame
    df = pd.DataFrame(zip_counts.items(), columns=["zip_code", "earthquake_amount"])
    
    # Sort by the highest number of earthquakes first
    df = df.sort_values(by="earthquake_amount", ascending=False).reset_index(drop=True)
    
    # Save to CSV
    os.makedirs("data/intermediate", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccess! Data exported to '{OUTPUT_FILE}'.")
    print(f"Total unique ZIP codes flagged: {len(df)}")

if __name__ == "__main__":
    fetch_zip_earthquake_data()