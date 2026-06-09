import os
import pandas as pd
from load_data import get_initial_data_path
from utilities.utils import get_zip_demographics
from utilities.national_ses import create_national_ses_model
from utilities.risk_index_model import calculate_disaster_risk_score
from utilities.tornado_impact import generate_csv as generate_tornado_csv
from utilities.earthquake_impact import fetch_zip_earthquake_data
from utilities.flood_impact import generate_csv as generate_flood_csv
from utilities.wildfire_impact import generate_csv as generate_wildfire_csv

# 1. Read data
df = pd.read_csv(get_initial_data_path()[1])

os.makedirs("data/final", exist_ok=True)

# Get all unique zip codes
unique_zip_codes = df['zip_code'].dropna().unique()

# Create zip Lookup-Dictionary
zip_lookup = {}
for z in unique_zip_codes:
    zip_lookup[z] = get_zip_demographics(int(z))

# Convert the dictionary to a DataFrame
# Name the index axis 'zip_code' and reset it so it becomes a regular column
zip_code_df = pd.DataFrame.from_dict(zip_lookup, orient='index').rename_axis('zip_code').reset_index()

print("Generating SES Model...")
ses_model = create_national_ses_model(zip_code_df)

# Save the SES Model
ses_model.to_csv("data/final/ses_model.csv", index=False)

generate_tornado_csv()

tornado_df = pd.read_csv("data/intermediate/tornado_hits_per_zip.csv")

fetch_zip_earthquake_data()

earthquake_df = pd.read_csv("data/intermediate/zip_code_earthquakes.csv")

generate_flood_csv()

flood_df = pd.read_csv("data/intermediate/flood_claims_per_zip.csv")

generate_wildfire_csv()

wildfire_df = pd.read_csv("data/intermediate/wildfire_amount_per_zip.csv")

risk_model = calculate_disaster_risk_score(tornado_df, earthquake_df, flood_df, wildfire_df)
risk_model.to_csv("data/final/disaster_risk_model.csv", index=False)

# Save the result
print("Saving final dataset...")
#df.to_csv("data/final/extended_dataset.csv", index=False)