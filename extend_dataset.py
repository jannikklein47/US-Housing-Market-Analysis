import os
import pandas as pd
from load_data import get_initial_data_path
from utilities.utils import get_zip_demographics
from utilities.national_ses import create_national_ses_model
from utilities.tornado_impact import generate_csv as generate_tornado_csv
from utilities.earthquake_impact import fetch_zip_earthquake_data

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

print("Merging datasets...")
# Merge ses model, using 'zip_code' as a common key
df = df.merge(ses_model, on='zip_code', how='left')

# Save the result
print("Saving final dataset...")
df.to_csv("data/final/extended_dataset.csv", index=False)