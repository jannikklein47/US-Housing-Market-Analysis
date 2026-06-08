import os
import pandas as pd
from load_data import get_initial_data_path
from utilities.utils import get_zip_demographics
from utilities.national_ses import create_national_ses_model

# 1. Read data
df = pd.read_csv(get_initial_data_path()[0])

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

ses_model['zip_code'] = ses_model['zip_code'].astype('Int64')
ses_model['population'] = ses_model['population'].astype('Int64')
ses_model['population_density'] = ses_model['population_density'].astype('Int64')
ses_model['housing_units'] = ses_model['housing_units'].astype('Int64')
ses_model['occupied_housing_units'] = ses_model['occupied_housing_units'].astype('Int64')
ses_model['median_home_value'] = ses_model['median_home_value'].astype('Int64')
ses_model['median_household_income'] = ses_model['median_household_income'].astype('Int64')

# Save the SES Model
ses_model.to_csv("data/final/ses_model.csv", index=False)

# Merge both dataframes, using 'zip_code' as a common key
df = df.merge(ses_model, on='zip_code', how='left')

# Save the result
df.to_csv("data/final/extended_dataset.csv", index=False)