import pandas as pd
from load_data import get_initial_data_path
from utilities.utils import filter_us_states, convert_imperial_to_metric

def export_final_csv():
    houses = pd.read_csv(get_initial_data_path()[1])
    houses = filter_us_states(houses)
    houses = convert_imperial_to_metric(houses)

    houses["total_size_sqm"] = houses["house_size_sqm"].fillna(0) + houses[
        "lot_size_sqm"
    ].fillna(0)

    houses["price_per_sqm"] = houses["price"] / houses["total_size_sqm"]

    disaster_risk = pd.read_csv("data/final/disaster_risk_model.csv")
    disaster_risk = disaster_risk[~disaster_risk['zip_code'].str.contains('-', na=False)]
    disaster_risk["zip_code"] = disaster_risk["zip_code"].astype("float64")
    ses = pd.read_csv("data/final/ses_model.csv")
    crime = pd.read_csv("data/final/crime_model.csv")

    # Check for risk score duplicates
    print("Disaster Risk dups:", disaster_risk['zip_code'].duplicated().any())
    disaster_risk = disaster_risk.drop_duplicates(subset=['zip_code'])
    print("Disaster Risk dups:", disaster_risk['zip_code'].duplicated().any())

    final = houses.merge(disaster_risk[['zip_code', 'Risk_Score']], on='zip_code', how='left')
    final = final.merge(ses[['zip_code', 'SES_Score', 'rent_to_income', 'occupancy_rate']], on='zip_code', how='left')
    final = final.merge(crime[['state', 'Crime_Score']], on='state', how='left')

    # Data Cleaning
    final["zip_code"] = final["zip_code"].round().astype("Int64")
    final["price"] = final["price"].round().astype("Int64")

    final["brokered_by"] = final["brokered_by"].round().astype("Int64")
    final["bed"] = final["bed"].round().astype("Int64")
    final["bath"] = final["bath"].round().astype("Int64")
    final["street"] = final["street"].round().astype("Int64")

    final["house_size_sqm"] = final["house_size_sqm"].map("{:.2f}".format)
    final["lot_size_sqm"] = final["lot_size_sqm"].map("{:.2f}".format)
    final["total_size_sqm"] = final["total_size_sqm"].map("{:.2f}".format)
    final["Risk_Score"] = final["Risk_Score"].map("{:.2f}".format)
    final["SES_Score"] = final["SES_Score"].map("{:.2f}".format)
    final["rent_to_income"] = final["rent_to_income"].map("{:.3f}".format)
    final["occupancy_rate"] = final["occupancy_rate"].map("{:.3f}".format)
    final["price_per_sqm"] = final["price_per_sqm"].map("{:.3f}".format)
    final["Crime_Score"] = final["Crime_Score"].map("{:.2f}".format)

    # 1. Drop rows missing any required single fields
    required_cols = [
        "price",
        "zip_code",
        "Risk_Score",
        "SES_Score",
        "occupancy_rate",
        "Crime_Score",
    ]
    final = final.dropna(subset=required_cols)

    # 2. Keep rows where at least one of house_size_sqm or lot_size_sqm is not missing
    final = final[final["house_size_sqm"].notna() | final["lot_size_sqm"].notna()]

    import numpy as np

    # Replace string "nan", "NaN", "None", or empty spaces with actual NaN
    final = final.replace(["nan", "NaN", "None", ""], np.nan)

    # Now export with na_rep=""
    final.to_csv("output.csv", na_rep="", index=False)
    final.to_csv("data/final/final_dataset_2.csv", na_rep="", index=False)

    # print the amount of rows
    print(final.shape)

if __name__ == "__main__":
    export_final_csv()