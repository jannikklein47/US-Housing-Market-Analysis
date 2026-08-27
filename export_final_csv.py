import pandas as pd
from load_data import get_initial_data_path

def export_final_csv():
    houses = pd.read_csv(get_initial_data_path()[1])
    disaster_risk = pd.read_csv("data/final/disaster_risk_model.csv")
    disaster_risk = disaster_risk[~disaster_risk['zip_code'].str.contains('-', na=False)]
    disaster_risk["zip_code"] = disaster_risk["zip_code"].astype("float64")
    ses = pd.read_csv("data/final/ses_model.csv")

    # Check for risk score duplicates
    print("Disaster Risk dups:", disaster_risk['zip_code'].duplicated().any())
    disaster_risk = disaster_risk.drop_duplicates(subset=['zip_code'])
    print("Disaster Risk dups:", disaster_risk['zip_code'].duplicated().any())

    final = houses.merge(disaster_risk[['zip_code', 'Risk_Score']], on='zip_code', how='left')
    final = final.merge(ses[['zip_code', 'SES_Score', 'rent_to_income']], on='zip_code', how='left')

    final.to_csv("data/final/final_dataset.csv", index=False)

if __name__ == "__main__":
    export_final_csv()