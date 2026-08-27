import numpy as np
import pandas as pd


def analyze_crime_hypothesis():
    """
    Testet die Hypothese "Höhere Gewaltkriminalität im Bundesstaat -> niedrigere
    Immobilienpreise", kontrolliert für SES_Score, auf State-Ebene.
    """
    houses = pd.read_csv(
        "data/final/extended_dataset.csv",
        usecols=['state', 'price', 'house_size', 'SES_Score'],
    )
    houses['price_per_sqft'] = houses['price'] / houses['house_size']

    state_stats = houses.groupby('state').agg(
        price=('price', 'median'),
        price_per_sqft=('price_per_sqft', 'median'),
        SES_Score=('SES_Score', 'median'),
    ).reset_index()

    crime = pd.read_csv("data/final/crime_model.csv")
    merged = state_stats.merge(crime[['state', 'Crime_Score']], on='state', how='inner').dropna(
        subset=['price_per_sqft', 'Crime_Score', 'SES_Score']
    )

    corr = merged[['Crime_Score', 'SES_Score', 'price_per_sqft']].corr()
    print("Korrelationsmatrix (State-Ebene):")
    print(corr)

    X = np.column_stack([np.ones(len(merged)), merged['Crime_Score'], merged['SES_Score']])
    y = merged['price_per_sqft'].to_numpy()
    coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coefs
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)

    print("\nRegression: price_per_sqft ~ Crime_Score + SES_Score")
    print(f"  Intercept:        {coefs[0]:.4f}")
    print(f"  Crime_Score coef: {coefs[1]:.4f}")
    print(f"  SES_Score coef:   {coefs[2]:.4f}")
    print(f"  R^2:              {r2:.4f}")

    merged.to_csv("data/final/crime_hypothesis_results.csv", index=False)


if __name__ == "__main__":
    analyze_crime_hypothesis()
