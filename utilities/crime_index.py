import numpy as np
import pandas as pd


def calculate_crime_index(df_crime: pd.DataFrame, known_states: list) -> pd.DataFrame:
    """
    Berechnet einen Crime_Score (0-100, höher = mehr Gewaltkriminalität) je Bundesstaat
    aus Murder/Assault/Rape (log1p + dynamisches Min-Max-Scaling + Mittelwert, analog zum
    Risk_Score in risk_index_model.py, da hier ebenfalls die volle Population der 50
    US-Staaten vorliegt). UrbanPop ist kein Kriminalitätsmaß und wird nur als Rohspalte
    durchgereicht. Staaten aus known_states ohne Treffer werden mit dem nationalen
    Crime_Score-Mittelwert imputiert.
    """
    df = df_crime.rename(columns={df_crime.columns[0]: 'state'}).copy()
    df['state'] = df['state'].astype(str).str.strip().str.strip('"')

    crime_cols = ['Murder', 'Assault', 'Rape']
    df_log = df[crime_cols].apply(np.log1p)

    norm_features = {}
    for col in crime_cols:
        col_min, col_max = df_log[col].min(), df_log[col].max()
        if col_max - col_min > 0:
            norm_features[f'norm_{col}'] = (df_log[col] - col_min) / (col_max - col_min)
        else:
            norm_features[f'norm_{col}'] = 0.0
    df_norm = pd.DataFrame(norm_features, index=df.index)

    df['Crime_Score'] = df_norm.mean(axis=1) * 100

    national_mean_score = df['Crime_Score'].mean()

    result = pd.DataFrame({'state': known_states}).merge(df, on='state', how='left')
    result['Crime_Score'] = result['Crime_Score'].fillna(national_mean_score)

    return result
