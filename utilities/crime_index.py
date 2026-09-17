import os
import time

import numpy as np
import pandas as pd
import requests

API_BASE = "https://api.usa.gov/crime/fbi/cde/summarized/state"
OFFENSES = ['homicide', 'rape', 'robbery', 'aggravated-assault']
YEAR = 2023
CACHE_PATH = "data/intermediate/crime_rates_per_state.csv"

# Volle Staatsnamen (wie in der 'state'-Spalte des Housing-Datensatzes) -> 2-Buchstaben-Kuerzel,
# das die FBI Crime Data API als Pfadparameter erwartet.
STATE_ABBR = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
}


def _load_api_key(env_path=".env"):
    if "FBI_API_KEY" in os.environ:
        return os.environ["FBI_API_KEY"]
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("FBI_API_KEY="):
                    return line.strip().split("=", 1)[1]
    raise RuntimeError("FBI_API_KEY nicht gefunden (weder als Env-Var noch in .env)")


def _fetch_state_offense_rate(state_name, offense, api_key, session, year=YEAR):
    """Holt die Jahresrate (pro 100.000 Einwohner) fuer einen Bundesstaat und eine Offense-Kategorie."""
    abbr = STATE_ABBR[state_name]
    url = f"{API_BASE}/{abbr}/{offense}"
    params = {"api_key": api_key, "from": f"01-{year}", "to": f"12-{year}"}
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    monthly_rates = resp.json()["offenses"]["rates"].get(f"{state_name} Offenses", {})
    if not monthly_rates:
        return None
    return float(np.mean(list(monthly_rates.values())))


def fetch_crime_rates(known_states) -> pd.DataFrame:
    """Holt fuer jeden bekannten US-Bundesstaat die Jahresraten je Offense-Kategorie von der FBI Crime Data API."""
    os.makedirs("data/intermediate", exist_ok=True)

    if os.path.exists(CACHE_PATH):
        print(f"Crime-Raten bereits gecacht, lade von {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH)

    api_key = _load_api_key()
    session = requests.Session()

    rows = []
    for state in known_states:
        if state not in STATE_ABBR:
            continue
        row = {"state": state}
        for offense in OFFENSES:
            row[offense] = _fetch_state_offense_rate(state, offense, api_key, session)
            time.sleep(0.2)
        rows.append(row)
        print(f"Crime-Raten geladen: {state}")

    df = pd.DataFrame(rows)
    df.to_csv(CACHE_PATH, index=False)
    return df


def calculate_crime_index(df_rates: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet einen Crime_Score (0-100, hoeher = mehr Gewaltkriminalitaet) je Bundesstaat aus den
    Jahresraten (log1p + dynamisches Min-Max-Scaling + Mittelwert), analog zum Risk_Score in
    risk_index_model.py, da hier ebenfalls die volle Population der abgefragten Staaten vorliegt.
    """
    df = df_rates.copy()
    df_log = df[OFFENSES].apply(np.log1p)

    norm_features = {}
    for col in OFFENSES:
        col_min, col_max = df_log[col].min(), df_log[col].max()
        if col_max - col_min > 0:
            norm_features[f'norm_{col}'] = (df_log[col] - col_min) / (col_max - col_min)
        else:
            norm_features[f'norm_{col}'] = 0.0
    df_norm = pd.DataFrame(norm_features, index=df.index)

    df['Crime_Score'] = df_norm.mean(axis=1) * 100
    return df
