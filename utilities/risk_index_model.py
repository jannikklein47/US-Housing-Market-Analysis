import numpy as np
import pandas as pd

def calculate_disaster_risk_score(df_tornados, df_earthquakes, df_wildfires, df_floods):
    """
    Kombiniert vier Katastrophen-DataFrames über einen Outer Join,
    imputiert fehlende ZIP-Codes mit 0, normalisiert die Risiken
    und berechnet einen Risiko-Score von 0 (sicher) bis 100 (Hochrisiko).
    
    Annahme: Jedes DF hat die Spalten 'zip_code' und einen numerischen Wert (z.B. 'count').
    """
    # 1. DATENSÄTZE SCHÜTZEN & SPALTEN EINDEUTIG BENENNEN
    # Wir stellen sicher, dass 'zip_code' der Index ist und die Spalten eindeutig heißen.
    t = df_tornados.set_index('zip_code').iloc[:, [0]].rename(columns={df_tornados.columns[1]: 'tornados'})
    e = df_earthquakes.set_index('zip_code').iloc[:, [0]].rename(columns={df_earthquakes.columns[1]: 'earthquakes'})
    w = df_wildfires.set_index('zip_code').iloc[:, [0]].rename(columns={df_wildfires.columns[1]: 'wildfires'})
    f = df_floods.set_index('zip_code').iloc[:, [0]].rename(columns={df_floods.columns[1]: 'floods'})
    
    # 2. OUTER JOIN ÜBER ALLE KATASTROPHEN
    # pd.concat mit axis=1 und join='outer' führt alle ZIP-Codes ohne Datenverlust zusammen
    df_risk = pd.concat([t, e, w, f], axis=1, join='outer')
    
    # 3. SAFE IMPUTATION
    # Da unbetroffene ZIP-Codes fehlten, setzen wir diese nun mathematisch korrekt auf 0.
    df_risk = df_risk.fillna(0.0)
    
    # 4. DATA ENGINEERING: AUSREISSER ABFEDERN (Optional, aber dringend empfohlen)
    # Naturkatastrophen sind extrem "right-skewed" (z.B. 1000 Erdbeben in LA vs. 1 in Berlin).
    # Ein reines Min-Max würde alle moderaten Risiken gegen 0 drücken.
    # Daher nutzen wir einen Log-Transform für eine sanftere Verteilung, genau wie beim SES-Einkommen.
    log_features = {}
    for col in df_risk.columns:
        log_features[f'log_{col}'] = np.log1p(df_risk[col]) # log1p fängt log(0) ab, indem es log(x + 1) rechnet
        
    df_log = pd.DataFrame(log_features, index=df_risk.index)
    
    # 5. DYNAMISCHES MIN-MAX SCALING (0.0 bis 1.0)
    # Da wir keine festen nationalen Benchmarks haben, skalieren wir dynamisch am Maximum des Datensatzes.
    norm_features = {}
    for col in df_log.columns:
        col_min = df_log[col].min()
        col_max = df_log[col].max()
        
        # Falls ein Risiko landesweit überall 0 ist, verhindern wir eine Division durch 0
        if col_max - col_min > 0:
            norm_features[f'norm_{col}'] = (df_log[col] - col_min) / (col_max - col_min)
        else:
            norm_features[f'norm_{col}'] = 0.0
            
    df_norm = pd.DataFrame(norm_features, index=df_risk.index)
    
    # 6. RISK SCORE BERECHNEN & INDEX ZURÜCKSETZEN
    # Höherer Wert = Höheres Risiko. Daher wird hier nichts invertiert!
    df_risk['Risk_Score'] = df_norm.mean(axis=1) * 100
    
    # Kombiniere die Rohdaten mit den normierten Werten für volle Transparenz
    df_final = df_risk.join(df_norm).reset_index()
    
    return df_final