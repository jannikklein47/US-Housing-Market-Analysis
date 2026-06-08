import numpy as np
import pandas as pd

# 1. DEFINE NATIONWIDE BENCHMARKS FOR MIN-MAX SCALING
NATIONAL_BENCHMARKS = {
    'log_median_income':  {'min': np.log(10000),  'max': np.log(250000)}, 
    'log_median_home':    {'min': np.log(30000),  'max': np.log(2000000)},
    'college_attainment': {'min': 0.0,            'max': 100.0},
    'occupancy_rate':     {'min': 0.30,           'max': 1.0},
    'gini_index':         {'min': 0.25,           'max': 0.65},
    'rent_to_income':     {'min': 0.05,           'max': 0.60}
}

# 2. NATIONAL MEDIANS FOR SAFE IMPUTATION
# If a metric is None/NaN, we substitute the national average. 
# This prevents math errors and ensures downstream models don't crash.
NATIONAL_IMPUTATION_VALUES = {
    'median_household_income': 83730.0,
    'median_home_value':       412000.0,
    'college_attainment_rate': 38.3,
    'housing_units':           1000.0, # these housing values generate a 0.897 occupancy rate as average in the usa
    'occupied_housing_units':  897.0, # these housing values generate a 0.897 occupancy rate as average in the usa
    'gini_index':              0.49,
    'average_rent':            1750.0
}

def create_national_ses_model(df_zips):
    """
    Processes a dataframe of ZIP code data, handles NaN/None safely via national
    imputation, normalizes features, and computes a nationwide comparable SES score.
    
    :param df_zips: pandas DataFrame containing raw ZIP code metrics
    :return: DataFrame with normalized features and a final 'SES_Score' (0-100)
    """
    df = df_zips.copy()
    
    # 3. TYPE COERCION & SAFE IMPUTATION LAYER
    for col, default_value in NATIONAL_IMPUTATION_VALUES.items():
        if col in df.columns:
            # Convert to numeric (turns problematic strings or None into np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill missing entries with the national baseline
            df[col] = df[col].fillna(default_value)
        else:
            # If an entire column is completely missing from the input, generate it safely
            df[col] = default_value
    
    if 'median_household_income' in df.columns:
        # If the median household income is less than $10k, set it to the national median
        df['median_household_income'] = df['median_household_income'].mask(df['median_household_income'] < 10000, NATIONAL_IMPUTATION_VALUES['median_household_income'])
            
    # 4. COMPUTE DERIVED METRICS (Guaranteed NaN-free now)
    df['occupancy_rate'] = df['occupied_housing_units'] / df['housing_units'].clip(lower=1)
    df['rent_to_income'] = (df['average_rent'] * 12) / df['median_household_income'].clip(lower=1)
    
    # Apply log transforms to squash right-skewed distributions safely
    df['log_median_income'] = np.log(df['median_household_income'].clip(lower=10000))
    df['log_median_home'] = np.log(df['median_home_value'].clip(lower=30000))
    
    # 5. NATIONWIDE MIN-MAX SCALING (0.0 to 1.0)
    norm_features = {}
    
    # Positive Drivers (Higher = Better Socioeconomic Health)
    pos_features = ['log_median_income', 'log_median_home', 'college_attainment_rate', 'occupancy_rate']
    for feat in pos_features:
        b_key = 'college_attainment' if feat == 'college_attainment_rate' else feat
        b = NATIONAL_BENCHMARKS[b_key]
        norm_features[f'norm_{feat}'] = (df[feat] - b['min']) / (b['max'] - b['min'])
        
    # Negative Drivers (Lower = Better Socioeconomic Health -> Inverted)
    neg_features = ['gini_index', 'rent_to_income']
    for feat in neg_features:
        b = NATIONAL_BENCHMARKS[feat]
        norm_features[f'norm_{feat}'] = (b['max'] - df[feat]) / (b['max'] - b['min'])
        
    # Convert normalized dictionary to DataFrame and clip bounds tightly to [0.0, 1.0]
    df_norm = pd.DataFrame(norm_features).clip(lower=0.0, upper=1.0)
    
    # Reset indexes to correctly align
    df = df.reset_index(drop=True)
    df_norm = df_norm.reset_index(drop=True)
    
    # COMPUTE NATIONAL SES SCORE
    df['SES_Score'] = df_norm.mean(axis=1) * 100

    # CHECK FOR GLITCHED ROWS
    glitched_rows = df[df['occupancy_rate'] > 1.0]
    if not glitched_rows.empty:
        print("\n--- Fehlerhafte Zeilen entdeckt! ---")
        print(glitched_rows[['occupied_housing_units', 'housing_units', 'occupancy_rate']])
    
    return pd.concat([df, df_norm], axis=1)