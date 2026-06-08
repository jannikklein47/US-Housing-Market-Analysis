from uszipcode import SearchEngine
from utilities.gini import calculate_gini_index
from utilities.college_attainment import calculate_college_attainment_rate
from utilities.rent import calculate_combined_median, calculate_combined_average

# Suchmaschine einmalig global initialisieren (schont die Performance in Schleifen)
_search_engine = SearchEngine(simple_or_comprehensive=SearchEngine.SimpleOrComprehensiveArgEnum.comprehensive)

# Definition der US-Küstenstaaten
COASTAL_STATES = {
    'ME', 'NH', 'MA', 'RI', 'CT', 'NY', 'NJ', 'DE', 'MD', 'VA', 'NC', 'SC', 'GA', 'FL', # Ostküste
    'AL', 'MS', 'LA', 'TX', # Golfküste
    'CA', 'OR', 'WA', 'AK', 'HI' # Westküste & Inseln
}

def get_zip_weighted_score(zip_code):
    """
    Berechnet einen gewichteten Score basierend auf dem ZIP-Code.
    Basis-Score ist 1. Wenn der Staat NICHT an der Küste liegt, wird er mit 0.9 multipliziert.
    """
    score = 1.0
    
    # Sicherstellen, dass der ZIP-Code ein 5-stelliger String ist
    zip_str = str(zip_code).zfill(5)
    
    # Bundesstaat ermitteln
    res = _search_engine.by_zipcode(zip_str)

    if res is not None and res.state is not None:
        state = res.state
        # Wenn der Staat NICHT in der Küstenliste ist -> Multipliziere mit 0.9
        if state not in COASTAL_STATES:
            score *= 0.9
    else:
        # Fallback, falls der ZIP-Code ungültig ist oder nicht gefunden wurde
        # Hier kannst du entscheiden, ob du 1.0, 0.0 oder None zurückgeben willst
        pass
        
    return round(score, 2)


def get_zip_demographics(zip_code):
    """
    Sucht umfassende Census-Daten für einen ZIP-Code heraus.
    """
    zip_str = str(zip_code)
    res = _search_engine.by_zipcode(zip_str)
    
    # Standardwerte, falls der ZIP-Code nicht existiert
    fallback = {
        'population': None,
        'population_density': None,
        'housing_units': None,
        'median_home_value': None,
        'median_household_income': None
    }
    
    if not res:
        print(f"Demographics: ZIP-Code {zip_str} not found.")
        return fallback
    else: print(f"Demographics: ZIP-Code {zip_str} processing.")


    result = {
        'population': res.population,
        'population_density': res.population_density,
        'housing_units': res.housing_units,
        'occupied_housing_units': res.occupied_housing_units,
        'median_home_value': res.median_home_value,
        'median_household_income': res.median_household_income,
        'college_attainment_rate': calculate_college_attainment_rate(res.educational_attainment_for_population_25_and_over),
        'gini_index': calculate_gini_index(res.household_income),
        'average_rent': calculate_combined_average(res.monthly_rent_including_utilities_1_b, res.monthly_rent_including_utilities_2_b, res.monthly_rent_including_utilities_3plus_b),
    }
        
    return result