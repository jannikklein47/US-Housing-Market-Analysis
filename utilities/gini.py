import re

def calculate_gini_index(data, total_population=None, top_bin_multiplier=1.25):
    """
    Calculates the Gini Index from grouped income category data.
    
    :param data: List containing dictionaries with a 'values' key holding 'x' and 'y' pairs.
    :param total_population: Optional preset total population. If None, it computes from data.
    :param top_bin_multiplier: Multiplier for the open-ended top bracket to estimate its mean.
    :return: Gini Index as a float between 0 and 1.
    """

    if data is None: return None
    
    def parse_income_bracket(x_str):
        # Remove currency signs, commas, and spaces, and convert to lowercase
        clean = re.sub(r'[\$,\s]', '', x_str).lower()
        
        # 1. Handle open-ended bottom brackets (e.g., "<25000", "under25000")
        if clean.startswith('<') or 'under' in clean or 'less' in clean:
            digits = re.findall(r'\d+', clean)
            if digits:
                return float(digits[0]) / 2  # Midpoint between 0 and upper bound
        
        # 2. Handle open-ended top brackets (e.g., "200000+", "over200000")
        if '+' in clean or 'over' in clean or 'more' in clean or clean.startswith('>'):
            digits = re.findall(r'\d+', clean)
            if digits:
                return float(digits[0]) * top_bin_multiplier
        
        # 3. Handle standard bounded ranges (e.g., "25000-44999")
        if '-' in clean:
            digits = re.findall(r'\d+', clean)
            if len(digits) == 2:
                return (float(digits[0]) + float(digits[1])) / 2
                
        # 4. Handle exact single numbers
        digits = re.findall(r'\d+', clean)
        if digits:
            return float(digits[0])
            
        return 0.0

    # Extract the values array from the data structure
    raw_values = []

    for item in data:
        if 'values' in item:
            raw_values.extend(item['values'])
            
    # Process brackets into a list of tuples: (estimated_income, population_count)
    income_bins = []
    for entry in raw_values:
        income_val = parse_income_bracket(entry['x'])
        population_count = entry['y']
        income_bins.append((income_val, population_count))
        
    # Sort bins in ascending order by income amount
    income_bins.sort(key=lambda x: x[0])
    
    # Determine the total population and total income
    calculated_pop = sum(b[1] for b in income_bins)
    if total_population is None:
        total_population = calculated_pop
        
    total_income = sum(b[0] * b[1] for b in income_bins)
    
    # Edge case: No population or no income
    if total_population == 0 or total_income == 0:
        return 0.0
        
    # Calculate Gini via the Area under the Lorenz Curve (Trapezoidal Rule)
    cum_inc_frac = 0.0
    lorenz_area = 0.0
    
    for income, pop in income_bins:
        pop_frac = pop / total_population
        inc_frac = (income * pop) / total_income
        
        prev_cum_inc_frac = cum_inc_frac
        cum_inc_frac += inc_frac
        
        # Add the trapezoidal area segment
        lorenz_area += pop_frac * (prev_cum_inc_frac + cum_inc_frac) / 2
        
    # Gini coefficient is 1 - 2 * (Area under Lorenz Curve)
    gini_index = 1 - 2 * lorenz_area
    return gini_index
