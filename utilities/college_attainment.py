def calculate_college_attainment_rate(data):
    """
    Calculates the college attainment rate (Bachelor's degree or higher)
    from a demographic education dataset.
    
    :param data: List containing dictionaries with a 'values' key holding 'x' and 'y' pairs.
    :return: Attainment rate as a percentage (0.0 to 100.0).
    """
    if data is None: return None
    
    # Extract the individual category records
    raw_values = []
    for item in data:
        if 'values' in item:
            raw_values.extend(item['values'])
            
    total_population = 0
    college_educated_population = 0
    
    # Keywords indicating a Bachelor's degree or higher
    target_keywords = ['bachelor', 'master', 'professional', 'doctorate', 'phd']
    
    for entry in raw_values:
        count = entry['y']
        category_name = entry['x'].lower()
        
        # Track total population over 25
        total_population += count
        
        # Check if the category matches college attainment
        if any(keyword in category_name for keyword in target_keywords):
            college_educated_population += count
            
    # Handle edge case for empty datasets
    if total_population == 0:
        return 0.0
        
    # Calculate percentage
    attainment_rate = (college_educated_population / total_population) * 100
    return attainment_rate