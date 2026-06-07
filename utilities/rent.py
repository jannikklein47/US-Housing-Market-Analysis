def calculate_combined_median(one_bed_data, two_bed_data, three_plus_bed_data):
    """
    Takes three datasets representing rent distribution by bedroom count,
    aggregates their frequencies, and calculates the combined median rent 
    using grouped data linear interpolation.
    """

    # Define the mathematical intervals for each string bin label
    # Note: For '$1,000+', an upper limit bound assumption is set to $1,500
    bin_intervals = {
        '< $200': (0, 200),
        '$200-$299': (200, 300),
        '$300-$499': (300, 500),
        '$500-$749': (500, 750),
        '$750-$999': (750, 1000),
        '$1,000+': (1000, 1500)
    }
    
    # Initialize a dictionary to accumulate counts for each rent bin
    aggregated_counts = {bin_label: 0 for bin_label in bin_intervals}
    
    # Combine the frequency 'y' for each matching 'x' interval label
    for dataset in [one_bed_data, two_bed_data, three_plus_bed_data]:
        if dataset and isinstance(dataset, list) and len(dataset) > 0:
            values_list = dataset[0].get('values', [])
            for item in values_list:
                label = item.get('x')
                frequency = item.get('y', 0)
                if label in aggregated_counts:
                    aggregated_counts[label] += frequency
                    
    # Map aggregated data into a list of tuples ordered by the interval lower bound:
    # (lower_bound, upper_bound, total_frequency)
    ordered_bins = []
    for label, total_f in aggregated_counts.items():
        lower, upper = bin_intervals[label]
        ordered_bins.append((lower, upper, total_f))
    
    # Sort bins to ensure progressive cumulative evaluation
    ordered_bins.sort(key=lambda b: b[0])
    
    # Find the total frequency across all combined categories
    total_frequency = sum(b[2] for b in ordered_bins)
    if total_frequency == 0:
        return 0
        
    # The target position for the 50th percentile (median)
    target = total_frequency / 2
    cumulative_frequency = 0
    
    # Find the specific interval where the median target resides
    for lower_bound, upper_bound, f in ordered_bins:
        if cumulative_frequency + f >= target:
            # Apply standard linear interpolation formula for binned data:
            # Median = L + ((N/2 - CF) / f) * W
            L = lower_bound
            CF = cumulative_frequency
            W = upper_bound - lower_bound
            
            if f == 0:
                return L
            return L + ((target - CF) / f) * W
            
        cumulative_frequency += f
        
    return None

def calculate_combined_average(one_bed_data, two_bed_data, three_plus_bed_data):
    """
    Takes three datasets representing rent distribution by bedroom count,
    aggregates their frequencies, and calculates the overall combined average.
    """
    # Define estimated midpoints for calculating the average
    # Note: For '$1,000+', an upper limit bound assumption is set to $1,500 ($1250 midpoint)
    bin_midpoints = {
        '< $200': 100,
        '$200-$299': 250,
        '$300-$499': 400,
        '$500-$749': 625,
        '$750-$999': 875,
        '$1,000+': 1250 
    }
    
    total_frequency = 0
    weighted_sum = 0
    
    # Process each of the three bedroom datasets
    for dataset in [one_bed_data, two_bed_data, three_plus_bed_data]:
        if dataset and isinstance(dataset, list) and len(dataset) > 0:
            values_list = dataset[0].get('values', [])
            for item in values_list:
                label = item.get('x')
                frequency = item.get('y', 0)
                
                # If the label exists in our midpoints, add to the tracking totals
                if label in bin_midpoints:
                    weighted_sum += bin_midpoints[label] * frequency
                    total_frequency += frequency
                    
    # Handle division by zero edge case if all inputs are empty or have 0 frequencies
    if total_frequency == 0:
        return None
        
    return weighted_sum / total_frequency