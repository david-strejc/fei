#!/usr/bin/env python3
import csv
import os
import statistics

def process_data(input_file, output_file):
    """
    Process data from a CSV file and write summary statistics to an output file.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output file for results
    """
    # Initialize data structures to store numeric data
    columns = []
    numeric_data = {}
    
    # Read the CSV file
    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Get the header row
        headers = next(reader)
        columns = headers
        
        # Initialize lists for each column
        for header in headers:
            numeric_data[header] = []
            
        # Process each row
        for row in reader:
            for i, value in enumerate(row):
                # Try to convert to float if it's a numeric column
                try:
                    numeric_value = float(value)
                    numeric_data[columns[i]].append(numeric_value)
                except ValueError:
                    # Not a numeric value, just append as string
                    numeric_data[columns[i]].append(value)
    
    # Calculate statistics for numeric columns
    results = []
    for column in columns:
        # Check if the column has numeric values
        if all(isinstance(x, (int, float)) for x in numeric_data[column]):
            values = numeric_data[column]
            column_sum = sum(values)
            column_avg = column_sum / len(values) if values else 0
            column_median = statistics.median(values) if values else 0
            
            results.append(f"Column: {column}")
            results.append(f"  Sum: {column_sum}")
            results.append(f"  Average: {column_avg:.2f}")
            results.append(f"  Median: {column_median:.2f}")
            results.append("")
    
    # Write results to the output file
    with open(output_file, 'w') as outfile:
        outfile.write("Data Processing Results\n")
        outfile.write("======================\n\n")
        outfile.write("\n".join(results))
    
    print(f"Processing complete. Results written to {output_file}")

if __name__ == "__main__":
    # Define input and output file paths
    input_file = "input.csv"
    output_file = "results_v2.txt"
    
    # Ensure we're using paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, input_file)
    output_path = os.path.join(script_dir, output_file)
    
    # Process the data
    process_data(input_path, output_path)
