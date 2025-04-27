#!/usr/bin/env python3
import os
import csv
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

def aggregate_square_data(output_dir=None):
    """
    Aggregate data from raw simulation files.

    For each combination of algorithm, dims, and n, calculate:
    - Number of simulations
    - Average/min/max/std of P (number of probes)
    - Average/min/max/std of D (distance traveled)
    - Average/min/max/std of num_responses
    - Average/min/max/std of D / hiker_distance

    Args:
        output_dir: Directory to save output files (default: 'data' directory)

    Returns:
        DataFrame containing the aggregated statistics
    """
    # Path to raw data directory
    raw_data_dir = Path('raw-data')

    # Set output directory
    if output_dir is None:
        output_dir = Path('data')
    else:
        output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Find all CSV files in the raw data directory
    data_files = glob.glob(str(raw_data_dir / 'simulations_*.csv'))

    if not data_files:
        print(f"No data files found in {raw_data_dir}")
        return None

    print(f"Found {len(data_files)} data files")

    # Create a DataFrame to store all data
    all_data = []

    # Read and process each data file
    for file_path in data_files:
        try:
            # Read the CSV file
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header

                # Process each row
                for row in reader:
                    if len(row) >= 7:  # Ensure row has all required fields
                        algorithm, dims, n, hiker_distance, P, D, num_responses = row[:7]

                        # Convert to appropriate types
                        dims = int(dims)
                        n = int(float(n))  # Handle scientific notation
                        hiker_distance = float(hiker_distance)
                        P = int(P)
                        D = float(D)
                        num_responses = int(num_responses)

                        # Calculate D / hiker_distance
                        d_hiker_ratio = D / hiker_distance if hiker_distance > 0 else 0

                        # Add to data list
                        all_data.append({
                            'algorithm': algorithm,
                            'dims': dims,
                            'n': n,
                            'hiker_distance': hiker_distance,
                            'P': P,
                            'D': D,
                            'num_responses': num_responses,
                            'D_hiker_ratio': d_hiker_ratio
                        })
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if not all_data:
        print("No valid data found in the files")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Group by algorithm, dims, and n
    grouped = df.groupby(['algorithm', 'dims', 'n'])

    # Calculate statistics for each group
    stats = grouped.agg({
        'P': ['count', 'mean', 'min', 'max', 'std'],
        'D': ['mean', 'min', 'max', 'std'],
        'num_responses': ['mean', 'min', 'max', 'std'],
        'D_hiker_ratio': ['mean', 'min', 'max', 'std'],
        'hiker_distance': ['mean', 'min', 'max', 'std']  # Added hiker_distance stats
    })

    # Flatten the multi-level column names
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]

    # Rename the count column to num_simulations
    stats = stats.rename(columns={'P_count': 'num_simulations'})

    # Reset index to make algorithm, dims, and n regular columns
    stats = stats.reset_index()

    # Save the aggregated data to a CSV file
    output_file = output_dir / 'aggregated_square_data.csv'
    stats.to_csv(output_file, index=False)

    # Save a more readable version with fewer decimal places
    readable_stats = stats.copy()
    for col in readable_stats.columns:
        if col not in ['algorithm', 'dims', 'n', 'num_simulations', 'P_min', 'P_max', 'num_responses_min', 'num_responses_max']:
            if readable_stats[col].dtype in [np.float64, np.float32]:
                readable_stats[col] = readable_stats[col].round(2)

    readable_output_file = output_dir / 'aggregated_square_data_readable.csv'
    readable_stats.to_csv(readable_output_file, index=False)

    print(f"Aggregated data saved to {output_file}")
    print(f"Readable version saved to {readable_output_file}")

    # Print summary statistics
    print("\nSummary:")
    print(f"Total simulations processed: {len(df)}")
    print(f"Unique algorithms: {df['algorithm'].nunique()}")
    print(f"Dimensions: {sorted(df['dims'].unique())}")
    print(f"Search area sizes (n): {sorted(df['n'].unique())}")

    # Print a sample of the aggregated data
    print("\nSample of aggregated data:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(readable_stats.head())

    # Generate comparison tables by dimension
    print("\nGenerating comparison tables by dimension...")
    for dim in sorted(df['dims'].unique()):
        dim_stats = stats[stats['dims'] == dim].copy()

        # Create a more readable comparison table
        comparison_cols = [
            'algorithm', 'num_simulations',
            'P_mean', 'P_std',
            'D_mean', 'D_std',
            'num_responses_mean', 'num_responses_std',
            'D_hiker_ratio_mean', 'D_hiker_ratio_std'
        ]

        if len(dim_stats) > 0:
            comparison = dim_stats[comparison_cols].copy()
            for col in comparison.columns:
                if col not in ['algorithm', 'num_simulations']:
                    if comparison[col].dtype in [np.float64, np.float32]:
                        comparison[col] = comparison[col].round(2)

            comparison_file = output_dir / f'comparison_{dim}d.csv'
            comparison.to_csv(comparison_file, index=False)
            print(f"Comparison for {dim}D saved to {comparison_file}")

    print("\nData aggregation complete. To visualize the data, run visualize_square_data.py")
    return stats

def main():
    """Main function to run the data aggregation."""
    import argparse

    parser = argparse.ArgumentParser(description='Aggregate square data from raw simulation files.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output files (default: "data" directory)')

    args = parser.parse_args()

    aggregate_square_data(output_dir=args.output_dir)

if __name__ == "__main__":
    main()
