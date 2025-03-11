from pathlib import Path
import csv
import numpy as np
from collections import defaultdict
import multiprocessing
import os
import time

DATA_DIRECTORY = Path(__file__).parent / "data"
DATA_DIRECTORY.mkdir(exist_ok=True)

def create_metrics_dict() -> dict[str, list[int|float]]:
    """
    Factory function for defaultdict to ensure picklability.
    Returns a dict with the three metrics initialized as empty lists.
    """
    return {'P': [], 'D': [], 'num_responses': []}

def process_algorithm_file(alg_num: int) -> tuple[int, dict[float, dict[str, dict[str, float | int]]]]:
    """
    Process a single algorithm file using a simple single pass approach
    Returns a tuple of (algorithm_number, statistics)
    """
    file_path = DATA_DIRECTORY / f"algorithm_{alg_num}.csv"
    
    start_time = time.time()
    print(f"Process for algorithm {alg_num} started (PID: {os.getpid()})")
    
    # Initialize data structure to hold all values for each n
    # Using a picklable factory function instead of a lambda
    data: dict[float, dict[str, list[float]]] = defaultdict(create_metrics_dict)
    
    # Read all data from the CSV file in a single pass
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 4:  # Ensure the row has the expected format
                try:
                    n = float(row[0])
                    p = float(row[1])
                    d = float(row[2])
                    num_responses = int(row[3])
                    
                    # Store all values
                    data[n]['P'].append(p)
                    data[n]['D'].append(d)
                    data[n]['num_responses'].append(num_responses)
                except (ValueError, IndexError):
                    # Skip rows that can't be parsed
                    continue
    
    # Calculate statistics for all n values
    stats: dict[float, dict[str, dict[str, float | int]]] = {}
    
    for n, metrics in data.items():
        stats[n] = {}
        
        for metric_name, values in metrics.items():
            if not values:  # Skip empty lists
                continue
                
            values_array = np.array(values)
            count = len(values_array)
            
            stats[n][metric_name] = {
                'count': count,
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'mean': float(np.mean(values_array)),
                'std_dev': float(np.std(values_array, ddof=1)) if count > 1 else 0.0,
                'q25': float(np.percentile(values_array, 25)),
                'median': float(np.percentile(values_array, 50)),
                'q75': float(np.percentile(values_array, 75))
            }
    
    elapsed_time = time.time() - start_time
    print(f"Algorithm {alg_num} completed in {elapsed_time:.2f} seconds")
    
    return alg_num, stats

def aggregate_results():
    """
    Process all algorithm files concurrently and aggregate results
    Saves the aggregated results to a CSV file
    """
    algorithms = range(1, 9)  # Algorithms 1-8
    
    start_time_total = time.time()
    
    # Use spawn method for better cross-platform compatibility
    multiprocessing.set_start_method('spawn', force=True)
    
    # Create a pool of worker processes
    algorithm_results: dict[int, dict[float, dict[str, dict[str, float | int]]]] = {}
    
    with multiprocessing.Pool(processes=min(8, os.cpu_count() or 8)) as pool:
        # Map each algorithm to a separate process
        jobs = []
        for alg_num in algorithms:
            file_path = DATA_DIRECTORY / f"algorithm_{alg_num}.csv"
            if file_path.exists():
                jobs.append(pool.apply_async(process_algorithm_file, (alg_num,))) # type: ignore
            else:
                print(f"Warning: File {file_path} does not exist")
        
        # Wait for all jobs to complete and collect results
        for job in jobs: # type: ignore
            alg_num, stats = job.get() # type: ignore
            algorithm_results[alg_num] = stats
    
    # Prepare final results
    output_rows: list[list[int|float|str]] = []
    
    # Header row with quartile columns
    header = ['algorithm', 'n', 'metric', 'avg', 'min', 'max', 'q25', 'median', 'q75', 'std_dev', 'count']
    output_rows.append(header) # type: ignore
    
    # Process each algorithm's results
    for alg_num, n_data in algorithm_results.items():
        for n_value, metrics in n_data.items():
            for metric_name, stats in metrics.items():
                row: list[int|float|str] = [
                    alg_num,
                    n_value,
                    metric_name,
                    stats['mean'],
                    stats['min'],
                    stats['max'],
                    stats['q25'],
                    stats['median'],
                    stats['q75'],
                    stats['std_dev'],
                    stats['count']
                ]
                output_rows.append(row)
    
    # Write aggregated results to file
    output_file = DATA_DIRECTORY / "aggregated_results.csv"
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output_rows)
    
    total_time = time.time() - start_time_total
    print(f"Aggregated results saved to {output_file}")
    print(f"Total processing time: {total_time:.2f} seconds")

if __name__ == '__main__':
    aggregate_results()