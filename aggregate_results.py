from pathlib import Path
import csv
import math
from collections import defaultdict
import multiprocessing
import os
import time

DATA_DIRECTORY = Path(__file__).parent / "data"
DATA_DIRECTORY.mkdir(exist_ok=True)

def create_default_stats() -> dict[str,dict[str,int|float]]:
    """
    Creates the default stats dictionary structure.
    This replaces the lambda function to ensure proper pickling.
    """
    return {
        'P': {'count': 0, 'sum': 0, 'min': float('inf'), 'max': float('-inf')},
        'D': {'count': 0, 'sum': 0, 'min': float('inf'), 'max': float('-inf')},
        'num_responses': {'count': 0, 'sum': 0, 'min': float('inf'), 'max': float('-inf')}
    }

def csv_row_generator(file_path: Path):
    """
    Generator that yields rows from a CSV file
    This avoids loading the entire file into memory
    """
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 4:  # Ensure the row has the expected format
                try:
                    yield (float(row[0]), float(row[1]), float(row[2]), int(row[3]))
                except (ValueError, IndexError):
                    # Skip rows that can't be parsed
                    continue

def first_pass_statistics(file_path: Path):
    """
    First pass to calculate count, sum, min, max for each n value
    Returns a dictionary with stats grouped by n value
    """
    stats_by_n: dict[float,dict[str,dict[str,int|float]]] = defaultdict(create_default_stats)
    
    for n, p, d, num_responses in csv_row_generator(file_path):
        # Update P stats
        stats_by_n[n]['P']['count'] += 1
        stats_by_n[n]['P']['sum'] += p
        stats_by_n[n]['P']['min'] = min(stats_by_n[n]['P']['min'], p)
        stats_by_n[n]['P']['max'] = max(stats_by_n[n]['P']['max'], p)
        
        # Update D stats
        stats_by_n[n]['D']['count'] += 1
        stats_by_n[n]['D']['sum'] += d
        stats_by_n[n]['D']['min'] = min(stats_by_n[n]['D']['min'], d)
        stats_by_n[n]['D']['max'] = max(stats_by_n[n]['D']['max'], d)
        
        # Update num_responses stats
        stats_by_n[n]['num_responses']['count'] += 1
        stats_by_n[n]['num_responses']['sum'] += num_responses
        stats_by_n[n]['num_responses']['min'] = min(stats_by_n[n]['num_responses']['min'], num_responses)
        stats_by_n[n]['num_responses']['max'] = max(stats_by_n[n]['num_responses']['max'], num_responses)
    
    # Calculate means
    for n_stats in stats_by_n.values():
        for metric_stats in n_stats.values():
            if metric_stats['count'] > 0:
                metric_stats['mean'] = metric_stats['sum'] / metric_stats['count']
            else:
                metric_stats['mean'] = 0
    
    return stats_by_n

def second_pass_statistics(file_path: Path, first_pass_results: dict[float,dict[str,dict[str,int|float]]]):
    """
    Second pass to calculate standard deviation for each n value
    Updates the first_pass_results dictionary with std_dev
    """
    # Initialize sum of squared differences
    for n_stats in first_pass_results.values():
        for metric_stats in n_stats.values():
            metric_stats['sum_squared_diff'] = 0
    
    # Calculate sum of squared differences from the mean
    for n, p, d, num_responses in csv_row_generator(file_path):
        if n in first_pass_results:
            first_pass_results[n]['P']['sum_squared_diff'] += (p - first_pass_results[n]['P']['mean']) ** 2
            first_pass_results[n]['D']['sum_squared_diff'] += (d - first_pass_results[n]['D']['mean']) ** 2
            first_pass_results[n]['num_responses']['sum_squared_diff'] += (num_responses - first_pass_results[n]['num_responses']['mean']) ** 2
    
    # Calculate standard deviations
    for n_stats in first_pass_results.values():
        for metric_stats in n_stats.values():
            if metric_stats['count'] > 1:
                metric_stats['std_dev'] = math.sqrt(metric_stats['sum_squared_diff'] / (metric_stats['count'] - 1))
            else:
                metric_stats['std_dev'] = 0
            
            # Clean up temporary values
            del metric_stats['sum_squared_diff']
            del metric_stats['sum']
    
    return first_pass_results

def process_algorithm_file(alg_num: int) -> tuple[int, dict[float,dict[str,dict[str,int|float]]]]:
    """
    Process a single algorithm file using two passes to conserve memory
    Returns a tuple of (algorithm_number, statistics)
    """
    file_path = DATA_DIRECTORY / f"algorithm_{alg_num}.csv"
    
    start_time = time.time()
    print(f"Process for algorithm {alg_num} started (PID: {os.getpid()})")
    
    print(f"First pass on {file_path.name}...")
    first_pass_results = first_pass_statistics(file_path)
    
    print(f"Second pass on {file_path.name}...")
    complete_stats = second_pass_statistics(file_path, first_pass_results)
    
    elapsed_time = time.time() - start_time
    print(f"Algorithm {alg_num} completed in {elapsed_time:.2f} seconds")
    
    return alg_num, complete_stats

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
    algorithm_results: dict[int,dict[float,dict[str,dict[str,int|float]]]] = {}
    
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
    output_rows: list[list[int | float | str]] = []
    
    # Header row
    header = ['algorithm', 'n', 'metric', 'avg', 'min', 'max', 'std_dev', 'count']
    output_rows.append(header) # type: ignore
    
    # Process each algorithm's results
    for alg_num, n_data in algorithm_results.items():
        for n_value, metrics in n_data.items():
            for metric_name, stats in metrics.items():
                row: list[int | float | str] = [
                    alg_num,
                    n_value,
                    metric_name,
                    stats['mean'],
                    stats['min'],
                    stats['max'],
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