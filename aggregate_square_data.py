#!/usr/bin/env python3
import csv
import glob
import math
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Any
import concurrent.futures

def _aggregate_files(args) -> dict[tuple[str, int, int, str], dict[str, Any]]:
    """Aggregate stats for a list of files (for multiprocessing, with tqdm progress)."""
    file_list, worker_id, total_workers = args
    stats_dict: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for file_path in tqdm(file_list, desc=f"Worker {worker_id+1}/{total_workers}", position=worker_id):
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                # Determine column indices for both L1 and L-infinity metrics
                col_map = {name: idx for idx, name in enumerate(header)}
                for row in reader:
                    # Defensive: skip incomplete rows
                    if len(row) < 10:
                        continue
                    algorithm = row[col_map['algorithm']]
                    dims = int(row[col_map['dims']])
                    n = int(float(row[col_map['n']]))
                    hiker_distance_l1 = float(row[col_map['hiker_distance_l1']])
                    hiker_distance_linf = float(row[col_map['hiker_distance_l_infinity']])
                    P = int(row[col_map['P']])
                    D_l1 = float(row[col_map['D_l1']])
                    D_linf = float(row[col_map['D_l_infinity']])
                    num_responses = int(row[col_map['num_responses']])
                    hiker_algorithm = row[col_map['hiker_algorithm']]

                    # L1 stats
                    key_l1 = (algorithm, dims, n, hiker_algorithm, 'l1')
                    d_hiker_ratio_l1 = D_l1 / hiker_distance_l1 if hiker_distance_l1 > 0 else 0
                    if key_l1 not in stats_dict:
                        stats_dict[key_l1] = {
                            'count': 0,
                            'P_sum': 0, 'P_sum2': 0, 'P_min': P, 'P_max': P,
                            'D_sum': 0, 'D_sum2': 0, 'D_min': D_l1, 'D_max': D_l1,
                            'num_responses_sum': 0, 'num_responses_sum2': 0, 'num_responses_min': num_responses, 'num_responses_max': num_responses,
                            'D_hiker_ratio_sum': 0, 'D_hiker_ratio_sum2': 0, 'D_hiker_ratio_min': d_hiker_ratio_l1, 'D_hiker_ratio_max': d_hiker_ratio_l1,
                            'hiker_distance_sum': 0, 'hiker_distance_sum2': 0, 'hiker_distance_min': hiker_distance_l1, 'hiker_distance_max': hiker_distance_l1,
                        }
                    s = stats_dict[key_l1]
                    s['count'] += 1
                    s['P_sum'] += P
                    s['P_sum2'] += P*P
                    s['P_min'] = min(s['P_min'], P)
                    s['P_max'] = max(s['P_max'], P)
                    s['D_sum'] += D_l1
                    s['D_sum2'] += D_l1*D_l1
                    s['D_min'] = min(s['D_min'], D_l1)
                    s['D_max'] = max(s['D_max'], D_l1)
                    s['num_responses_sum'] += num_responses
                    s['num_responses_sum2'] += num_responses*num_responses
                    s['num_responses_min'] = min(s['num_responses_min'], num_responses)
                    s['num_responses_max'] = max(s['num_responses_max'], num_responses)
                    s['D_hiker_ratio_sum'] += d_hiker_ratio_l1
                    s['D_hiker_ratio_sum2'] += d_hiker_ratio_l1*d_hiker_ratio_l1
                    s['D_hiker_ratio_min'] = min(s['D_hiker_ratio_min'], d_hiker_ratio_l1)
                    s['D_hiker_ratio_max'] = max(s['D_hiker_ratio_max'], d_hiker_ratio_l1)
                    s['hiker_distance_sum'] += hiker_distance_l1
                    s['hiker_distance_sum2'] += hiker_distance_l1*hiker_distance_l1
                    s['hiker_distance_min'] = min(s['hiker_distance_min'], hiker_distance_l1)
                    s['hiker_distance_max'] = max(s['hiker_distance_max'], hiker_distance_l1)

                    # L-infinity stats
                    key_linf = (algorithm, dims, n, hiker_algorithm, 'linf')
                    d_hiker_ratio_linf = D_linf / hiker_distance_linf if hiker_distance_linf > 0 else 0
                    if key_linf not in stats_dict:
                        stats_dict[key_linf] = {
                            'count': 0,
                            'P_sum': 0, 'P_sum2': 0, 'P_min': P, 'P_max': P,
                            'D_sum': 0, 'D_sum2': 0, 'D_min': D_linf, 'D_max': D_linf,
                            'num_responses_sum': 0, 'num_responses_sum2': 0, 'num_responses_min': num_responses, 'num_responses_max': num_responses,
                            'D_hiker_ratio_sum': 0, 'D_hiker_ratio_sum2': 0, 'D_hiker_ratio_min': d_hiker_ratio_linf, 'D_hiker_ratio_max': d_hiker_ratio_linf,
                            'hiker_distance_sum': 0, 'hiker_distance_sum2': 0, 'hiker_distance_min': hiker_distance_linf, 'hiker_distance_max': hiker_distance_linf,
                        }
                    s_inf = stats_dict[key_linf]
                    s_inf['count'] += 1
                    s_inf['P_sum'] += P
                    s_inf['P_sum2'] += P*P
                    s_inf['P_min'] = min(s_inf['P_min'], P)
                    s_inf['P_max'] = max(s_inf['P_max'], P)
                    s_inf['D_sum'] += D_linf
                    s_inf['D_sum2'] += D_linf*D_linf
                    s_inf['D_min'] = min(s_inf['D_min'], D_linf)
                    s_inf['D_max'] = max(s_inf['D_max'], D_linf)
                    s_inf['num_responses_sum'] += num_responses
                    s_inf['num_responses_sum2'] += num_responses*num_responses
                    s_inf['num_responses_min'] = min(s_inf['num_responses_min'], num_responses)
                    s_inf['num_responses_max'] = max(s_inf['num_responses_max'], num_responses)
                    s_inf['D_hiker_ratio_sum'] += d_hiker_ratio_linf
                    s_inf['D_hiker_ratio_sum2'] += d_hiker_ratio_linf*d_hiker_ratio_linf
                    s_inf['D_hiker_ratio_min'] = min(s_inf['D_hiker_ratio_min'], d_hiker_ratio_linf)
                    s_inf['D_hiker_ratio_max'] = max(s_inf['D_hiker_ratio_max'], d_hiker_ratio_linf)
                    s_inf['hiker_distance_sum'] += hiker_distance_linf
                    s_inf['hiker_distance_sum2'] += hiker_distance_linf*hiker_distance_linf
                    s_inf['hiker_distance_min'] = min(s_inf['hiker_distance_min'], hiker_distance_linf)
                    s_inf['hiker_distance_max'] = max(s_inf['hiker_distance_max'], hiker_distance_linf)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    return stats_dict

def _merge_stats_dicts(dicts: list[dict[tuple[str, int, int, str], dict[str, Any]]]) -> dict[tuple[str, int, int, str], dict[str, Any]]:
    """Merge a list of stats_dicts into one."""
    merged: dict[tuple[str, int, int, str, str], dict[str, Any]] = {}
    for d in dicts:
        for key, s in d.items():
            if key not in merged:
                merged[key] = s.copy()
            else:
                m = merged[key]
                m['count'] += s['count']
                for k in ['P_sum', 'P_sum2', 'D_sum', 'D_sum2', 'num_responses_sum', 'num_responses_sum2', 'D_hiker_ratio_sum', 'D_hiker_ratio_sum2', 'hiker_distance_sum', 'hiker_distance_sum2']:
                    m[k] += s[k]
                for k in ['P_min', 'D_min', 'num_responses_min', 'D_hiker_ratio_min', 'hiker_distance_min']:
                    m[k] = min(m[k], s[k])
                for k in ['P_max', 'D_max', 'num_responses_max', 'D_hiker_ratio_max', 'hiker_distance_max']:
                    m[k] = max(m[k], s[k])
    return merged

def aggregate_square_data(output_dir: Optional[str] = None, processes: int = 1) -> Optional[pd.DataFrame]:
    """
    Aggregate data from raw simulation files in a memory-efficient way.

    For each combination of algorithm, dims, n, and hiker_algorithm, calculate:
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
    from math import sqrt

    # Path to raw data directory
    raw_data_dir: Path = Path('raw-data')

    # Set output directory
    output_path: Path = Path('data') if output_dir is None else Path(str(output_dir))
    output_path.mkdir(exist_ok=True)

    # Find all CSV files in the raw data directory
    data_files: list[str] = glob.glob(str(raw_data_dir / 'simulations_*.csv'))

    if not data_files:
        print(f"No data files found in {raw_data_dir}")
        return None

    print(f"Found {len(data_files)} data files")

    if processes > 1:
        print(f"Using {processes} processes for aggregation...")
        chunk_size = math.ceil(len(data_files) / processes)
        file_chunks = [data_files[i:i+chunk_size] for i in range(0, len(data_files), chunk_size)]
        worker_args = [(chunk, idx, len(file_chunks)) for idx, chunk in enumerate(file_chunks)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
            dicts = list(executor.map(_aggregate_files, worker_args))
        stats_dict = _merge_stats_dicts(dicts)
    else:
        stats_dict = _aggregate_files((data_files, 0, 1))

    if not stats_dict:
        print("No valid data found in the files")
        return None

    # Build DataFrame from running stats
    rows: list[dict[str, Any]] = []
    for key, s in stats_dict.items():
        # key: (algorithm, dims, n, hiker_algorithm, metric)
        algorithm, dims, n, hiker_algorithm, metric = key
        count = s['count']
        def mean(sum_: float) -> float:
            return sum_ / count if count > 0 else float('nan')
        def std(sum_: float, sum2: float) -> float:
            if count <= 1:
                return float('nan')
            m = mean(sum_)
            return sqrt((sum2 - 2*m*sum_ + count*m*m) / (count-1))
        row: dict[str, Any] = {
            'algorithm': algorithm,
            'dims': dims,
            'n': n,
            'hiker_algorithm': hiker_algorithm,
            'metric': metric,  # 'l1' or 'linf'
            'num_simulations': count,
            'P_mean': mean(s['P_sum']),
            'P_min': s['P_min'],
            'P_max': s['P_max'],
            'P_std': std(s['P_sum'], s['P_sum2']),
            'D_mean': mean(s['D_sum']),
            'D_min': s['D_min'],
            'D_max': s['D_max'],
            'D_std': std(s['D_sum'], s['D_sum2']),
            'num_responses_mean': mean(s['num_responses_sum']),
            'num_responses_min': s['num_responses_min'],
            'num_responses_max': s['num_responses_max'],
            'num_responses_std': std(s['num_responses_sum'], s['num_responses_sum2']),
            'D_hiker_ratio_mean': mean(s['D_hiker_ratio_sum']),
            'D_hiker_ratio_min': s['D_hiker_ratio_min'],
            'D_hiker_ratio_max': s['D_hiker_ratio_max'],
            'D_hiker_ratio_std': std(s['D_hiker_ratio_sum'], s['D_hiker_ratio_sum2']),
            'hiker_distance_mean': mean(s['hiker_distance_sum']),
            'hiker_distance_min': s['hiker_distance_min'],
            'hiker_distance_max': s['hiker_distance_max'],
            'hiker_distance_std': std(s['hiker_distance_sum'], s['hiker_distance_sum2']),
        }
        rows.append(row)
    stats: pd.DataFrame = pd.DataFrame(rows)  # type: ignore

    # Save the aggregated data to a CSV file
    output_file: Path = output_path / 'aggregated_square_data.csv'
    stats.to_csv(output_file, index=False)  # type: ignore

    # Save a more readable version with fewer decimal places
    readable_stats: pd.DataFrame = stats.copy()  # type: ignore
    for col in readable_stats.columns:
        if col not in ['algorithm', 'dims', 'n', 'num_simulations', 'P_min', 'P_max', 'num_responses_min', 'num_responses_max']:
            if readable_stats[col].dtype in [np.float64, np.float32]:
                readable_stats[col] = readable_stats[col].round(2)  # type: ignore
    readable_output_file: Path = output_path / 'aggregated_square_data_readable.csv'
    readable_stats.to_csv(readable_output_file, index=False)  # type: ignore

    print(f"Aggregated data saved to {output_file}")
    print(f"Readable version saved to {readable_output_file}")

    # Print summary statistics
    print("\nSummary:")
    print(f"Total simulations processed: {stats['num_simulations'].sum()}")  # type: ignore
    print(f"Unique algorithms: {stats['algorithm'].nunique()}")  # type: ignore
    print(f"Dimensions: {sorted(stats['dims'].unique())}")  # type: ignore
    print(f"Search area sizes (n): {sorted(stats['n'].unique())}")  # type: ignore

    # Print a sample of the aggregated data
    print("\nSample of aggregated data:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(readable_stats.head())  # type: ignore

    # Generate comparison tables by dimension and metric
    print("\nGenerating comparison tables by dimension and metric...")
    for dim in sorted(stats['dims'].unique()):  # type: ignore
        for metric in ['l1', 'linf']:
            dim_stats: pd.DataFrame = stats[(stats['dims'] == dim) & (stats['metric'] == metric)].copy()  # type: ignore
            comparison_cols: list[str] = [
                'algorithm', 'hiker_algorithm', 'metric', 'num_simulations',
                'P_mean', 'P_std',
                'D_mean', 'D_std',
                'num_responses_mean', 'num_responses_std',
                'D_hiker_ratio_mean', 'D_hiker_ratio_std'
            ]
            if len(dim_stats) > 0:
                comparison = dim_stats[comparison_cols].copy()
                for col in comparison.columns:
                    if col not in ['algorithm', 'num_simulations', 'metric']:
                        if hasattr(comparison[col], 'dtype') and comparison[col].dtype in [np.float64, np.float32]:
                            comparison[col] = comparison[col].round(2)
                comparison_file: Path = output_path / f'comparison_{dim}d_{metric}.csv'
                comparison.to_csv(comparison_file, index=False)
                print(f"Comparison for {dim}D ({metric}) saved to {comparison_file}")
    print("\nData aggregation complete. To visualize the data, run visualize_square_data.py")
    return stats

def main():
    """Main function to run the data aggregation."""
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Aggregate square data from raw simulation files.')
    default_procs = max(1, (os.cpu_count() or 2) - 1)
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output files (default: "data" directory)')
    parser.add_argument('--processes', type=int, default=default_procs,
                        help=f'Number of processes to use for aggregation (default: all CPU cores minus 1, currently {default_procs})')
    args = parser.parse_args()
    aggregate_square_data(output_dir=args.output_dir, processes=args.processes)

if __name__ == "__main__":
    main()
