#!/usr/bin/env python3

import pandas as pd
import os
import math

def generate_latex_table():
    # Path to the aggregated results
    csv_file = os.path.join('data', 'aggregated_results.csv')
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Create a dictionary to store the formatted results
    table_data = {}
    
    # Theoretical bounds for each algorithm
    bounds = {
        "Alg. 1": {"P": 6, "D": 10.39, "num_responses": 1},
        "Alg. 2": {"P": 5, "D": 8.81, "num_responses": 2},
        "Alg. 3": {"P": 4.08, "D": 6.95, "num_responses": 4.08},
        "Alg. 4": {"P": 3.54, "D": 9.31, "num_responses": 3.54},
        "Alg. 5": {"P": 3.83, "D": 6.72, "num_responses": 3.83},
        "Alg. 6": {"P": 3.34, "D": 6.02, "num_responses": 3.34},
        "Alg. 7": {"P": 2.93, "D": 25.8, "num_responses": 2.93},
        "Alg. 8": {"P": 2.53, "D": 45.4, "num_responses": 2.53}
    }
    
    # Find minimum bounds for each metric
    min_bounds = {
        "P": min(algo["P"] for algo in bounds.values()),
        "D": min(algo["D"] for algo in bounds.values()),
        "num_responses": min(algo["num_responses"] for algo in bounds.values())
    }
    
    # Process the dataframe to extract min, max, and avg for each algorithm and metric
    for _, row in df.iterrows():
        algorithm = f"Alg. {int(row['algorithm'])}"
        metric = row['metric']
        n_value = row['n']
        log2_n = math.log2(n_value)
        
        if algorithm not in table_data:
            table_data[algorithm] = {}
            table_data[algorithm]['n'] = n_value
            
        # Store min, max, and avg values for this metric with appropriate scaling
        if metric in ['P', 'num_responses']:
            # Scale down by log2(n)
            table_data[algorithm][f"{metric}_min"] = row['min'] / log2_n
            table_data[algorithm][f"{metric}_max"] = row['max'] / log2_n
            table_data[algorithm][f"{metric}_avg"] = row['avg'] / log2_n
            table_data[algorithm][f"{metric}_bound"] = bounds[algorithm][metric]
        elif metric == 'D':
            # Scale down by n
            table_data[algorithm][f"{metric}_min"] = row['min'] / n_value
            table_data[algorithm][f"{metric}_max"] = row['max'] / n_value
            table_data[algorithm][f"{metric}_avg"] = row['avg'] / n_value
            table_data[algorithm][f"{metric}_bound"] = bounds[algorithm][metric]
    
    # Find the minimum values for each metric to highlight
    min_values = {
        'P_min': float('inf'), 'P_max': float('inf'), 'P_avg': float('inf'),
        'D_min': float('inf'), 'D_max': float('inf'), 'D_avg': float('inf'),
        'num_responses_min': float('inf'), 'num_responses_max': float('inf'), 'num_responses_avg': float('inf')
    }
    
    for algorithm, data in table_data.items():
        for key in min_values.keys():
            if key in data and data[key] < min_values[key]:
                min_values[key] = data[key]
    
    # Generate LaTeX table header
    latex_table = [
        "\\begin{table*}[tb!]",
        "\\centering",
        "\\begin{tabular}{|l|l|rrll|rrll|rrll|}",  # Added an extra l column for categories
        "\\hline",
        "& & \\multicolumn{4}{c|}{Probes ($P/\\lceil \\log{n} \\rceil$)} & \\multicolumn{4}{c|}{Total Distance ($D/n$)} & \\multicolumn{4}{c|}{Responses ($R/\\lceil \\log{n} \\rceil$)} \\\\",
        "Category & Alg. \\# & Min & Avg & Max & Bound & Min & Avg & Max & Bound & Min & Avg & Max & Bound \\\\",
        "\\hline"
    ]
    
    # Add data rows
    current_algo = 0
    for algorithm in sorted(table_data.keys(), key=lambda x: int(x.split()[-1])):
        data = table_data[algorithm]
        algo_num = int(algorithm.split()[-1])
        row_parts = []
        
        # Add category label using multirow
        if algo_num == 1:
            row_parts.append("\\multirow{2}{*}{Hexagonal}")
        elif algo_num == 3:
            row_parts.append("\\multirow{2}{*}{Chord-Based}")
        elif algo_num == 5:
            row_parts.append("\\multirow{2}{*}{Monotonic}")
        elif algo_num == 7:
            row_parts.append("\\multirow{2}{*}{Darting}")
        else:
            row_parts.append("")  # Empty cell for second row of each pair
            
        # Add algorithm name
        row_parts.append(algorithm)
        
        # Format P values (scaled by log2(n))
        for metric in ['P_min', 'P_avg', 'P_max']:
            value = data[metric]
            formatted = f"{value:.2f}"
            if value == min_values[metric]:
                formatted = f"\\textbf{{{formatted}}}"
            row_parts.append(formatted)
        
        # Add P bound
        bound_value = data['P_bound']
        formatted_bound = f"{bound_value:.2f}"
        if bound_value == min_bounds['P']:
            formatted_bound = f"\\textbf{{{formatted_bound}}}"
        row_parts.append(formatted_bound)
        
        # Format D values (scaled by n)
        for metric in ['D_min', 'D_avg', 'D_max']:
            value = data[metric]
            formatted = f"{value:.2f}"
            if value == min_values[metric]:
                formatted = f"\\textbf{{{formatted}}}"
            row_parts.append(formatted)
        
        # Add D bound
        bound_value = data['D_bound']
        formatted_bound = f"{bound_value:.2f}"
        if bound_value == min_bounds['D']:
            formatted_bound = f"\\textbf{{{formatted_bound}}}"
        row_parts.append(formatted_bound)
        
        # Format num_responses values (scaled by log2(n))
        for metric in ['num_responses_min', 'num_responses_avg', 'num_responses_max']:
            value = data[metric]
            formatted = f"{value:.2f}"
            if value == min_values[metric]:
                formatted = f"\\textbf{{{formatted}}}"
            row_parts.append(formatted)
        
        # Add num_responses bound
        bound_value = data['num_responses_bound']
        formatted_bound = f"{bound_value:.2f}"
        if bound_value == min_bounds['num_responses']:
            formatted_bound = f"\\textbf{{{formatted_bound}}}"
        row_parts.append(formatted_bound)
        
        # Create the row
        row = " & ".join(row_parts) + " \\\\"
        latex_table.append(row)
        
        # Add horizontal lines after each pair of algorithms
        if algo_num in [2, 4, 6]:
            latex_table.append("\\hline")
    
    # Add table footer
    latex_table.extend([
        "\\hline",
        "\\end{tabular}",
        "\\caption{A numerical comparison of simulation results for our 8 algorithms on three normalized performance metrics, namely the number of probes made ($P$), the total distance traveled by the drone ($D$), and the number of hiker responses ($R$). The best values are highlighted in bold. The category names used are crude abbreviations; see the main paper for their proper names.}",
        "\\label{tab:algorithm_metrics}",
        "\\end{table*}"
    ])
    
    # Join all lines and return
    return "\n".join(latex_table)

if __name__ == "__main__":
    latex_table = generate_latex_table()
    print(latex_table)