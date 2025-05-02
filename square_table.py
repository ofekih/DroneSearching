#!/usr/bin/env python3

from __future__ import annotations

from enum import Enum, auto
import pandas as pd
import os
import math
from typing import Union, Any

class Algorithm(Enum):
	Domino = auto()
	Orthant = auto()
	GeneralizedCBS = auto()

	def __str__(self):
		match self:
			case Algorithm.Domino:
				return "Domino Algorithms"
			case Algorithm.Orthant:
				return "Orthant Algorithm"
			case Algorithm.GeneralizedCBS:
				return "Generalized CBS Algorithm"

	@classmethod
	def from_string(cls, algorithm_str: str) -> Algorithm | None:
		"""Convert algorithm string to enum value"""
		mapping = {
			'simple_hypercube_search': cls.Orthant,
			'central_binary_search': cls.GeneralizedCBS,
			'domino_2d_search': cls.Domino,
			'domino_3d_search': cls.Domino
		}
		return mapping.get(algorithm_str)

class Metric(Enum):
	Probes = "P"
	Distance = "D"
	Responses = "R"

	def __str__(self):
		return self.value

def load_aggregated_square_data() -> pd.DataFrame:
	"""
	Load the aggregated square data from the CSV file

	Returns:
		DataFrame containing the filtered data
	"""
	csv_file = os.path.join('data', 'aggregated_square_data.csv')

	# Read the CSV file
	df = pd.read_csv(csv_file)

	# Filter for L_infinity metric only
	df = df[df['metric'] == 'linf']

	return df

def format_float_value(value: float) -> str:
	"""
	Format a float value with at most 3 significant figures.
	For large values, use scientific notation.

	Args:
		value: The float value to format

	Returns:
		Formatted string representation of the value
	"""
	if abs(value) >= 1000:
		# Use simplified scientific notation for large values
		magnitude = int(math.floor(math.log10(abs(value))))
		return f"$\\sim 10^{{{magnitude}}}$"
	else:
		# Use regular notation with 3 sig figs for smaller values
		if abs(value) >= 100:
			return f"{value:.0f}"
		elif abs(value) >= 10:
			return f"{value:.1f}"
		else:
			return f"{value:.2f}"

def get_bound(algorithm: Algorithm, dims: int, metric: Metric, n: int) -> float | tuple[float, float] | None:
	"""
	Get the bound for a specific algorithm, dimension, and metric
	Returns either a single float, a tuple of two floats, or None
	"""
	log2_n = math.log2(n)

	match algorithm:
		case Algorithm.Domino:
			match metric:
				case Metric.Probes:
					if dims == 2:
						return (2 * log2_n + 1) / log2_n
					elif dims == 3:
						return (3 * log2_n + 4) / log2_n
					else:
						return None
				case Metric.Responses:
					if dims == 2:
						return (2 * log2_n + 1) / log2_n
					elif dims == 3:
						return (3 * log2_n + 4) / log2_n
					else:
						return None
				case Metric.Distance:
					return None
		case Algorithm.Orthant:
			match metric:
				case Metric.Probes:
					return 2 ** dims - 1
				case Metric.Responses:
					return 1
				case Metric.Distance:
					return 2 ** (dims + 1) * n
		case Algorithm.GeneralizedCBS:
			match metric:
				case Metric.Probes:
					a, b = (dims * (log2_n - 1) + dims * (dims + 1) - 1) / log2_n, (dims * (log2_n - 1) + 3 ** dims - 2) / log2_n
					return a if a == b else (a, b)
				case Metric.Responses:
					return (dims * log2_n + dims) / log2_n
				case Metric.Distance:
					return dims

def process_data_for_metric(df: pd.DataFrame, metric: Metric) -> tuple[dict[tuple[int, Algorithm], dict[str, float]], dict[int, dict[str, tuple[float, Algorithm]]]]:
	"""
	Process dataframe to extract data for a specific metric

	Args:
		df: The dataframe containing the data
		metric: The metric to process (Probes, Distance, or Responses)

	Returns:
		tuple containing:
			- table_data: dictionary mapping (dimension, algorithm) to metric data
			- best_values: dictionary mapping dimension to best values for each metric component and the algorithm that has it
	"""
	# Create a dictionary to store the formatted results
	table_data = {}

	# Process the dataframe to extract min, max, and avg for each algorithm and dimension
	for _, row in df.iterrows():
		algorithm_str = str(row['algorithm'])
		dims = int(row['dims'])
		n_value = int(row['n'])
		log2_n = math.log2(n_value) if metric != Metric.Distance else 1

		# Skip domino algorithms for dimensions other than 2D and 3D
		if algorithm_str.startswith('domino') and dims not in [2, 3]:
			continue

		# Map algorithm to its display name
		display_algorithm = Algorithm.from_string(algorithm_str)

		# Skip if algorithm mapping not found
		if display_algorithm is None:
			continue

		# Create a key for the dimension and algorithm
		key = (dims, display_algorithm)

		if key not in table_data:
			table_data[key] = {}

		# Store min, max, avg, and std values based on the metric
		metric_prefix = str(metric)
		if metric == Metric.Probes:
			table_data[key][f'{metric_prefix}_min'] = row['P_min'] / log2_n
			table_data[key][f'{metric_prefix}_max'] = row['P_max'] / log2_n
			table_data[key][f'{metric_prefix}_avg'] = row['P_mean'] / log2_n
			table_data[key][f'{metric_prefix}_std'] = row['P_std'] / log2_n
		elif metric == Metric.Distance:
			table_data[key][f'{metric_prefix}_min'] = row['D_hiker_ratio_min']
			table_data[key][f'{metric_prefix}_max'] = row['D_hiker_ratio_max']
			table_data[key][f'{metric_prefix}_avg'] = row['D_hiker_ratio_mean']
			table_data[key][f'{metric_prefix}_std'] = row['D_hiker_ratio_std']
		elif metric == Metric.Responses:
			table_data[key][f'{metric_prefix}_min'] = row['num_responses_min'] / log2_n
			table_data[key][f'{metric_prefix}_max'] = row['num_responses_max'] / log2_n
			table_data[key][f'{metric_prefix}_avg'] = row['num_responses_mean'] / log2_n
			table_data[key][f'{metric_prefix}_std'] = row['num_responses_std'] / log2_n

		# Get bound for this algorithm, dimension, and metric
		bound = get_bound(display_algorithm, dims, metric, n_value)
		if bound is None:
			table_data[key][f'{metric_prefix}_bound'] = float('nan')
		elif isinstance(bound, tuple):
			# Store both values from the tuple
			table_data[key][f'{metric_prefix}_bound'] = bound
		else:
			table_data[key][f'{metric_prefix}_bound'] = bound

	# Find the best values for each dimension and metric component
	metric_prefix = str(metric)
	best_values = {}

	# Initialize best_values for each dimension
	dimensions = sorted(set(key[0] for key in table_data.keys()))
	for dim in dimensions:
		best_values[dim] = {
			f'{metric_prefix}_min': (float('inf'), None),
			f'{metric_prefix}_max': (float('inf'), None),
			f'{metric_prefix}_avg': (float('inf'), None),
			f'{metric_prefix}_bound': (float('inf'), None),
			f'{metric_prefix}_std': (float('inf'), None)
		}

	# Find the best (lowest) values for each dimension and metric component
	# First pass: find the minimum values for each dimension and metric
	min_values_by_dim = {}
	for dim in dimensions:
		min_values_by_dim[dim] = {
			f'{metric_prefix}_min': float('inf'),
			f'{metric_prefix}_avg': float('inf'),
			f'{metric_prefix}_max': float('inf'),
			f'{metric_prefix}_bound': float('inf'),
			f'{metric_prefix}_std': float('inf')
		}

	# Find the minimum values
	for key, data in table_data.items():
		dim, algorithm = key

		for metric_key in [f'{metric_prefix}_min', f'{metric_prefix}_avg', f'{metric_prefix}_max']:
			if metric_key in data and data[metric_key] < min_values_by_dim[dim][metric_key]:
				min_values_by_dim[dim][metric_key] = data[metric_key]

		# Handle standard deviation
		std_key = f'{metric_prefix}_std'
		if std_key in data and data[std_key] < min_values_by_dim[dim][std_key]:
			min_values_by_dim[dim][std_key] = data[std_key]

		# Handle bound separately due to possible tuple values
		bound_key = f'{metric_prefix}_bound'
		if bound_key in data:
			bound_value = data[bound_key]

			# Skip NaN values
			if isinstance(bound_value, float) and math.isnan(bound_value):
				continue

			# For tuples, use the higher value for comparison
			if isinstance(bound_value, tuple):
				comparison_value = max(bound_value)
			else:
				comparison_value = bound_value

			if comparison_value < min_values_by_dim[dim][bound_key]:
				min_values_by_dim[dim][bound_key] = comparison_value

	# Second pass: store all algorithms that have the minimum value
	for dim in dimensions:
		best_values[dim] = {
			f'{metric_prefix}_min': (min_values_by_dim[dim][f'{metric_prefix}_min'], []),
			f'{metric_prefix}_avg': (min_values_by_dim[dim][f'{metric_prefix}_avg'], []),
			f'{metric_prefix}_max': (min_values_by_dim[dim][f'{metric_prefix}_max'], []),
			f'{metric_prefix}_bound': (min_values_by_dim[dim][f'{metric_prefix}_bound'], []),
			f'{metric_prefix}_std': (min_values_by_dim[dim][f'{metric_prefix}_std'], [])
		}

	# Find all algorithms that have the minimum value
	for key, data in table_data.items():
		dim, algorithm = key

		for metric_key in [f'{metric_prefix}_min', f'{metric_prefix}_avg', f'{metric_prefix}_max', f'{metric_prefix}_std']:
			if metric_key in data:
				# If this value equals the minimum, add this algorithm to the list
				if abs(data[metric_key] - best_values[dim][metric_key][0]) < 1e-6:  # Use small epsilon for float comparison
					best_values[dim][metric_key][1].append(algorithm)

		# Handle bound separately due to possible tuple values
		bound_key = f'{metric_prefix}_bound'
		if bound_key in data:
			bound_value = data[bound_key]

			# Skip NaN values
			if isinstance(bound_value, float) and math.isnan(bound_value):
				continue

			# For tuples, use the higher value for comparison
			if isinstance(bound_value, tuple):
				comparison_value = max(bound_value)
			else:
				comparison_value = bound_value

			# If this value equals the minimum, add this algorithm to the list
			if abs(comparison_value - best_values[dim][bound_key][0]) < 1e-6:  # Use small epsilon for float comparison
				best_values[dim][bound_key][1].append(algorithm)

	return table_data, best_values

def format_table_row(dim: int, table_data: dict[tuple[int, Algorithm], dict[str, float]], best_values: dict[int, dict[str, tuple[float, Algorithm]]], metric: Metric) -> str:
	"""
	Format a table row for a specific dimension

	Args:
		dim: The dimension
		table_data: dictionary mapping (dimension, algorithm) to metric data
		best_values: dictionary mapping dimension to best values for each metric component and the algorithm that has it
		metric: The metric being processed

	Returns:
		Formatted LaTeX table row
	"""
	row_parts = [f"{dim}D"]
	metric_prefix = str(metric)

	# Add data for each algorithm
	for algorithm in [Algorithm.Domino, Algorithm.Orthant, Algorithm.GeneralizedCBS]:
		key = (dim, algorithm)

		# If we have data for this algorithm and dimension
		if key in table_data:
			data = table_data[key]

			# Add standard deviation (σ) first
			std_key = f'{metric_prefix}_std'
			std_value = data[std_key]

			# Format standard deviation
			formatted_std = format_float_value(std_value)

			# Bold if this algorithm has the lowest standard deviation for this dimension
			if dim in best_values and algorithm in best_values[dim][std_key][1]:
				formatted_std = f"\\textbf{{{formatted_std}}}"

			row_parts.append(formatted_std)

			# Format metric values (avg, max) - skip min
			for metric_key in [f'{metric_prefix}_avg', f'{metric_prefix}_max']:
				value = data[metric_key]

				# Format the value
				formatted = format_float_value(value)

				# Bold if this algorithm is in the list of algorithms with the lowest value for this metric
				if dim in best_values and algorithm in best_values[dim][metric_key][1]:
					formatted = f"\\textbf{{{formatted}}}"

				row_parts.append(formatted)

			# Add bound
			bound_key = f'{metric_prefix}_bound'
			bound_value = data[bound_key]

			if math.isnan(bound_value) if isinstance(bound_value, float) else False:
				formatted_bound = "---"
			elif isinstance(bound_value, tuple):
				# Format both values from the tuple
				formatted_values = []
				for val in bound_value:
					formatted_values.append(format_float_value(val))

				# Format both values from the tuple separated by a slash
				formatted_bound = f"{formatted_values[0]}/{formatted_values[1]}"

				# Bold if this algorithm is in the list of algorithms with the lowest bound value
				if dim in best_values and algorithm in best_values[dim][bound_key][1]:
					formatted_bound = f"\\textbf{{{formatted_bound}}}"
			else:
				# Format the bound value
				formatted_bound = format_float_value(bound_value)

				# Bold if this algorithm is in the list of algorithms with the lowest bound value
				if dim in best_values and algorithm in best_values[dim][bound_key][1]:
					formatted_bound = f"\\textbf{{{formatted_bound}}}"

			row_parts.append(formatted_bound)
		else:
			# If no data for this algorithm and dimension, add empty cells
			if algorithm == Algorithm.Domino and dim not in [2, 3]:
				row_parts.extend(["---", "---", "---", "---"])
			else:
				row_parts.extend(["", "", "", ""])

	# Create the row
	return " & ".join(row_parts) + " \\\\"

def generate_latex_table(table_data: dict[tuple[int, Algorithm], dict[str, float]], best_values: dict[int, dict[str, tuple[float, Algorithm]]], metric: Metric) -> str:
	"""
	Generate a LaTeX table for a specific metric

	Args:
		table_data: dictionary mapping (dimension, algorithm) to metric data
		best_values: dictionary mapping dimension to best values for each metric component and the algorithm that has it
		metric: The metric being processed

	Returns:
		LaTeX table as a string
	"""
	# Generate LaTeX table header
	latex_table = [
		"\\begin{table*}[tb!]",
		"\\centering",
		"\\begin{tabular}{|c|rrrr|rrrr|rrrr|}",
		"\\hline",
		f"& \\multicolumn{{4}}{{c|}}{{{Algorithm.Domino}}} & \\multicolumn{{4}}{{c|}}{{{Algorithm.Orthant}}} & \\multicolumn{{4}}{{c|}}{{{Algorithm.GeneralizedCBS}}} \\\\",
		"$d$ & $\\sigma$ & Avg & Max & Bound & $\\sigma$ & Avg & Max & Bound & $\\sigma$ & Avg & Max & Bound \\\\",
		"\\hline"
	]

	# Add data rows for each dimension
	for dim in sorted(set(key[0] for key in table_data.keys())):
		row = format_table_row(dim, table_data, best_values, metric)
		latex_table.append(row)

	# Add table footer with appropriate caption and label
	caption = ""
	label = ""

	match metric:
		case Metric.Probes:
			caption = "Normalized number of probes ($P/\\log n$) for different search algorithms across dimensions."
			label = "tab:algorithm_metrics_P"
		case Metric.Distance:
			caption = "Normalized L-infinity distance ($D/\\delta_{min}$) for different search algorithms across dimensions."
			label = "tab:algorithm_metrics_D"
		case Metric.Responses:
			caption = "Normalized number of responses ($R/\\log n$) for different search algorithms across dimensions."
			label = "tab:algorithm_metrics_R"

	latex_table.extend([
		"\\hline",
		"\\end{tabular}",
		f"\\caption{{{caption}}}",
		f"\\label{{{label}}}",
		"\\end{table*}"
	])

	# Join all lines and return
	return "\n".join(latex_table)

def generate_metric_table(df: pd.DataFrame, metric: Metric) -> str:
	"""
	Generate a LaTeX table for a specific metric

	Args:
		df: The dataframe containing the data
		metric: The metric to process (Probes, Distance, or Responses)

	Returns:
		LaTeX table as a string
	"""
	# Process data for the metric
	table_data, best_values = process_data_for_metric(df, metric)

	# Generate and return the LaTeX table
	return generate_latex_table(table_data, best_values, metric)

def generate_probes_table(df: pd.DataFrame) -> str:
	"""
	Generate a LaTeX table for the Probes / log n metric
	"""
	return generate_metric_table(df, Metric.Probes)

def generate_distance_table(df: pd.DataFrame) -> str:
	"""
	Generate a LaTeX table for the Distance / delta_min metric (L_infinity)
	"""
	return generate_metric_table(df, Metric.Distance)

def generate_responses_table(df: pd.DataFrame) -> str:
	"""
	Generate a LaTeX table for the Responses / log n metric
	"""
	return generate_metric_table(df, Metric.Responses)

def generate_all_tables() -> tuple[str, str, str]:
	"""
	Generate all three LaTeX tables

	Returns:
		tuple containing probes table, distance table, and responses table as strings
	"""
	# Load the data
	df = load_aggregated_square_data()

	# Generate the tables
	probes_table = generate_probes_table(df)
	distance_table = generate_distance_table(df)
	responses_table = generate_responses_table(df)

	return probes_table, distance_table, responses_table

if __name__ == "__main__":
	probes_table, distance_table, responses_table = generate_all_tables()

	print("% === Probes Table ===")
	print(probes_table)
	print("\n\n% === Distance Table ===")
	print(distance_table)
	print("\n\n% === Responses Table ===")
	print(responses_table)
