import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# Set up plot style
OKABE_COLORS = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler('color', OKABE_COLORS)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 24  # Base font size
plt.rcParams['axes.labelsize'] = 24  # Axis label font size
plt.rcParams['xtick.labelsize'] = 24  # Tick label font size
plt.rcParams['ytick.labelsize'] = 24

def load_aggregated_square_data():
    """
    Load the aggregated square data from the CSV file
    """
    data_path = Path(__file__).parent / "data" / "aggregated_square_data.csv"
    if not data_path.exists():
        print(f"Error: Data file {data_path} not found.")
        print("Run aggregate_square_data.py first to generate the aggregated data.")
        sys.exit(1)
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def plot_square_metrics():
    """
    Create plots for:
    1. Normalized # probes (P / d log n) - lin-lin plot
    2. Normalized L_infty distance (D / hiker distance) - lin-x, log-y plot
    3. Normalized # responses (R_max / log n) - lin-x, log-y plot (note: not scaled by dimension)

    Each plot includes:
    - Points for simple_hypercube_search (labeled "Orthant")
    - Points for central_binary_search (labeled "CBS")
    - Points for domino_2d_search and domino_3d_search (both labeled "Domino")
    - Error bars for each point based on standard deviation
    - Two versions: one with all dimensions (1-8) and one with only dimensions 1-4
    """
    print("Starting plot generation...")
    start_time = time.time()

    # Load the data
    df = load_aggregated_square_data()
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")

    # Filter for L_infinity metric only
    df = df[df['metric'] == 'linf']
    print(f"Found {len(df)} rows with L_infinity metric")

    # Define the algorithms to plot
    algorithm_mapping = {
        'simple_hypercube_search': 'Orthant Algorithm',
        'central_binary_search': 'Generalized CBS Algorithm',
        'domino_2d_search': 'Domino Algorithms',
        'domino_3d_search': 'Domino Algorithms'
    }

    # Filter for the algorithms we want
    available_algorithms = df['algorithm'].unique()
    print(f"Available algorithms: {available_algorithms}")

    # Check which algorithms from our mapping are actually in the data
    valid_algorithms = [algo for algo in algorithm_mapping.keys() if algo in available_algorithms]
    print(f"Valid algorithms for plotting: {valid_algorithms}")

    if not valid_algorithms:
        print("Error: None of the required algorithms found in the data")
        sys.exit(1)

    # Filter the dataframe to only include the valid algorithms
    df = df[df['algorithm'].isin(valid_algorithms)]

    # Create a new column with the mapped algorithm names
    df['algorithm_label'] = df['algorithm'].map(algorithm_mapping)

    # Get unique dimensions
    all_dimensions = sorted(df['dims'].unique())
    print(f"Found dimensions: {all_dimensions}")

    # Create two sets of dimensions: all dimensions and dimensions 1-4
    all_dims = all_dimensions
    limited_dims = [d for d in all_dimensions if d <= 4]

    # Check if we have the required algorithms
    for algo in algorithm_mapping.keys():
        if algo not in df['algorithm'].unique():
            print(f"Warning: Algorithm '{algo}' not found in the data")

    # Print info about the data we're plotting
    print(f"All dimensions: {all_dims}")
    print(f"Limited dimensions (1-4): {limited_dims}")
    print(f"Found algorithms: {df['algorithm'].unique()}")

    # Create plots for all dimensions
    create_dimension_plots(df, algorithm_mapping, all_dims, "all")

    # Create plots for limited dimensions (1-4)
    create_dimension_plots(df, algorithm_mapping, limited_dims, "limited")

    # Don't show plots interactively to avoid hanging
    # plt.show()

def create_dimension_plots(df, algorithm_mapping, dimensions, suffix):
    """
    Create all three plots for a specific set of dimensions

    Args:
        df: The dataframe containing the data
        algorithm_mapping: Dictionary mapping algorithm names to display labels
        dimensions: List of dimensions to plot
        suffix: Suffix to add to the output filenames
    """
    print(f"\nCreating plots for {suffix} dimensions: {dimensions}")

    # Create three separate figures
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    fig3, ax3 = plt.subplots(figsize=(10, 8))

    # Plot 1: Normalized # probes (P / d log n) - lin-lin plot
    print(f"Creating normalized probes plot ({suffix})...")
    plot_normalized_metric(
        ax1,
        df,
        'P_mean',
        'P_std',
        lambda d, n: d * np.log2(n),
        'Normalized \\# of Probes $(P / k \\lceil \\log n \\rceil)$',
        # 'Y tmp',
        algorithm_mapping,
        dimensions,
        log_y=True
    )
    plt.figure(fig1.number)
    plt.tight_layout()
    plt.savefig(f'normalized_probes_{suffix}.png', dpi=300, bbox_inches='tight')
    print(f"Saved normalized_probes_{suffix}.png")

    # Plot 2: Normalized L_infty distance (D / hiker distance) - lin-x, log-y plot
    print(f"Creating normalized distance plot ({suffix})...")
    plot_normalized_metric(
        ax2,
        df,
        'D_hiker_ratio_mean',
        'D_hiker_ratio_std',
        lambda d, n: 1,  # Already normalized in the data
        'Normalized $L_{\\infty}$ Distance $(D/\\delta_{\\mathrm{min}})$',
        algorithm_mapping,
        dimensions,
        log_y=True
    )
    plt.figure(fig2.number)
    plt.tight_layout()
    plt.savefig(f'normalized_distance_{suffix}.png', dpi=300, bbox_inches='tight')
    print(f"Saved normalized_distance_{suffix}.png")

    # Plot 3: Normalized # responses (R_max / log n) - lin-x, log-y plot
    # Note: Now only normalized by log(n), not by dimension
    print(f"Creating normalized responses plot ({suffix})...")
    plot_normalized_metric(
        ax3,
        df,
        'num_responses_mean',
        'num_responses_std',
        lambda d, n: np.log2(n),  # Only normalize by log(n), not by dimension
        'Normalized \\# Responses $(R / \\lceil \\log n \\rceil)$',  # Updated label
        algorithm_mapping,
        dimensions,
        log_y=True
    )
    plt.figure(fig3.number)
    plt.tight_layout()
    plt.savefig(f'normalized_responses_{suffix}.png', dpi=300, bbox_inches='tight')
    print(f"Saved normalized_responses_{suffix}.png")

    # Close the figures to free up memory
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

def plot_normalized_metric(ax, df, mean_col, std_col, normalizer, ylabel, algorithm_mapping, dimensions, log_y=False):
    """
    Plot a normalized metric with error bars

    Args:
        ax: The matplotlib axis to plot on
        df: The dataframe containing the data
        mean_col: The column name for the mean values
        std_col: The column name for the standard deviation values
        normalizer: A function that takes dimension and n and returns the normalization factor
        ylabel: The y-axis label
        algorithm_mapping: Dictionary mapping algorithm names to display labels
        dimensions: List of dimensions to plot
        log_y: Whether to use a logarithmic y-axis
    """
    # Get unique algorithm labels from the filtered dataframe
    available_algorithms = df['algorithm'].unique()
    valid_algorithms = [algo for algo in algorithm_mapping.keys() if algo in available_algorithms]

    # Get unique labels that are actually present in the data
    unique_labels_unsorted = set(algorithm_mapping[algo] for algo in valid_algorithms)

    # Define a specific order for the labels to ensure Orthant is plotted first
    # This prevents Orthant from covering other algorithms in the plot
    preferred_order = ['Orthant Algorithm', 'Generalized CBS Algorithm', 'Domino Algorithms']
    unique_labels = [label for label in preferred_order if label in unique_labels_unsorted]

    print(f"Plotting for labels in order: {unique_labels}")

    # Set up markers for each algorithm
    markers = {'Orthant Algorithm': 'o', 'Generalized CBS Algorithm': 's', 'Domino Algorithms': '^'}
    colors = {'Orthant Algorithm': OKABE_COLORS[0], 'Generalized CBS Algorithm': OKABE_COLORS[1], 'Domino Algorithms': OKABE_COLORS[2]}

    # Plot each algorithm
    for label in unique_labels:
        # For each algorithm label, get all algorithms that map to this label
        algos = [k for k, v in algorithm_mapping.items() if v == label and k in valid_algorithms]
        print(f"  For label '{label}', using algorithms: {algos}")

        # For each dimension, collect data points
        x_values = []
        y_values = []
        y_errors = []

        for dim in dimensions:
            # Filter data for this dimension and these algorithms
            dim_data = df[(df['dims'] == dim) & (df['algorithm'].isin(algos))]

            if not dim_data.empty:
                # If we have multiple algorithms with the same label (e.g., Domino), average them
                x_values.append(dim)

                # Calculate normalized values
                normalized_values = []
                normalized_stds = []

                for _, row in dim_data.iterrows():
                    try:
                        norm_factor = normalizer(float(row['dims']), float(row['n']))
                        normalized_values.append(float(row[mean_col]) / norm_factor)
                        normalized_stds.append(float(row[std_col]) / norm_factor)
                    except Exception as e:
                        print(f"Error normalizing {row['algorithm']} for dim={row['dims']}: {e}")
                        continue

                if normalized_values:
                    y_values.append(np.mean(normalized_values))
                    # Combine standard deviations (simple average for now)
                    y_errors.append(np.mean(normalized_stds))
                else:
                    # Remove the x value if we couldn't calculate any y values
                    x_values.pop()

        if x_values:  # Only plot if we have data points
            print(f"  Plotting {len(x_values)} points for {label}")
            # Plot the data points with error bars
            ax.errorbar(
                x_values,
                y_values,
                yerr=y_errors,
                fmt=markers[label],
                color=colors[label],
                label=label,
                markersize=10,
                capsize=5,
                linestyle='None'  # Ensure no lines between points
            )
        else:
            print(f"  No valid data points for {label}")

    # Set up the plot
    # Remove x-axis label and set custom tick labels (1D, 2D, etc.)
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel(ylabel)
    ax.set_xticks(dimensions)

    # Create custom tick labels (1D, 2D, etc.)
    dimension_labels = [f"${d}\\mathrm D$" for d in dimensions]
    ax.set_xticklabels(dimension_labels)

    ax.grid(True, linestyle='--', alpha=0.7)

    if log_y:
        # Use log base 2 instead of default base 10
        ax.set_yscale('log', base=2)

    # Cap the maximum y-limit to be at most twice the maximum y-value (without error bars)
    # This helps with cases where the standard deviation is very large by cutting them off
    all_y_values = []

    # Collect all y values (without errors) across all algorithms
    for label in unique_labels:
        algos = [k for k, v in algorithm_mapping.items() if v == label and k in valid_algorithms]
        for algo in algos:
            algo_data = df[df['algorithm'] == algo]
            for dim in dimensions:
                dim_data = algo_data[algo_data['dims'] == dim]
                if not dim_data.empty:
                    for _, row in dim_data.iterrows():
                        try:
                            norm_factor = normalizer(float(row['dims']), float(row['n']))
                            y_val = float(row[mean_col]) / norm_factor
                            all_y_values.append(y_val)
                        except Exception:
                            continue

    if all_y_values:
        # Calculate the maximum y value (without error bars)
        max_y = max([y for y in all_y_values if y is not None] or [0])

        # Set the y-limit to at most twice the maximum value
        # This will cut off large error bars that extend beyond this limit
        if not log_y or max_y > 1e-10:
            current_ylim = ax.get_ylim()
            # If the current limit is larger than 2x max_y, reduce it
            if current_ylim[1] > max_y * 2:
                ax.set_ylim(current_ylim[0], max_y * 2)
                print(f"  Capping y-limit to {max_y * 2} (2x max value without error bars)")

    # Add legend
    ax.legend()

if __name__ == "__main__":
    try:
        print("Starting plot_squares.py...")
        plot_square_metrics()
        print("All plots created successfully!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
