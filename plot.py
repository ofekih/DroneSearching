from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import math
from typing import Callable

from pandas import DataFrame

OKABE_COLORS = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=OKABE_COLORS) # type: ignore

def load_aggregated_results():
    """
    Load the aggregated results from the CSV file
    """
    data_path = Path(__file__).parent / "data" / "aggregated_results.csv"
    return pd.read_csv(data_path) # type: ignore

def plot_normalized_metrics():
    """
    Create whisker plots for P(n)/log n and D(n)/n for each algorithm
    """
    # Load the data
    df = load_aggregated_results()
    
    # Create figure with two subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7)) # type: ignore
    
    # Process data for each algorithm
    algorithms: list[int] = sorted(df['algorithm'].unique()) # type: ignore
    
    # Plot P(n)/log n
    plot_metric_normalized(ax1, df, 'P', lambda n: math.log(n), 'P(n)/log n', algorithms)
    
    # Plot D(n)/n
    plot_metric_normalized(ax2, df, 'D', lambda n: n, 'D(n)/n', algorithms)
    
    # Set legends and save
    plt.tight_layout()
    plt.savefig('algorithm_performance.png', dpi=300) # type: ignore
    plt.show() # type: ignore

def plot_metric_normalized(ax: Axes, df: DataFrame, metric: str, normalizer_func: Callable[[float], float], title: str, algorithms: list[int]):
    """
    Plot a specific metric normalized by the given function
    
    Args:
        ax: Matplotlib axis to plot on
        df: DataFrame with the data
        metric: The metric to plot (e.g., 'P' or 'D')
        normalizer_func: Function to normalize the metric (e.g., log(n) or n)
        title: Title for the plot
        algorithms: List of algorithm numbers to plot
    """
    # For each algorithm
    x_positions: list[int] = []
    x_labels: list[str] = []
    
    for alg in algorithms:
        # Filter data for this algorithm and metric
        alg_data = df[(df['algorithm'] == alg) & (df['metric'] == metric)]
        
        # Group by n value
        for n_val in sorted(alg_data['n'].unique()): # type: ignore
            n_data = alg_data[alg_data['n'] == n_val] # type: ignore
            if not n_data.empty: # type: ignore
                row: dict[str,float] = n_data.iloc[0] # type: ignore
                
                # Calculate normalized values
                normalized_avg: float = row['avg'] / normalizer_func(n_val)
                normalized_std: float = row['std_dev'] / normalizer_func(n_val)
                
                # Plot error bar at position
                x_pos = len(x_positions)
                x_positions.append(x_pos)
                x_labels.append(f"A{alg}\nn={int(n_val)}")
                
                # Plot whisker-style error bar
                ax.errorbar(
                    x_pos, normalized_avg, 
                    yerr=normalized_std,
                    fmt='o', capsize=5, 
                    color=OKABE_COLORS[(alg - 1) % len(OKABE_COLORS)]
                )
    
    # Set labels and title
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel(title)
    ax.set_title(f"{title} by Algorithm and n")
    ax.grid(True, linestyle='--', alpha=0.7)

def plot_raw_metrics():
    """
    Create whisker plots for raw P(n) and D(n) values for each algorithm
    """
    # Load the data
    df = load_aggregated_results()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Process data for each algorithm
    algorithms = sorted(df['algorithm'].unique())
    
    # Plot P(n)
    plot_raw_metric(ax1, df, 'P', 'P(n)', algorithms)
    
    # Plot D(n)
    plot_raw_metric(ax2, df, 'D', 'D(n)', algorithms)
    
    # Set legends and save
    plt.tight_layout()
    plt.savefig('algorithm_performance_raw.png', dpi=300)
    plt.show()

def plot_raw_metric(ax, df, metric, title, algorithms):
    """
    Plot a specific raw metric
    
    Args:
        ax: Matplotlib axis to plot on
        df: DataFrame with the data
        metric: The metric to plot (e.g., 'P' or 'D')
        title: Title for the plot
        algorithms: List of algorithm numbers to plot
    """
    # For each algorithm
    x_positions = []
    x_labels = []
    
    for alg in algorithms:
        # Filter data for this algorithm and metric
        alg_data = df[(df['algorithm'] == alg) & (df['metric'] == metric)]
        
        # Group by n value
        for n_val in sorted(alg_data['n'].unique()):
            n_data = alg_data[alg_data['n'] == n_val]
            if not n_data.empty:
                row = n_data.iloc[0]
                
                # Plot error bar at position
                x_pos = len(x_positions)
                x_positions.append(x_pos)
                x_labels.append(f"A{alg}\nn={int(n_val)}")
                
                # Plot whisker-style error bar
                ax.errorbar(
                    x_pos, row['avg'], 
                    yerr=row['std_dev'],
                    fmt='o', capsize=5, 
                    color=OKABE_COLORS[(alg - 1) % len(OKABE_COLORS)]
                )
    
    # Set labels and title
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel(title)
    ax.set_title(f"{title} by Algorithm and n")
    ax.grid(True, linestyle='--', alpha=0.7)

if __name__ == "__main__":
    # Plot normalized metrics
    plot_normalized_metrics()
    
    # Also plot raw metrics for comparison
    # plot_raw_metrics()

