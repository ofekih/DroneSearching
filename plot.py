from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import math
from typing import Callable, Literal
from pandas import DataFrame

OKABE_COLORS = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', OKABE_COLORS)  # type: ignore
plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['font.size'] = 21  # Increase base font size
plt.rcParams['axes.labelsize'] = 24  # Increase axis label font size
plt.rcParams['xtick.labelsize'] = 21  # Increase tick label font size
plt.rcParams['ytick.labelsize'] = 21

def load_aggregated_results():
    """
    Load the aggregated results from the CSV file
    """
    data_path = Path(__file__).parent / "data" / "aggregated_results.csv"
    return pd.read_csv(data_path) # type: ignore

def get_figures_dir():
    """
    Get the figures directory path, creating it if it doesn't exist
    """
    figures_dir = Path(__file__).parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    return figures_dir

def plot_normalized_metrics(plot_type: Literal['error_bars', 'boxplot'] = 'boxplot', show_minmax: bool = True):
    """
    Create plots for P/log2 n, D/n, and num_responses/log2 n for each algorithm
    
    Args:
        plot_type: Type of plot to create - 'error_bars' or 'boxplot' (default: 'boxplot')
        show_minmax: Whether to show the min/max values as triangular markers (default: True)
    """
    # Load the data
    df = load_aggregated_results()
    
    # Get figures directory
    figures_dir = get_figures_dir()
    
    # Create three separate figures
    fig1, ax1 = plt.subplots(figsize=(8, 7))  # type: ignore
    fig2, ax2 = plt.subplots(figsize=(8, 7))  # type: ignore
    fig3, ax3 = plt.subplots(figsize=(8, 7))  # type: ignore
    
    # Process data for each algorithm
    algorithms: list[int] = sorted(df['algorithm'].unique())  # type: ignore
    
    # Plot P/log2 n
    if plot_type == 'error_bars':
        plot_metric_normalized(ax1, df, 'P', lambda n: math.log2(n), 'Normalized \\# of Probes $P/\\lceil \\log n \\rceil$', algorithms, show_minmax)
        plt.figure(fig1.number)  # type: ignore
        plt.tight_layout()  # type: ignore
        plt.savefig(figures_dir / 'P.png', dpi=300, bbox_inches='tight')  # type: ignore
    else:
        plot_box_whisker_normalized(ax1, df, 'P', lambda n: math.log2(n), 'Normalized \\# of Probes $P/\\lceil \\log n \\rceil$', algorithms)
        plt.figure(fig1.number)  # type: ignore
        plt.tight_layout()  # type: ignore
        plt.savefig(figures_dir / 'P_box.png', dpi=300, bbox_inches='tight')  # type: ignore
    
    # Plot D/n
    if plot_type == 'error_bars':
        plot_metric_normalized(ax2, df, 'D', lambda n: n, 'Normalized Distance Traveled $D/n$', algorithms, show_minmax)
        plt.figure(fig2.number)  # type: ignore
        plt.tight_layout()  # type: ignore
        plt.savefig(figures_dir / 'D.png', dpi=300, bbox_inches='tight')  # type: ignore
    else:
        plot_box_whisker_normalized(ax2, df, 'D', lambda n: n, 'Normalized Distance Traveled $D/n$', algorithms)
        plt.figure(fig2.number)  # type: ignore
        plt.tight_layout()  # type: ignore
        plt.savefig(figures_dir / 'D_box.png', dpi=300, bbox_inches='tight')  # type: ignore
    
    # Plot num_responses/log2 n
    if plot_type == 'error_bars':
        plot_metric_normalized(ax3, df, 'num_responses', lambda n: math.log2(n), 'Normalized \\# POI Responses $R/\\lceil \\log n \\rceil$', algorithms, show_minmax)
        plt.figure(fig3.number)  # type: ignore
        plt.tight_layout()  # type: ignore
        plt.savefig(figures_dir / 'R.png', dpi=300, bbox_inches='tight')  # type: ignore
    else:
        plot_box_whisker_normalized(ax3, df, 'num_responses', lambda n: math.log2(n), 'Normalized \\# POI Responses $R/\\lceil \\log n \\rceil$', algorithms)
        plt.figure(fig3.number)  # type: ignore
        plt.tight_layout()  # type: ignore
        plt.savefig(figures_dir / 'R_box.png', dpi=300, bbox_inches='tight')  # type: ignore

    plt.show()  # type: ignore

def plot_box_whisker_normalized(ax: Axes, df: DataFrame, metric: str, normalizer_func: Callable[[float], float], 
                               title: str, algorithms: list[int]):
    """
    Plot a box and whisker plot for a specific metric normalized by the given function
    
    Args:
        ax: Matplotlib axis to plot on
        df: DataFrame with the data
        metric: The metric to plot (e.g., 'P' or 'D')
        normalizer_func: Function to normalize the metric (e.g., log2(n) or n)
        title: Title for the plot
        algorithms: List of algorithm numbers to plot
    """
    # For each algorithm
    x_positions: list[int] = []
    x_labels: list[str] = []
    boxplot_data: list[list[float]] = []
    colors: list[str] = []
    
    for alg in algorithms:  # type: ignore
        # Filter data for this algorithm and metric
        alg_data = df[(df['algorithm'] == alg) & (df['metric'] == metric)]
        
        # Group by n value
        for n_val in sorted(alg_data['n'].unique()):  # type: ignore
            n_data = alg_data[alg_data['n'] == n_val]  # type: ignore
            if not n_data.empty:  # type: ignore
                row = n_data.iloc[0]  # type: ignore
                
                # Get the normalized quartile values for boxplot
                norm_factor = normalizer_func(n_val)
                whisker_min = row['min'] / norm_factor  # type: ignore
                q25 = row['q25'] / norm_factor  # type: ignore
                median = row['median'] / norm_factor  # type: ignore
                q75 = row['q75'] / norm_factor  # type: ignore
                whisker_max = row['max'] / norm_factor  # type: ignore

                print(whisker_max)  # type: ignore
                
                # Store the box plot statistics in the required format
                # [whisker_min, q25, median, q75, whisker_max]
                boxplot_data.append([whisker_min, q25, median, q75, whisker_max])
                
                # Keep track of positions and labels
                x_pos = len(x_positions)
                x_positions.append(x_pos)
                x_labels.append(f"A{alg}")
                
                # Store color for this algorithm
                colors.append(OKABE_COLORS[(alg - 1) % len(OKABE_COLORS)])
    
    # Create the box and whisker plot
    if boxplot_data:
        # Create custom box plots
        bplot = ax.boxplot(  # type: ignore
            boxplot_data,
            positions=x_positions,
            widths=0.6, 
            patch_artist=True,  # Fill boxes with color
            showmeans=False,
            showfliers=False,   # Don't show outliers
            medianprops={'color': 'black', 'linewidth': 1.5},
            boxprops={'linewidth': 1.5},
            whiskerprops={'linewidth': 1.5},
            capprops={'linewidth': 1.5}
        )
        
        # Apply Okabe-Ito colors to boxes
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)  # type: ignore
            patch.set_alpha(0.7)
    
    # Set labels
    ax.set_xticks(x_positions)  # type: ignore
    ax.set_xticklabels(x_labels, rotation=45, ha='right')  # type: ignore
    ax.set_ylabel(title)  # type: ignore
    ax.grid(True, linestyle='--', alpha=0.7)  # type: ignore

def plot_metric_normalized(ax: Axes, df: DataFrame, metric: str, normalizer_func: Callable[[float], float], 
                          title: str, algorithms: list[int], show_minmax: bool = True):
    """
    Plot a specific metric normalized by the given function using error bars
    
    Args:
        ax: Matplotlib axis to plot on
        df: DataFrame with the data
        metric: The metric to plot (e.g., 'P' or 'D')
        normalizer_func: Function to normalize the metric (e.g., log2(n) or n)
        title: Title for the plot
        algorithms: List of algorithm numbers to plot
        show_minmax: Whether to show the min/max values as triangular markers (default: True)
    """
    # For each algorithm
    x_positions: list[int] = []
    x_labels: list[str] = []
    
    for alg in algorithms:
        # Filter data for this algorithm and metric
        alg_data = df[(df['algorithm'] == alg) & (df['metric'] == metric)]
        
        # Group by n value
        for n_val in sorted(alg_data['n'].unique()):  # type: ignore
            n_data = alg_data[alg_data['n'] == n_val]  # type: ignore
            if not n_data.empty:  # type: ignore
                row = n_data.iloc[0]  # type: ignore
                normalized_avg: float = row['avg'] / normalizer_func(n_val)  # type: ignore
                normalized_std: float = row['std_dev'] / normalizer_func(n_val)  # type: ignore
                normalized_min: float = row['min'] / normalizer_func(n_val)  # type: ignore
                normalized_max: float = row['max'] / normalizer_func(n_val)  # type: ignore
                
                # Plot error bar at position
                x_pos = len(x_positions)
                x_positions.append(x_pos)
                x_labels.append(f"A{alg}")
                
                color = OKABE_COLORS[(alg - 1) % len(OKABE_COLORS)]
                
                # Plot whisker-style error bar with larger marker
                ax.errorbar(  # type: ignore
                    x_pos, normalized_avg,  # type: ignore
                    yerr=normalized_std,  # type: ignore
                    fmt='o', capsize=5,  # type: ignore
                    color=color,
                    markersize=8
                )
                
                # Plot min and max points with larger markers if enabled
                if show_minmax:
                    ax.plot(x_pos, normalized_min, 'v', color=color, alpha=0.7, markersize=8)  # type: ignore
                    ax.plot(x_pos, normalized_max, '^', color=color, alpha=0.7, markersize=8)  # type: ignore
    
    # Set labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel(title) # type: ignore
    ax.grid(True, linestyle='--', alpha=0.7) # type: ignore

if __name__ == "__main__":
    plot_normalized_metrics(plot_type='error_bars', show_minmax=False)
    plot_normalized_metrics(plot_type='boxplot', show_minmax=False)
