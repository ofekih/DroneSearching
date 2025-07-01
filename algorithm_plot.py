"""
Plotting and visualization functions for the delta searching algorithm.
"""
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import patches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle as PltCircle
from shapely import Polygon

from geometry_types import Circle, Square


OKABE_COLORS = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=OKABE_COLORS) # type: ignore


def get_circles_plot(circles: list[Circle], *,
                    title: str | None = None,
                    p: float | None = None,
                    c: float | None = None,
                    ct: float | None = None,
                    cpu_time: float | None = None,
                    ax: Axes | None = None,
                    squares: list[Square] = [],
                    polygons: list[Polygon] = []):
    """Plot circles on either a new figure or an existing axes."""
    if ax is None:
        _, ax = plt.subplots(1, 1) # type: ignore
    else:
        _ = ax.figure
    
    # Draw unit circle with dashed lines in black
    ax.add_patch(PltCircle((0, 0), 1, fill=False, linestyle='--', color='black'))

    # Draw the circles
    for i, circle in enumerate(circles):
        color = OKABE_COLORS[(i + 1) % len(OKABE_COLORS)]
        ax.add_patch(PltCircle((circle.x, circle.y), circle.r, fill=False, color=color))
        ax.text(circle.x, circle.y, str(i + 1), # type: ignore
                horizontalalignment='center', verticalalignment='center',
                color=color, fontsize=18)
        
    for square in squares:
        ax.add_patch(patches.Rectangle((square.x, square.y), square.side_length, square.side_length, fill=False, color='black'))

    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.plot(x, y, color='black', linewidth=0.5) # type: ignore

    # Set plot limits and aspect ratio
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    
    # Add title and information
    if title:
        ax.set_title(title) # type: ignore
    
    stat_text: list[str] = []
    if p is not None:
        stat_text.append(f"p = {float(p):.3f}")
    if c is not None:
        stat_text.append(f"T(n) = {float(c):.3f} log n")
    if ct is not None:
        # D(n) = ct * n
        stat_text.append(f"D(n) = {float(ct):.3f} n")
    if cpu_time is not None:
        stat_text.append(f"done in {cpu_time:.2f}s")

    if stat_text:
        ax.set_xlabel(", ".join(stat_text), fontsize=10) # type: ignore
    
    return ax


def draw_circles(circles: list[Circle], *,
                title: str | None = None,
                p: float | None = None,
                c: float | None = None,
                ct: float | None = None,
                cpu_time: float | None = None,
                squares: list[Square] = [],
                polygons: list[Polygon] = []) -> None:
    """Draw a single set of circles."""
    get_circles_plot(circles, title=title, p=p, c=c, ct=ct, cpu_time=cpu_time, squares=squares, polygons=polygons)
    plt.show() # type: ignore


def print_latex_circles(circles: list[Circle]):
    """Print circles in a LaTeX-friendly format that can be copy-pasted."""
    for i, circle in enumerate(sorted(circles, key=lambda c: c.r, reverse=True)):
        print(f"    {{{circle.x}/{circle.y}/{circle.r}}}", end="")
        if i == len(circles) - 1:
            print("%")
        else:
            print(",")
