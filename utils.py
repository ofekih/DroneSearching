from collections import deque
from dataclasses import dataclass
from typing import Generator, NamedTuple, TypedDict, Callable
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle as PltCircle
from shapely import Polygon
import shapely
import math

OKABE_COLORS = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=OKABE_COLORS) # type: ignore
plt.rcParams['text.usetex'] = True

class Circle(NamedTuple):
    x: float
    y: float
    r: float

class Square(NamedTuple):
    x: float
    y: float
    side_length: float

class HorizontalLine(NamedTuple):
    start: float
    end: float

class CirclesPlotKwargs(TypedDict, total=False):
    title: str | None
    p: float
    c: float
    cpu_time: float

UNIT_CIRCLE = Circle(0.0, 0.0, 1.0)

@dataclass
class Precision:
    precision: int = 7
    epsilon: float = 1e-3
    unit_circle_polygon: Polygon = shapely.Point(0.0, 0.0).buffer(1.0)

    def __post_init__(self):
        self.unit_circle_polygon = self.get_circle_polygon(UNIT_CIRCLE)

    def set_precision(self, precision: int) -> None:
        if precision == self.precision:
            return

        self.precision = precision
        self.epsilon = 1 / 10 ** (precision // 2)
        self.unit_circle_polygon = self.get_circle_polygon(UNIT_CIRCLE)

    def get_circle_polygon(self, circle: Circle) -> Polygon:
        quad_segs = min(math.ceil(circle.r * math.pi / 2 / self.epsilon), 2 ** 20)

        return shapely.Point(circle.x, circle.y).buffer(circle.r, quad_segs=quad_segs)

PRECISION = Precision()

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
    for i, circle in enumerate(sorted(circles, key=lambda circle: circle.r, reverse=True)):
        color = OKABE_COLORS[(i + 1) % len(OKABE_COLORS)]
        ax.add_patch(PltCircle((circle.x, circle.y), circle.r, fill=False, color=color))
        ax.text(circle.x, circle.y, str(i + 1), # type: ignore
                horizontalalignment='center', verticalalignment='center',
                color=color, fontsize=18)
        
    for square in squares:
        ax.add_patch(plt.Rectangle((square.x, square.y), square.side_length, square.side_length, fill=False, color='black')) # type: ignore

    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.plot(x, y, color='black', linewidth=0.5) # type: ignore

    # Set plot limits and aspect ratio
    ax.set_xlim(-1.1, 1.1) # type: ignore
    ax.set_ylim(-1.1, 1.1) # type: ignore
    ax.set_aspect('equal') # type: ignore
    
    # Add title and information
    if title:
        ax.set_title(title) # type: ignore
    
    stat_text: list[str] = []
    if p is not None:
        stat_text.append(f"$p = {float(p):.3f}$")
    if c is not None:
        stat_text.append(f"$T(n) = {float(c):.3f} \\log n$")
    if ct is not None:
        # D(n) = ct * n
        stat_text.append(f"$D(n) = {float(ct):.3f} n$")
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

def get_horizontal_line(circle: Circle, y: float) -> HorizontalLine | None:
    """Get the horizontal line that intersects the circle at y."""
    if abs(y - circle.y) > circle.r:
        return None

    delta = (circle.r ** 2 - (y - circle.y) ** 2) ** 0.5
    return HorizontalLine(circle.x - delta, circle.x + delta)

def get_line_union(lines: list[HorizontalLine]) -> list[HorizontalLine]:
    """Merge horizontal lines into a minimal set of non-overlapping lines."""
    if not lines:
        return []

    # Sort by start coordinate
    sorted_lines = sorted(lines, key=lambda line: line.start)
    merged_lines: list[HorizontalLine] = []
    current = sorted_lines[0]

    for line in sorted_lines[1:]:
        if line.start <= current.end:
            # Lines overlap, update end point
            current = HorizontalLine(current.start, max(current.end, line.end))
        else:
            # No overlap, add current line and start new one
            merged_lines.append(current)
            current = line

    merged_lines.append(current)
    return merged_lines

def do_circles_cover_unit_circle(circles: list[Circle], y: float) -> bool:
    lines = [line for circle in circles for line in [get_horizontal_line(circle, y)] if line is not None]
    union = get_line_union(lines)

    unit_circle_line = get_horizontal_line(UNIT_CIRCLE, y)

    if unit_circle_line is None:
        return True
    
    return any(union_line.start <= unit_circle_line.start and union_line.end >= unit_circle_line.end for union_line in union)

def covers_unit_circle_2(circles: list[Circle]) -> bool:
    y = -1.0
    while y < 1:
        if not do_circles_cover_unit_circle(circles, y):
            return False
        y += PRECISION.epsilon

    return True

def is_point_covered(circle: Circle, x: float, y: float) -> bool:
    return (x - circle.x) ** 2 + (y - circle.y) ** 2 <= circle.r ** 2

def is_point_covered_by_any(circles: list[Circle], x: float, y: float) -> bool:
    return any(is_point_covered(circle, x, y) for circle in circles)

def is_fully_covered(square: Square, circle: Circle) -> bool:
    return is_point_covered(circle, square.x, square.y) and \
            is_point_covered(circle, square.x + square.side_length, square.y) and \
            is_point_covered(circle, square.x, square.y + square.side_length) and \
            is_point_covered(circle, square.x + square.side_length, square.y + square.side_length)

def is_square_covered(circles: list[Circle], square: Square) -> bool:
    x, y = square.x, square.y
    side_length = square.side_length
    corners = [(x, y), (x + side_length, y), (x, y + side_length), (x + side_length, y + side_length)]
    
    num_outside_unit_circle = sum(1 for corner in corners if not is_point_covered(UNIT_CIRCLE, *corner))
    
    if num_outside_unit_circle == 4:
        return True
        
    # Check if any point inside unit circle is not covered
    for corner in corners:
        if is_point_covered(UNIT_CIRCLE, *corner) and not is_point_covered_by_any(circles, *corner):
            return False
            
    # Check if square is entirely covered by any circle
    if any(is_fully_covered(square, circle) for circle in circles):
        return True

    # Recurse into four sub-quadrants
    new_side_length = square.side_length / 2

    if new_side_length < PRECISION.epsilon:
        return True

    subsquares = [
        Square(x, y, new_side_length),
        Square(x + new_side_length, y, new_side_length),
        Square(x, y + new_side_length, new_side_length),
        Square(x + new_side_length, y + new_side_length, new_side_length)
    ]
    return all(is_square_covered(circles, subsquare) for subsquare in subsquares)

def get_all_uncovered_squares(circles: list[Circle]) -> Generator[Square, None, None]:
    def get_uncovered_squares(square: Square) -> Generator[Square, None, None]:
        x, y = square.x, square.y
        side_length = square.side_length
        corners = [(x, y), (x + side_length, y), (x, y + side_length), (x + side_length, y + side_length)]

        num_outside_unit_circle = sum(1 for corner in corners if not is_point_covered(UNIT_CIRCLE, *corner))

        if num_outside_unit_circle == 4:
            return
        
        num_uncovered_corners = sum(1 for corner in corners if is_point_covered(UNIT_CIRCLE, *corner) and not is_point_covered_by_any(circles, *corner))

        if num_uncovered_corners > 3:
            yield square
            return

        if num_uncovered_corners == 0 and any(is_fully_covered(square, circle) for circle in circles):
            return

        new_side_length = square.side_length / 2

        if new_side_length < PRECISION.epsilon:
            return

        subsquares = [
            Square(x, y, new_side_length),
            Square(x + new_side_length, y, new_side_length),
            Square(x, y + new_side_length, new_side_length),
            Square(x + new_side_length, y + new_side_length, new_side_length)
        ]
        for subsquare in subsquares:
            yield from get_uncovered_squares(subsquare)

    yield from get_uncovered_squares(Square(-1.0, -1.0, 1.0))
    yield from get_uncovered_squares(Square(-1.0, 0.0, 1.0))
    yield from get_uncovered_squares(Square(0.0, -1.0, 1.0))
    yield from get_uncovered_squares(Square(0.0, 0.0, 1.0))

def get_biggest_uncovered_square(circles: list[Circle]):
    # use BFS instead of DFS, first square found is guaranteed to be the biggest

    q = deque([Square(-1.0, -1.0, 1.0), Square(-1.0, 0.0, 1.0),
               Square(0.0, -1.0, 1.0), Square(0.0, 0.0, 1.0)])
    
    while q:
        square = q.popleft()

        x, y = square.x, square.y
        side_length = square.side_length
        corners = [(x, y), (x + side_length, y), (x, y + side_length), (x + side_length, y + side_length)]

        num_outside_unit_circle = sum(1 for corner in corners if not is_point_covered(UNIT_CIRCLE, *corner))

        if num_outside_unit_circle == 4:
            continue

        num_uncovered_corners = sum(1 for corner in corners if is_point_covered(UNIT_CIRCLE, *corner) and not is_point_covered_by_any(circles, *corner))

        if num_uncovered_corners > 3:
            return square
        
        if num_uncovered_corners == 0 and any(is_fully_covered(square, circle) for circle in circles):
            continue

        new_side_length = square.side_length / 2

        if new_side_length < PRECISION.epsilon:
            continue

        subsquares = [
            Square(x, y, new_side_length),
            Square(x + new_side_length, y, new_side_length),
            Square(x, y + new_side_length, new_side_length),
            Square(x + new_side_length, y + new_side_length, new_side_length)
        ]

        q.extend(subsquares)

    return None

def covers_unit_circle(circles: list[Circle]) -> bool:
    # (x, y) are the bottom left coordinates of the square
    return all(is_square_covered(circles, Square(x, y, 1.0)) 
              for x, y in [(-1.0, -1.0), (-1.0, 0.0), (0.0, -1.0), (0.0, 0.0)])

def binary_search(
    start: float,
    end: float,
    evaluator: Callable[[float], tuple[bool, list[Circle]]],
    debug: bool = False,
) -> tuple[float, list[Circle]]:
    """Binary search for the smallest value that works."""

    # Keep track of the largest failure and smallest success
    largest_failure: float | None = None
    smallest_success: float | None = None
    smallest_success_circles: list[Circle] | None = None

    # Run binary search
    while end - start > PRECISION.epsilon:
        p = (start + end) / 2
        success, circles = evaluator(p)

        if success:
            end = p
            if smallest_success is None or p < smallest_success:
                smallest_success = p
                smallest_success_circles = circles
                if debug:
                    print(f'p = {p} succeeded')
        else:
            start = p
            if largest_failure is None or p > largest_failure:
                largest_failure = p
                if debug:
                    print(f'p = {p} failed')

    if smallest_success_circles is None or smallest_success is None:
        raise ValueError('No solution found')

    return smallest_success, smallest_success_circles

def covers_unit_circle_3(circles: list[Circle]) -> bool:
    circle_polygons = [PRECISION.get_circle_polygon(circle) for circle in circles]
    diff = PRECISION.unit_circle_polygon.difference(shapely.union_all(circle_polygons)) # type: ignore

    return diff.area < PRECISION.epsilon

def get_distance_traveled(circles: list[Circle], debug: bool = False) -> float:
    """Calculate the total distance traveled between circle centers."""
    if not circles:
        return 0.0
    
    total_distance = 0.0
    current = circles[0]
    
    for next_circle in circles[1:]:
        dx = next_circle.x - current.x
        dy = next_circle.y - current.y
        distance = math.sqrt(dx * dx + dy * dy)
        if debug:
            print(f'Distance from ({current.x:.6f}, {current.y:.6f}) to ({next_circle.x:.6f}, {next_circle.y:.6f}): {distance:.6f}')
        total_distance += distance
        current = next_circle
    
    return total_distance
