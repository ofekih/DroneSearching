from collections import deque
from decimal import Decimal, getcontext
from functools import cache
from typing import Generator, NamedTuple, Callable, TypeVar, TypedDict
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle as PltCircle

OKABE_COLORS = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=OKABE_COLORS) # type: ignore
plt.rcParams['text.usetex'] = True

class Circle(NamedTuple):
    x: Decimal
    y: Decimal
    r: Decimal

class Square(NamedTuple):
    x: Decimal
    y: Decimal
    side_length: Decimal

class HorizontalLine(NamedTuple):
    start: Decimal
    end: Decimal
class CirclesPlotKwargs(TypedDict, total=False):
    title: str | None
    p: Decimal
    c: Decimal
    cpu_time: float

UNIT_CIRCLE = Circle(Decimal(0), Decimal(0), Decimal(1))

def get_circles_plot(circles: list[Circle], *,
                    title: str | None = None,
                    p: Decimal | None = None,
                    c: Decimal | None = None,
                    ct: Decimal | None = None,
                    cpu_time: float | None = None,
                    ax: Axes | None = None,
                    squares: list[Square] = []):
    """Plot circles on either a new figure or an existing axes."""
    if ax is None:
        fig, ax = plt.subplots(1, 1) # type: ignore
    else:
        fig = ax.figure
    
    # Draw unit circle with dashed lines in black
    ax.add_patch(PltCircle((0, 0), 1, fill=False, linestyle='--', color='black'))

    # Draw the circles
    for i, circle in enumerate(sorted(circles, key=lambda circle: circle.r, reverse=True)):
        color = OKABE_COLORS[(i + 1) % len(OKABE_COLORS)]
        ax.add_patch(PltCircle((float(circle.x), float(circle.y)), float(circle.r), fill=False, color=color))
        ax.text(float(circle.x), float(circle.y), str(i + 1), # type: ignore
                horizontalalignment='center', verticalalignment='center',
                color=color, fontsize=18)
        
    for square in squares:
        ax.add_patch(plt.Rectangle((float(square.x), float(square.y)), float(square.side_length), float(square.side_length), fill=False, color='black')) # type: ignore

    # Add statistics if provided
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

    if title:
        ax.set_title(title, pad=20) # type: ignore

    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    
    return fig, ax 

def draw_circles(circles: list[Circle], *,
                title: str | None = None,
                p: Decimal | None = None,
                c: Decimal | None = None,
                ct: Decimal | None = None,
                cpu_time: float | None = None,
                squares: list[Square] = []) -> None:
    """Draw a single set of circles."""
    get_circles_plot(circles, title=title, p=p, c=c, ct=ct, cpu_time=cpu_time, squares=squares)
    plt.show() # type: ignore

@cache
def EPSILON() -> Decimal:
    return Decimal('1') / (Decimal('10') ** Decimal(str(getcontext().prec // 2)))

def get_horizontal_line(circle: Circle, y: Decimal):
    """Get the horizontal line that intersects the circle at y."""
    y_offset = circle.y - y

    if abs(y_offset) > circle.r:
        return None
    
    x_offset = (circle.r ** 2 - y_offset ** 2).sqrt()

    return HorizontalLine(circle.x - x_offset, circle.x + x_offset)

def get_line_union(lines: list[HorizontalLine]) -> list[HorizontalLine]:
    """Get the union of a list of horizontal lines."""
    if not lines:
        return []

    lines.sort(key=lambda line: line.start)
    union: list[HorizontalLine] = []
    current_line = lines[0]

    for line in lines[1:]:
        if line.start <= current_line.end:
            current_line = HorizontalLine(current_line.start, max(current_line.end, line.end))
        else:
            union.append(current_line)
            current_line = line

    union.append(current_line)
    return union

def do_circles_cover_unit_circle(circles: list[Circle], y: Decimal) -> bool:
    lines = [line for circle in circles for line in [get_horizontal_line(circle, y)] if line is not None]
    union = get_line_union(lines)

    unit_circle_line = get_horizontal_line(UNIT_CIRCLE, y)

    if unit_circle_line is None:
        return True
    
    return any(union_line.start <= unit_circle_line.start and union_line.end >= unit_circle_line.end for union_line in union)

def covers_unit_circle_2(circles: list[Circle]) -> bool:
    y = Decimal(-1)
    # circles.sort(key=lambda circle: circle.x + circle.r, reverse=True)
    while y < 1:
        if not do_circles_cover_unit_circle(circles, y):
            return False
        y += EPSILON()

    return True

def is_point_covered(circle: Circle, x: Decimal, y: Decimal) -> bool:
    return (x - circle.x) ** 2 + (y - circle.y) ** 2 <= circle.r ** 2

def is_fully_covered(square: Square, circle: Circle) -> bool:
    return is_point_covered(circle, square.x, square.y) and \
            is_point_covered(circle, square.x + square.side_length, square.y) and \
            is_point_covered(circle, square.x, square.y + square.side_length) and \
            is_point_covered(circle, square.x + square.side_length, square.y + square.side_length)

def is_point_covered_by_any(circles: list[Circle], x: Decimal, y: Decimal) -> bool:
    return any(is_point_covered(circle, x, y) for circle in circles)

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

    if new_side_length < EPSILON():
        return True

    subsquares = [
        Square(x, y, new_side_length),
        Square(x + new_side_length, y, new_side_length),
        Square(x, y + new_side_length, new_side_length),
        Square(x + new_side_length, y + new_side_length, new_side_length)
    ]
    return all(is_square_covered(circles, subsquare) for subsquare in subsquares)

def covers_unit_circle(circles: list[Circle]) -> bool:
    # (x, y) are the bottom left coordinates of the square
    return all(is_square_covered(circles, Square(Decimal(x), Decimal(y), Decimal(1))) 
              for x, y in [(-1, -1), (-1, 0), (0, -1), (0, 0)])

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

        if new_side_length < EPSILON():
            return

        subsquares = [
            Square(x, y, new_side_length),
            Square(x + new_side_length, y, new_side_length),
            Square(x, y + new_side_length, new_side_length),
            Square(x + new_side_length, y + new_side_length, new_side_length)
        ]
        for subsquare in subsquares:
            yield from get_uncovered_squares(subsquare)

    yield from get_uncovered_squares(Square(Decimal(-1), Decimal(-1), Decimal(1)))
    yield from get_uncovered_squares(Square(Decimal(-1), Decimal(0), Decimal(1)))
    yield from get_uncovered_squares(Square(Decimal(0), Decimal(-1), Decimal(1)))
    yield from get_uncovered_squares(Square(Decimal(0), Decimal(0), Decimal(1)))

def get_biggest_uncovered_square(circles: list[Circle]):
    # use BFS instead of DFS, first square found is guaranteed to be the biggest

    q = deque([Square(Decimal(-1), Decimal(-1), Decimal(1)), Square(Decimal(-1), Decimal(0), Decimal(1)),
               Square(Decimal(0), Decimal(-1), Decimal(1)), Square(Decimal(0), Decimal(0), Decimal(1))])
    
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

        if new_side_length < EPSILON():
            continue

        subsquares = [
            Square(x, y, new_side_length),
            Square(x + new_side_length, y, new_side_length),
            Square(x, y + new_side_length, new_side_length),
            Square(x + new_side_length, y + new_side_length, new_side_length)
        ]

        q.extend(subsquares)

    return None

def get_distance_traveled(circles: list[Circle], debug: bool = False):
    # D(n) = max(dist to get to kth circle + D(r_k * n))
    # max(d_k / (1 - r_k))

    circles.sort(key=lambda circle: circle.r, reverse=True)
    distance = Decimal(0)
    current_point = (Decimal(0), Decimal(0))

    max_ct = Decimal(0)

    for circle in circles:
        x, y, r = circle
        distance_to_circle = ((x - current_point[0]) ** 2 + (y - current_point[1]) ** 2).sqrt()
        distance += distance_to_circle

        ct = distance / (1 - r)

        # if debug:
        #     print(f"Circle {circles.index(circle) + 1}: {distance}, {r} => {ct}") 

        max_ct = max(max_ct, ct)
        current_point = (x, y)

    return max_ct

T = TypeVar('T')
R = TypeVar('R')

def binary_search(
    min_param: Decimal,
    max_param: Decimal,
    evaluate: Callable[[Decimal], tuple[bool, R]],
    debug: bool = True
) -> tuple[Decimal, R]:
    """Generic binary search that returns both the parameter and result."""
    # Initial evaluation to ensure result is bound
    mid_param = (min_param + max_param) / 2
    valid, result = evaluate(mid_param)
    found_valid = False

    while max_param - min_param > EPSILON():
        mid_param = (min_param + max_param) / 2
        if debug:
            print(mid_param)
        valid, current_result = evaluate(mid_param)
        if valid:
            max_param = mid_param
            result = current_result
            found_valid = True
        else:
            min_param = mid_param

    if not found_valid:
        raise ValueError("No valid parameter found")

    return min_param, result
