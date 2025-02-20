from decimal import Decimal, getcontext
from functools import cache
from typing import NamedTuple, Callable, TypeVar
import matplotlib.pyplot as plt

OKABE_COLORS = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=OKABE_COLORS) # type: ignore

class Circle(NamedTuple):
    x: Decimal
    y: Decimal
    r: Decimal

class Square(NamedTuple):
    x: Decimal
    y: Decimal
    side_length: Decimal

UNIT_CIRCLE = Circle(Decimal(0), Decimal(0), Decimal(1))

def draw_circles(circles: list[Circle]) -> None:
    _, ax = plt.subplots()  # type: ignore

    # Draw unit circle with dashed lines in black
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, linestyle='--', color='black')) # type: ignore

    for i, circle in enumerate(sorted(circles, key=lambda circle: circle.r, reverse=True)):
        color = OKABE_COLORS[(i + 1) % len(OKABE_COLORS)]
        ax.add_patch(plt.Circle((circle.x, circle.y), circle.r, fill=False, color=color)) # type: ignore
        ax.text(float(circle.x), float(circle.y), str(i + 1), # type: ignore 
            horizontalalignment='center', verticalalignment='center', color=color,
            fontsize=18)

    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    plt.show() # type: ignore

@cache
def EPSILON() -> Decimal:
    return Decimal('1') / (Decimal('10') ** Decimal(str(getcontext().prec // 2)))

def covers_unit_circle(circles: list[Circle]) -> bool:
    def is_point_covered(circle: Circle, x: Decimal, y: Decimal) -> bool:
        return (x - circle.x) ** 2 + (y - circle.y) ** 2 <= circle.r ** 2
    
    def is_fully_covered(square: Square, circle: Circle) -> bool:
        return is_point_covered(circle, square.x, square.y) and \
               is_point_covered(circle, square.x + square.side_length, square.y) and \
               is_point_covered(circle, square.x, square.y + square.side_length) and \
               is_point_covered(circle, square.x + square.side_length, square.y + square.side_length)

    def is_point_covered_by_any(circles: list[Circle], x: Decimal, y: Decimal) -> bool:
        return any(is_point_covered(circle, x, y) for circle in circles)

    # (x, y) are the bottom left coordinates of the square
    def is_square_covered(square: Square) -> bool:
        if square.side_length < EPSILON():
            return True
        
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
        subsquares = [
            Square(x, y, new_side_length),
            Square(x + new_side_length, y, new_side_length),
            Square(x, y + new_side_length, new_side_length),
            Square(x + new_side_length, y + new_side_length, new_side_length)
        ]
        return all(is_square_covered(subsquare) for subsquare in subsquares)

    return all(is_square_covered(Square(Decimal(x), Decimal(y), Decimal(1))) 
              for x, y in [(-1, -1), (-1, 0), (0, -1), (0, 0)])

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
