from typing import Callable
from decimal import Decimal, getcontext
import argparse
import time
from decimal_math import asin, atan, cos, log2, pi, sin
from utils import EPSILON, Circle, covers_unit_circle, draw_circles, binary_search, get_biggest_uncovered_square, get_distance_traveled

def place_algorithm_4(p: Decimal, pk: Callable[[Decimal, Decimal], Decimal] = lambda p, k: p**k) -> list[Circle]:
    circles: list[Circle] = []

    current_angle = Decimal('0')
    k = Decimal(1)

    while current_angle < 2 * pi():
        current_radius = pk(p, k)
        if current_radius < EPSILON():
            return [] # failure

        current_coord = (cos(current_angle), 
                        sin(current_angle))
        current_angle += 2 * asin(current_radius)
        # current_angle = min(current_angle + 2 * asin(current_radius), 2 * pi())

        next_coord = (cos(current_angle), 
                     sin(current_angle))
        
        new_circle = Circle(
            (current_coord[0] + next_coord[0]) / 2,
            (current_coord[1] + next_coord[1]) / 2,
            current_radius
        )

        circles.append(new_circle)
        k += 1

    return circles

def place_algorithm_5(p: Decimal, pk: Callable[[Decimal, Decimal], Decimal] = lambda p, k: p**k) -> list[Circle]:
    chords: list[tuple[Decimal, Decimal]] = []

    current_angle = Decimal('0')
    k = Decimal(1)
    while current_angle < 2 * pi():
        current_radius = pk(p, k)
        if current_radius < EPSILON():
            return [] # failure

        chord_angle = 2 * asin(current_radius)
        chords.append((chord_angle, current_radius))
        current_angle += chord_angle

        k += 1

    CCW: list[tuple[Decimal, Decimal]] = [chords[0]]
    CW: list[tuple[Decimal, Decimal]] = []
    CCW_sum = CW_sum = 0

    for chord_angle, radius in chords[:0:-1]:
        if CCW_sum < CW_sum:
            CCW.append((chord_angle, radius))
            CCW_sum += chord_angle
        else:
            CW.append((chord_angle, radius))
            CW_sum += chord_angle

    circles: list[Circle] = []

    current_angle = Decimal('0')
    for chord_angle, radius in CCW + CW[::-1]:
        current_coord = (cos(current_angle), 
                        sin(current_angle))
        current_angle += chord_angle

        next_coord = (cos(current_angle), 
                     sin(current_angle))

        new_circle = Circle(
            (current_coord[0] + next_coord[0]) / 2,
            (current_coord[1] + next_coord[1]) / 2,
            radius
        )

        circles.append(new_circle)

    return circles

def get_intersections(circle1: Circle, circle2: Circle):
    """Calculate the intersection points of two circles.
    Returns tuple of (x3,y3,x4,y4) representing the two intersection points,
    or None if the circles don't intersect properly."""
    x0, y0, r0 = circle1.x, circle1.y, circle1.r
    x1, y1, r1 = circle2.x, circle2.y, circle2.r

    # Calculate distance between circle centers
    d = ((x1-x0)**2 + (y1-y0)**2).sqrt()
    
    # Check intersection conditions
    if d > r0 + r1:  # Non intersecting
        return None
    if d < abs(r0-r1):  # One circle within other
        return None
    if d == 0 and r0 == r1:  # Coincident circles
        return None

    # Calculate intersection points
    a = (r0**2 - r1**2 + d**2)/(2*d)
    h = (r0**2 - a**2).sqrt()
    
    x2 = x0 + a*(x1-x0)/d   
    y2 = y0 + a*(y1-y0)/d   
    
    x3 = x2 + h*(y1-y0)/d     
    y3 = y2 - h*(x1-x0)/d 

    x4 = x2 - h*(y1-y0)/d
    y4 = y2 + h*(x1-x0)/d
    
    return ((x3, y3), (x4, y4))

def place_algorithm_5_5(p: Decimal, pk: Callable[[Decimal, Decimal], Decimal] = lambda p, k: p**k, final_optimization: bool = True) -> list[Circle]:
    """Central Plus Chords placement algorithm.
    Places a central circle and places surrounding circles."""
    circles: list[Circle] = []
    
    # Place central circle
    central_radius = pk(p, Decimal(1))
    circles.append(Circle(Decimal('0'), Decimal('0'), central_radius))
    
    # Place surrounding circles
    k = Decimal(2)
    current_angle = Decimal(0)
    while current_angle < 2 * pi():
        current_radius = pk(p, k)
        if current_radius < EPSILON():
            return [] # failure
        
        current_coord = (cos(current_angle), sin(current_angle))
        current_angle = min(current_angle + 2 * asin(current_radius), 2 * pi())

        if final_optimization and current_angle >= 2 * pi():
            current_radius = pk(p, k - 1)
            points = get_intersections(circles[0], circles[-1])
            if points:
                current_coord = max(points, key=lambda p: p[0])

        next_coord = (cos(current_angle), sin(current_angle))
        
        new_circle = Circle(
            (current_coord[0] + next_coord[0]) / 2,
            (current_coord[1] + next_coord[1]) / 2,
            current_radius
        )

        circles.append(new_circle)
        k += 1
    
    return circles

def compute_R_T(R: Decimal, b: Decimal):
    a = b * R
    c = 1 - R
    d = (a * b + c ** 2).sqrt()
    q = (b + c + d) / 2

    return b * c * d / (4 * (q * (q - b) * (q - c) * (q - d)).sqrt())

def compute_x2(R: Decimal, r: Decimal):
    theta = asin(r)
    beta = pi() / 2 - theta
    sin_beta = sin(beta)
    c = 1 - R
    z = c * sin_beta
    y = (c ** 2 * (1 - sin_beta ** 2)).sqrt()

    return (r - y) ** 2 + z ** 2

def place_algorithm_6(p: Decimal, pk: Callable[[Decimal, Decimal], Decimal] = lambda p, k: p**k, final_optimization: bool = True) -> list[Circle]:
    """Central Plus Optimized Chords placement algorithm.
    Places a central circle and optimizes the placement of surrounding circles."""
    circles: list[Circle] = []
    
    # Place central circle
    central_radius = pk(p, Decimal(1))
    circles.append(Circle(Decimal('0'), Decimal('0'), central_radius))
    
    # Place surrounding circles
    k = Decimal(2)
    current_angle = Decimal(0)
    while current_angle < 2 * pi():
        current_radius = pk(p, k)
        if current_radius < EPSILON():
            return [] # failure

        theta = 0
        points = None
        
        if final_optimization and current_angle + 2 * asin(current_radius) >= 2 * pi():
            current_radius = pk(p, k - 1)
            points = get_intersections(circles[1], circles[-1])
        
        if points:
            remaining_angle = pi() - current_angle / 2
            next_coord = (cos(-remaining_angle), sin(-remaining_angle))

            current_coord = max(points, key=lambda p: p[0])
            new_circle_center = ((current_coord[0] + next_coord[0]) / 2, (current_coord[1] + next_coord[1]) / 2)
            theta = 2 * pi()
        else:
            R = circles[0].r
            r = current_radius
            B = (1 - R) / 2
            if current_radius ** 2 < compute_x2(circles[0].r, current_radius) and B < current_radius:
                theta = atan((4 * r ** 2 - (1 - R) ** 2).sqrt() / (R + 1))
                distance_from_center = (1 + R) / (2 * cos(theta))
                new_circle_center = (distance_from_center * cos(current_angle + theta), distance_from_center * sin(current_angle + theta))
            else:
                b = 2 * current_radius
                theta = asin(b / 2)

                current_coord = (cos(current_angle), sin(current_angle))
                next_coord = (cos(current_angle + 2 * theta), sin(current_angle + 2 * theta))
                new_circle_center = ((current_coord[0] + next_coord[0]) / 2, (current_coord[1] + next_coord[1]) / 2)

        current_angle += 2 * theta

        new_circle = Circle(
            new_circle_center[0],
            new_circle_center[1],
            current_radius
        )

        circles.append(new_circle)
        k += 1
    
    return circles

def place_algorithm_10(p: Decimal, pk: Callable[[Decimal, Decimal], Decimal] = lambda p, k: p**k) -> list[Circle]:
    # algorithm 5 + additional circles
    circles = place_algorithm_5(p, pk)[:-1]
    # circles = []

    k = Decimal(len(circles) + 1)
    while True:
        current_radius = pk(p, k)

        biggest_uncovered_square = get_biggest_uncovered_square(circles)
        if biggest_uncovered_square is None:
            break

        if current_radius < EPSILON():
            return []

        new_circle = Circle(
            biggest_uncovered_square.x + biggest_uncovered_square.side_length / 2,
            biggest_uncovered_square.y + biggest_uncovered_square.side_length / 2,
            current_radius
        )
        
        circles.append(new_circle)
        k += 1

    return circles

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=float, required=True, choices=[4, 5, 5.5, 5.75, 6, 6.5, 10],
                       help='4: Progressive Chords, 5: Reordered Chords Placement, 5.5: Central + Chords, 5.75: Central + Chords w/ Final Adjustment, 6: Central + Optimized Chords, 6.5: Central + Optimized Chords w/ Final Adjustment, 10: Reordered Chords + Square Fill')
    parser.add_argument('--find-all', action='store_true', help='Use p^((k+1)/2) for radius calculation')
    parser.add_argument('--precision', type=int, default=5, help='Decimal precision for calculations (minimum 1)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()
    
    # Validate precision after parsing
    if args.precision < 1:
        parser.error('Precision must be at least 1')
        
    return args

def get_configuration(args: argparse.Namespace) -> tuple[Callable[[Decimal, Decimal], Decimal], int]:
    """Get pk function and c_multiplier based on arguments."""
    if args.find_all:
        return (lambda p, k: p ** ((k + 1) / 2)), 2
    return (lambda p, k: p ** k), 1

def get_placement_algorithm(algorithm: float) -> Callable[[Decimal, Callable[[Decimal, Decimal], Decimal]], list[Circle]]:
    """Get the appropriate placement algorithm based on argument."""
    if algorithm == 4:
        return place_algorithm_4
    elif algorithm == 5:
        return place_algorithm_5
    elif algorithm == 5.5:
        return lambda p, pk: place_algorithm_5_5(p, pk, final_optimization=False)
    elif algorithm == 5.75:
        return place_algorithm_5_5
    elif algorithm == 6:
        return lambda p, pk: place_algorithm_6(p, pk, final_optimization=False)
    elif algorithm == 6.5:
        return place_algorithm_6
    elif algorithm == 10:
        return place_algorithm_10
    else:
        raise ValueError("Invalid algorithm selection. Choose 4, 5, 5.5, 5.75, 6, 6.5, or 10.")

def create_evaluator(
    place_algorithm: Callable[[Decimal, Callable[[Decimal, Decimal], Decimal]], list[Circle]],
    pk: Callable[[Decimal, Decimal], Decimal]
) -> Callable[[Decimal], tuple[bool, list[Circle]]]:
    """Create an evaluation function for binary search."""
    def evaluate(p: Decimal) -> tuple[bool, list[Circle]]:
        circles = place_algorithm(p, pk)
        return covers_unit_circle(circles), circles
    return evaluate

def run_search(evaluator: Callable[[Decimal], tuple[bool, list[Circle]]], debug: bool = False) -> tuple[Decimal, list[Circle]]:
    """Run binary search with the given evaluator."""
    return binary_search(Decimal('0'), Decimal('1'), evaluator, debug=debug)

def calculate_result(p: Decimal, c_multiplier: int) -> Decimal:
    """Calculate final result using p and c_multiplier."""
    return c_multiplier / log2(1 / p)

def run_simulation(
    algorithm: float = 4.0,
    find_all: bool = False,
    precision: int = 5,
    debug: bool = False
) -> tuple[Decimal, Decimal, Decimal, list[Circle], float]:
    """
    Run the circle packing simulation with the specified parameters.
    
    Args:
        algorithm (float): Algorithm choice (4, 5, 5.5, 6, or 6.5)
        find_all (bool): Whether to use p^((k+1)/2) for radius calculation
        precision (int): Decimal precision for calculations (minimum 1)
        debug (bool): Enable debug output
    
    Returns:
        tuple[Decimal, Decimal, list[Circle], float]: (p value, c value, list of circles, CPU time)
    """
    if precision < 1:
        raise ValueError('Precision must be at least 1')
    
    # Set precision to double the requested precision for internal calculations
    calc_precision = (precision + 2) * 2
    getcontext().prec = calc_precision
    
    pk, c_multiplier = get_configuration(argparse.Namespace(find_all=find_all))
    place_algorithm = get_placement_algorithm(algorithm)
    evaluator = create_evaluator(place_algorithm, pk)
    
    start_time = time.process_time()
    p, circles = run_search(evaluator, debug=debug)
    cpu_time = time.process_time() - start_time
    
    c = calculate_result(p, c_multiplier)
    ct = get_distance_traveled(circles, debug=debug)
    return p, c, ct, circles, cpu_time

def main() -> None:
    """Main execution function for command-line usage."""
    args = parse_args()
    
    p, c, ct, circles, cpu_time = run_simulation(
        algorithm=args.algorithm,
        find_all=args.find_all,
        precision=args.precision,
        debug=args.debug
    )

    # Format output with requested precision
    p_str = f"{p:.{args.precision}f}"
    c_str = f"{c:.{args.precision}f}"
    ct_str = f"{ct:.{args.precision}f}"
    print(f"p = {p_str}, c = {c_str}, ct = {ct_str}")
    print(f"CPU Time: {cpu_time:.3f} seconds")
    
    draw_circles(circles,
        title=f"Algorithm {args.algorithm}" + (" (Find All)" if args.find_all else ""),
        p=p,
        c=c,
        ct=ct,
        cpu_time=cpu_time)

if __name__ == '__main__':
    main()
    # getcontext().prec = 7
    # circles = place_algorithm_5(Decimal('0.79'))
    # # for square in get_all_uncovered_squares(circles):
    # #     print(square)
    # squares = list(get_all_uncovered_squares(circles))
    # # square = get_biggest_uncovered_square(circles)
    # draw_circles(circles, squares=squares)
