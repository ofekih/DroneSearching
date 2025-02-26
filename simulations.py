from typing import Callable
from decimal import Decimal, getcontext
import argparse
import time

import shapely
from decimal_math import asin, atan, cos, log2, pi, sin
from utils import EPSILON, Circle, covers_unit_circle, draw_circles, binary_search, get_biggest_uncovered_square, get_distance_traveled

from scipy import optimize

def default_pk(p: Decimal, k: Decimal) -> Decimal:
    """Default pk function that returns p^k"""
    return p ** k

def find_all_pk(p: Decimal, k: Decimal) -> Decimal:
    """pk function that returns p^((k+1)/2) for find-all mode"""
    return p ** ((k + Decimal(1)) / 2)

def get_configuration(args: argparse.Namespace) -> tuple[Callable[[Decimal, Decimal], Decimal], int]:
    """Get pk function and c_multiplier based on arguments."""
    if args.find_all:
        return find_all_pk, 2
    return default_pk, 1

def place_algorithm_4(p: Decimal, pk: Callable[[Decimal, Decimal], Decimal] = default_pk) -> list[Circle]:
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

def place_algorithm_5(p: Decimal, pk: Callable[[Decimal, Decimal], Decimal] = default_pk) -> list[Circle]:
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

def place_algorithm_5_5(p: Decimal, pk: Callable[[Decimal, Decimal], Decimal] = default_pk, final_optimization: bool = True) -> list[Circle]:
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

def place_algorithm_6(p: Decimal, pk: Callable[[Decimal, Decimal], Decimal] = default_pk, final_optimization: bool = True) -> list[Circle]:
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

def place_algorithm_10(p: Decimal, pk: Callable[[Decimal, Decimal], Decimal] = default_pk) -> list[Circle]:
    # algorithm 5 + additional circles
    circles = place_algorithm_6(p, pk)[:-1]
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

def add_centroid_circles(p: Decimal, pk: Callable[[Decimal, Decimal], Decimal], circles: list[Circle]):
    quad_segs = int(pi() / 2 / EPSILON())

    unit_circle = shapely.Point(0, 0).buffer(1, quad_segs)
    circle_polygons = [shapely.Point(float(circle.x), float(circle.y)).buffer(float(circle.r), quad_segs) for circle in circles]
    uncovered_polygons = unit_circle.difference(shapely.unary_union(circle_polygons))

    k = Decimal(len(circles) + 1)

    while True:
        current_radius = pk(p, k)

        if current_radius < EPSILON():
            break

        if uncovered_polygons.area < 1e-5:
            biggest_uncovered_square = get_biggest_uncovered_square(circles)
            if biggest_uncovered_square is None:
                break
            
            new_circle = Circle(
                biggest_uncovered_square.x + biggest_uncovered_square.side_length / 2,
                biggest_uncovered_square.y + biggest_uncovered_square.side_length / 2,
                current_radius
            )
            
            circles.append(new_circle)
            k += 1
            continue
        
        # print(f'Remaining area: {uncovered_polygons.area}')

        largest_polygon = max(uncovered_polygons.geoms, key=lambda p: p.area) if hasattr(uncovered_polygons, 'geoms') else uncovered_polygons
        centroid = largest_polygon.centroid
        new_circle = Circle(Decimal(centroid.x), Decimal(centroid.y), current_radius)
        circles.append(new_circle)
        k += 1

        uncovered_polygons = uncovered_polygons.difference(shapely.Point(float(centroid.x), float(centroid.y)).buffer(float(current_radius), quad_segs))

    return circles, uncovered_polygons.area

# Algorithm 11 helper functions
def create_circles_from_params(params: list[float], k: int, p: Decimal, pk: Callable[[Decimal, Decimal], Decimal]) -> list[Circle]:
    """Helper function to create circles from optimization parameters"""
    circles = []
    
    # First circle is constrained to the -x axis
    dx = Decimal(params[0])
    circle1 = Circle(-dx, Decimal('0'), pk(p, Decimal(1)))
    circles.append(circle1)
    
    # Add the remaining k-1 circles
    for i in range(1, k):
        theta_idx = 2*i - 1
        d_idx = 2*i
        
        theta = Decimal(params[theta_idx])
        d = Decimal(params[d_idx])
        
        circle = Circle(d * cos(theta), d * sin(theta), pk(p, Decimal(i+1)))
        circles.append(circle)
        
    return circles

def objective_function_global(x: list[float], p: Decimal, k: int, pk: Callable[[Decimal, Decimal], Decimal]) -> float:
    """Calculate the objective function value (remaining area) for given parameters"""
    circles = create_circles_from_params(x, k, p, pk)
    _, remaining_area = add_centroid_circles(p, pk, circles)
    return remaining_area

class ObjectiveFunctionWrapper:
    """A pickleable wrapper class for the objective function."""
    def __init__(self, p: Decimal, k: int, pk: Callable[[Decimal, Decimal], Decimal]):
        self.p = p
        self.k = k
        self.pk = pk
    
    def __call__(self, x: list[float]) -> float:
        return objective_function_global(x, self.p, self.k, self.pk)

def optimize_circle_placement(p: Decimal, k: int, pk: Callable[[Decimal, Decimal], Decimal],
                          callback: Callable[[int, float], None] | None = None,
                          optimization_kwargs: dict | None = None):
    """Optimize the placement of k circles with parameter p and radius function pk."""
    # Create bounds for optimization
    # First parameter (dx) is between 0 and 1
    bounds: list[tuple[float, float]] = [(0, 1)]
    
    # For each of the k-1 remaining circles, we need bounds for theta and d
    # theta can be between -pi and pi
    # d can be between 0 and 1
    for _ in range(1, k):
        bounds.extend([(-float(pi()), float(pi())), (0, 1)])
    
    # Create a pickleable objective function
    obj_func = ObjectiveFunctionWrapper(p, k, pk)
    
    # Default optimization parameters
    kwargs = {
        'bounds': bounds,
        'workers': -1,
        'updating': 'deferred',  # Required when using parallel workers
        'seed': 42,  # For reproducibility
        # 'strategy': 'best1bin',  # Good for optimization problems like ours
        # 'tol': 1e-10,  # Tighter tolerance for better optimization
        'mutation': (0.5, 1.0),  # Allow mutation rate to adapt
    }
    
    # Update with any custom parameters
    if optimization_kwargs:
        kwargs.update(optimization_kwargs)
    
    if callback:
        kwargs['callback'] = callback
    
    result = optimize.differential_evolution(obj_func, **kwargs)
    return result

def place_algorithm_11(p: Decimal, pk: Callable[[Decimal, Decimal], Decimal] = default_pk,
                      initial_circles: int = 2,
                      callback: Callable[[int, float], None] | None = None,
                      optimization_kwargs: dict | None = None) -> list[Circle]:
    """Algorithm that uses differential evolution to optimize circle placement.
    
    Places initial_circles optimally, then fills remaining space with centroid circles.
    
    Args:
        p: The scaling parameter p
        pk: The radius function to use (default: p^k)
        initial_circles: Number of initial circles to optimize (default: 2)
        callback: Optional callback function called after each optimization iteration 
                 with (iteration, best_objective_value)
        optimization_kwargs: Optional dictionary of parameters to pass to differential_evolution
    """
    # Run the optimization
    result = optimize_circle_placement(p, initial_circles, pk, callback, optimization_kwargs)

    if not result.success:
        print(f"Warning: Optimization may not have converged: {result.message}")
    
    print(f"Optimization complete in {result.nit} iterations")
    
    # Create the final circles using the optimized parameters
    circles = create_circles_from_params(result.x, initial_circles, p, pk)
    
    # Print the positions of all circles
    for i, circle in enumerate(circles):
        print(f"Circle {i+1}: center=({circle.x:.7f}, {circle.y:.7f}), radius={circle.r:.7f}")
    
    # Add centroid circles
    result = add_centroid_circles(p, pk, circles)
    
    print(f'Final Remaining Area: {result[1]:.6f}')
    
    return circles

def place_algorithm_5_5_no_optimization(p: Decimal, pk: Callable[[Decimal, Decimal], Decimal]) -> list[Circle]:
    return place_algorithm_5_5(p, pk, final_optimization=False)

def place_algorithm_6_no_optimization(p: Decimal, pk: Callable[[Decimal, Decimal], Decimal]) -> list[Circle]:
    return place_algorithm_6(p, pk, final_optimization=False)

def get_placement_algorithm(algorithm: float) -> Callable[[Decimal, Callable[[Decimal, Decimal], Decimal]], list[Circle]]:
    """Get the appropriate placement algorithm based on argument."""
    if algorithm == 4:
        return place_algorithm_4
    elif algorithm == 5:
        return place_algorithm_5
    elif algorithm == 5.5:
        return place_algorithm_5_5_no_optimization
    elif algorithm == 5.75:
        return place_algorithm_5_5
    elif algorithm == 6:
        return place_algorithm_6_no_optimization
    elif algorithm == 6.5:
        return place_algorithm_6
    elif algorithm == 10:
        return place_algorithm_10
    elif algorithm == 11:
        return place_algorithm_11
    else:
        raise ValueError("Invalid algorithm selection. Choose 4, 5, 5.5, 5.75, 6, 6.5, or 10.")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=float, required=True, choices=[4, 5, 5.5, 5.75, 6, 6.5, 10, 11],
                       help='4: Progressive Chords, 5: Reordered Chords Placement, 5.5: Central + Chords, 5.75: Central + Chords w/ Final Adjustment, 6: Central + Optimized Chords, 6.5: Central + Optimized Chords w/ Final Adjustment, 10: Reordered Chords + Square Fill')
    parser.add_argument('--find-all', action='store_true', help='Use p^((k+1)/2) for radius calculation')
    parser.add_argument('--precision', type=int, default=5, help='Decimal precision for calculations (minimum 1)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()
    
    # Validate precision after parsing
    if args.precision < 1:
        parser.error('Precision must be at least 1')
        
    return args

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