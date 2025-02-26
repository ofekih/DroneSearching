from typing import Callable, Any, Optional
import argparse
import time
import math
from dataclasses import dataclass

import shapely
from utils import PRECISION, Circle, covers_unit_circle, draw_circles, binary_search, get_biggest_uncovered_square, get_distance_traveled

from scipy import optimize
from scipy.optimize._optimize import OptimizeResult

def default_pk(p: float, k: int) -> float:
    """Default pk function that returns p^k"""
    return p ** k

def find_all_pk(p: float, k: int) -> float:
    """pk function that returns p^((k + 1) / 2)"""
    return p ** ((k + 1) / 2)

def get_configuration(args: argparse.Namespace) -> tuple[Callable[[float, int], float], int]:
    """Get pk function and c_multiplier based on arguments."""
    if args.find_all:
        return find_all_pk, 2
    return default_pk, 1

def place_algorithm_4(p: float, pk: Callable[[float, int], float] = default_pk) -> list[Circle]:
    circles: list[Circle] = []

    current_angle = 0
    k = 1

    while current_angle < 2 * math.pi:
        current_radius = pk(p, k)
        if current_radius < PRECISION.epsilon:
            return [] # failure

        current_coord = (math.cos(current_angle), 
                        math.sin(current_angle))
        current_angle += 2 * math.asin(current_radius)
        # current_angle = min(current_angle + 2 * asin(current_radius), 2 * pi())

        next_coord = (math.cos(current_angle), 
                     math.sin(current_angle))
        
        new_circle = Circle(
            (current_coord[0] + next_coord[0]) / 2,
            (current_coord[1] + next_coord[1]) / 2,
            current_radius
        )

        circles.append(new_circle)
        k += 1

    return circles

def place_algorithm_5(p: float, pk: Callable[[float, int], float] = default_pk) -> list[Circle]:
    chords: list[tuple[float, float]] = []

    current_angle = 0
    k = 1
    while current_angle < 2 * math.pi:
        current_radius = pk(p, k)
        if current_radius < PRECISION.epsilon:
            return [] # failure

        chord_angle = 2 * math.asin(current_radius)
        chords.append((chord_angle, current_radius))
        current_angle += chord_angle

        k += 1

    CCW: list[tuple[float, float]] = [chords[0]]
    CW: list[tuple[float, float]] = []
    CCW_sum = CW_sum = 0

    for chord_angle, radius in chords[:0:-1]:
        if CCW_sum < CW_sum:
            CCW.append((chord_angle, radius))
            CCW_sum += chord_angle
        else:
            CW.append((chord_angle, radius))
            CW_sum += chord_angle

    circles: list[Circle] = []

    current_angle = 0
    for chord_angle, radius in CCW + CW[::-1]:
        current_coord = (math.cos(current_angle), 
                        math.sin(current_angle))
        current_angle += chord_angle

        next_coord = (math.cos(current_angle), 
                     math.sin(current_angle))

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
    d = math.sqrt((x1-x0)**2 + (y1-y0)**2)
    
    # Check intersection conditions
    if d > r0 + r1:  # Non intersecting
        return None
    if d < abs(r0-r1):  # One circle within other
        return None
    if d == 0 and r0 == r1:  # Coincident circles
        return None

    # Calculate intersection points
    a = (r0**2 - r1**2 + d**2)/(2*d)
    h = math.sqrt(r0**2 - a**2)
    
    x2 = x0 + a*(x1-x0)/d   
    y2 = y0 + a*(y1-y0)/d   
    
    x3 = x2 + h*(y1-y0)/d     
    y3 = y2 - h*(x1-x0)/d 

    x4 = x2 - h*(y1-y0)/d
    y4 = y2 + h*(x1-x0)/d
    
    return ((x3, y3), (x4, y4))

def place_algorithm_5_5(p: float, pk: Callable[[float, int], float] = default_pk, final_optimization: bool = True) -> list[Circle]:
    """Central Plus Chords placement algorithm.
    Places a central circle and places surrounding circles."""
    circles: list[Circle] = []
    
    # Place central circle
    central_radius = pk(p, 1)
    circles.append(Circle(0, 0, central_radius))
    
    # Place surrounding circles
    k = 2
    current_angle = 0
    while current_angle < 2 * math.pi:
        current_radius = pk(p, k)
        if current_radius < PRECISION.epsilon:
            return [] # failure
        
        current_coord = (math.cos(current_angle), math.sin(current_angle))
        current_angle = min(current_angle + 2 * math.asin(current_radius), 2 * math.pi)

        if final_optimization and current_angle >= 2 * math.pi:
            current_radius = pk(p, k - 1)
            points = get_intersections(circles[0], circles[-1])
            if points:
                current_coord = max(points, key=lambda p: p[0])

        next_coord = (math.cos(current_angle), math.sin(current_angle))
        
        new_circle = Circle(
            (current_coord[0] + next_coord[0]) / 2,
            (current_coord[1] + next_coord[1]) / 2,
            current_radius
        )

        circles.append(new_circle)
        k += 1
    
    return circles

def compute_R_T(R: float, b: float):
    a = b * R
    c = 1 - R
    d = math.sqrt(a * b + c ** 2)
    q = (b + c + d) / 2

    return b * c * d / (4 * math.sqrt(q * (q - b) * (q - c) * (q - d)))

def compute_x2(R: float, r: float):
    theta = math.asin(r)
    beta = math.pi / 2 - theta
    sin_beta = math.sin(beta)
    c = 1 - R
    z = c * sin_beta
    y = math.sqrt(c ** 2 * (1 - sin_beta ** 2))

    return (r - y) ** 2 + z ** 2

def place_algorithm_6(p: float, pk: Callable[[float, int], float] = default_pk, final_optimization: bool = True) -> list[Circle]:
    """Central Plus Optimized Chords placement algorithm.
    Places a central circle and optimizes the placement of surrounding circles."""
    circles: list[Circle] = []
    
    # Place central circle
    central_radius = pk(p, 1)
    circles.append(Circle(0, 0, central_radius))
    
    # Place surrounding circles
    k = 2
    current_angle = 0
    while current_angle < 2 * math.pi:
        current_radius = pk(p, k)
        if current_radius < PRECISION.epsilon:
            return [] # failure

        theta = 0
        points = None
        
        if final_optimization and current_angle + 2 * math.asin(current_radius) >= 2 * math.pi:
            current_radius = pk(p, k - 1)
            points = get_intersections(circles[1], circles[-1])
        
        if points:
            remaining_angle = math.pi - current_angle / 2
            next_coord = (math.cos(-remaining_angle), math.sin(-remaining_angle))

            current_coord = max(points, key=lambda p: p[0])
            new_circle_center = ((current_coord[0] + next_coord[0]) / 2, (current_coord[1] + next_coord[1]) / 2)
            theta = 2 * math.pi
        else:
            R = circles[0].r
            r = current_radius
            B = (1 - R) / 2
            if current_radius ** 2 < compute_x2(circles[0].r, current_radius) and B < current_radius:
                theta = math.atan(math.sqrt(4 * r ** 2 - (1 - R) ** 2) / (R + 1))
                distance_from_center = (1 + R) / (2 * math.cos(theta))
                new_circle_center = (distance_from_center * math.cos(current_angle + theta), distance_from_center * math.sin(current_angle + theta))
            else:
                b = 2 * current_radius
                theta = math.asin(b / 2)

                current_coord = (math.cos(current_angle), math.sin(current_angle))
                next_coord = (math.cos(current_angle + 2 * theta), math.sin(current_angle + 2 * theta))
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

def place_algorithm_10(p: float, pk: Callable[[float, int], float] = default_pk) -> list[Circle]:
    # algorithm 5 + additional circles
    circles = place_algorithm_6(p, pk)[:-1]
    # circles = []

    k = len(circles) + 1
    while True:
        current_radius = pk(p, k)

        biggest_uncovered_square = get_biggest_uncovered_square(circles)
        if biggest_uncovered_square is None:
            break

        if current_radius < PRECISION.epsilon:
            return []

        new_circle = Circle(
            biggest_uncovered_square.x + biggest_uncovered_square.side_length / 2,
            biggest_uncovered_square.y + biggest_uncovered_square.side_length / 2,
            current_radius
        )
        
        circles.append(new_circle)
        k += 1

    return circles

def get_empty_area(circles: list[Circle]) -> float:
    circle_polygons = [PRECISION.get_circle_polygon(circle) for circle in circles]
    uncovered_polygons = PRECISION.unit_circle_polygon.difference(shapely.union_all(circle_polygons)) # type: ignore

    return uncovered_polygons.area

def add_centroid_circles(p: float, pk: Callable[[float, int], float], circles: list[Circle]) -> tuple[list[Circle], float]:
    circle_polygons = [PRECISION.get_circle_polygon(circle) for circle in circles]
    uncovered_polygons = PRECISION.unit_circle_polygon.difference(shapely.union_all(circle_polygons)) # type: ignore

    k = len(circles) + 1

    while True:
        current_radius = pk(p, k)

        if current_radius < PRECISION.epsilon:
            break

        if uncovered_polygons.area < PRECISION.epsilon ** 2:
            biggest_uncovered_square = get_biggest_uncovered_square(circles)
            if biggest_uncovered_square is None:
                return circles, 0
            
            new_circle = Circle(
                biggest_uncovered_square.x + biggest_uncovered_square.side_length / 2,
                biggest_uncovered_square.y + biggest_uncovered_square.side_length / 2,
                current_radius
            )
            
            circles.append(new_circle)
            k += 1
            continue
        
        # print(f'Remaining area: {uncovered_polygons.area}')

        largest_geom = max(uncovered_polygons.geoms, key=lambda g: g.area) if hasattr(uncovered_polygons, 'geoms') else uncovered_polygons # type: ignore
        centroid = largest_geom.centroid # type: ignore
        new_circle = Circle(centroid.x, centroid.y, current_radius) # type: ignore
        circles.append(new_circle)
        k += 1

        uncovered_polygons = uncovered_polygons.difference(PRECISION.get_circle_polygon(new_circle)) # type: ignore

    return circles, uncovered_polygons.area

# Algorithm 11 helper functions
def create_circles_from_params(params: list[float], k: int, p: float, pk: Callable[[float, int], float]) -> list[Circle]:
    """Helper function to create circles from optimization parameters"""
    circles: list[Circle] = []
    
    # First circle is constrained to the -x axis
    dx = params[0]
    circle1 = Circle(-dx, 0, pk(p, 1))
    circles.append(circle1)
    
    # Add the remaining k-1 circles
    for i in range(1, k):
        theta_idx = 2*i - 1
        d_idx = 2*i
        
        theta = params[theta_idx]
        d = params[d_idx]
        
        circle = Circle(d * math.cos(theta), d * math.sin(theta), pk(p, i+1))
        circles.append(circle)
        
    return circles

def params_from_created_circles(circles: list[Circle]) -> list[float]:
    """Convert a list of Circle objects into optimization parameters.
    
    For the first circle, extracts the distance from origin on -x axis.
    For remaining circles, extracts angle (theta) and distance (d) from origin.
    
    Args:
        circles: List of Circle objects to convert into parameters
        
    Returns:
        List of parameters in the format [dx, theta1, d1, theta2, d2, ...]
    """
    params: list[float] = []
    
    # First circle: distance from origin on -x axis
    params.append(-circles[0].x)  # Convert to positive distance
    
    # Remaining circles: extract theta and distance
    for circle in circles[1:]:
        x, y = circle.x, circle.y
        theta = math.atan(y/x) if x != 0 else math.pi/2 if y > 0 else -math.pi/2
        if x < 0:  # atan gives angle in [-pi/2, pi/2], adjust for x < 0
            theta = theta + math.pi if y >= 0 else theta - math.pi
        d = math.sqrt(x*x + y*y)
        params.extend([theta, d])
    
    return params

def objective_function_global(x: list[float], p: float, k: int, pk: Callable[[float, int], float]) -> float:
    """Calculate the objective function value (remaining area) for given parameters"""
    circles = create_circles_from_params(x, k, p, pk)
    _, remaining_area = add_centroid_circles(p, pk, circles)
    return remaining_area

@dataclass
class ObjectiveFunctionWrapper:
    """A pickleable wrapper class for the objective function."""
    p: float
    k: int
    pk: Callable[[float, int], float]
    
    def __call__(self, x: list[float]) -> float:
        return objective_function_global(x, self.p, self.k, self.pk)

def optimize_circle_placement(p: float, k: int, pk: Callable[[float, int], float],
                          callback: Optional[Callable[[list[float], float], bool]] = None,
                          optimization_kwargs: Optional[dict[str, Any]] = None) -> OptimizeResult:
    """Optimize the placement of k circles with parameter p and radius function pk."""
    # Create bounds for optimization
    # First parameter (dx) is between 0 and 1
    bounds: list[tuple[float, float]] = [(0, 1)]
    
    # For each of the k-1 remaining circles, we need bounds for theta and d
    # theta can be between -pi and pi
    # d can be between 0 and 1
    for _ in range(1, k):
        bounds.extend([(-math.pi, math.pi), (0, 1)])
    
    obj_func = ObjectiveFunctionWrapper(p, k, pk)

    # Default optimization parameters
    kwargs: dict[str, Any] = {
        'bounds': bounds,
        'seed': 0,  # For reproducibility
        'mutation': (0.5, 1.0),  # Allow mutation rate to adapt
    }
    
    # Update with any custom parameters
    if optimization_kwargs:
        kwargs.update(optimization_kwargs)
    
    if callback:
        kwargs['callback'] = callback
    
    result = optimize.differential_evolution(obj_func, **kwargs) # type: ignore
    return result

def place_algorithm_11(p: float, pk: Callable[[float, int], float] = default_pk,
                      initial_circles: int = 6,
                      optimization_kwargs: Optional[dict[str, Any]] = None) -> list[Circle]:
    """Algorithm that uses differential evolution to optimize circle placement."""
    start_time = time.time()
    best_score = float('inf')
    iterations = 0

    def progress_callback(xk: list[float], _: float) -> bool:
        nonlocal best_score, iterations
        iterations += 1
        score = objective_function_global(xk, p, initial_circles, pk)
        if score < best_score:
            best_score = score
            elapsed = time.time() - start_time
            print(f"Iteration {iterations}: New best score = {best_score:.6f} (elapsed: {elapsed:.2f}s)")
        return False

    # Run the optimization
    result: OptimizeResult = optimize_circle_placement(p, initial_circles, pk, progress_callback, optimization_kwargs)

    if not result.success:
        print(f"Warning: Optimization may not have converged: {result.message}")
    
    total_time = time.time() - start_time
    print(f"Optimization complete in {result.nit} iterations ({total_time:.2f}s)")
    
    # Create the final circles using the optimized parameters
    circles = create_circles_from_params(result.x, initial_circles, p, pk)
    
    # Print the positions of all circles
    for i, circle in enumerate(circles):
        print(f"Circle {i+1}: center=({circle.x:.7f}, {circle.y:.7f}), radius={circle.r:.7f}")
    
    # Add centroid circles
    circles, area = add_centroid_circles(p, pk, circles)
    
    print(f'Final Remaining Area: {area:.6f}')
    
    return circles

def place_algorithm_5_5_no_optimization(p: float, pk: Callable[[float, int], float]) -> list[Circle]:
    return place_algorithm_5_5(p, pk, final_optimization=False)

def place_algorithm_6_no_optimization(p: float, pk: Callable[[float, int], float]) -> list[Circle]:
    return place_algorithm_6(p, pk, final_optimization=False)

def get_placement_algorithm(algorithm: float) -> Callable[[float, Callable[[float, int], float]], list[Circle]]:
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
    place_algorithm: Callable[[float, Callable[[float, int], float]], list[Circle]],
    pk: Callable[[float, int], float]
) -> Callable[[float], tuple[bool, list[Circle]]]:
    """Create an evaluation function for binary search."""
    def evaluate(p: float) -> tuple[bool, list[Circle]]:
        circles = place_algorithm(p, pk)
        return covers_unit_circle(circles), circles
    return evaluate

def run_search(evaluator: Callable[[float], tuple[bool, list[Circle]]], debug: bool = False) -> tuple[float, list[Circle]]:
    """Run binary search with the given evaluator."""
    return binary_search(0, 1, evaluator, debug=debug)

def calculate_result(p: float, c_multiplier: int) -> float:
    """Calculate final result using p and c_multiplier."""
    return c_multiplier / math.log2(1 / p)

def run_simulation(
    algorithm: float = 4.0,
    find_all: bool = False,
    precision: int = 5,
    debug: bool = False
) -> tuple[float, float, float, list[Circle], float]:
    """
    Run the circle packing simulation with the specified parameters.
    
    Args:
        algorithm (float): Algorithm choice (4, 5, 5.5, 6, or 6.5)
        find_all (bool): Whether to use p^((k+1)/2) for radius calculation
        precision (int): Decimal precision for calculations (minimum 1)
        debug (bool): Enable debug output
    
    Returns:
        tuple[float, float, list[Circle], float]: (p value, c value, list of circles, CPU time)
    """
    if precision < 1:
        raise ValueError('Precision must be at least 1')
    
    # Set precision to double the requested precision for internal calculations
    # calc_precision = (precision + 2) * 2
    
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
    # main()
    # p = 2 ** (-1 / 2.5)
    PRECISION.set_precision(7)
    print(PRECISION.epsilon)
    # p = 0.76
    p = 0.665
    # p = 0.66
    initial_circles = 2
    # initial_circles = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(initial_circles)
    circles = place_algorithm_11(p, initial_circles=initial_circles, pk=find_all_pk)
    # circles = [Circle(x=-0.4660305, y=0, r=0.7578583), Circle(x=0.6219811, y=0.2293612, r=0.5743492), Circle(x=0.3799721, y=-0.6418602, r=0.4352753), Circle(x=0.09652912, y=0.8269215, r=0.3298770), Circle(x=-0.1954859, y=-0.8560651, r=0.2500000), Circle(x=-0.3625750, y=0.8458253, r=0.1894646), Circle(x=0.8360550, y=-0.3959104, r=0.1435873), Circle(x=-0.5261192, y=-0.8089158, r=0.1088188), Circle(x=0.480869862587108898299703696466167457401752471923828125, y=0.83068605528566052953465259633958339691162109375, r=0.08246926), Circle(x=-0.59274082610672007565000285467249341309070587158203125, y=0.77314890488008958246979318573721684515476226806640625, r=0.06250001), Circle(x=0.05955299818111030318856791154757956974208354949951171875, y=-0.97322066399223505772653197709587402641773223876953125, r=0.04736615), Circle(x=0.945071667223251754563762006000615656375885009765625, y=-0.267485111240606254767726568388752639293670654296875, r=0.03589683), Circle(x=-0.64508836223532295406357661704532802104949951171875, y=-0.7483326183722491808936183588230051100254058837890625, r=0.02720471), Circle(x=0.8187915289415081954160768873407505452632904052734375, y=-0.55365355153100981322467077916371636092662811279296875, r=0.02061732), Circle(x=-0.66262275981678431246990612635272555053234100341796875, y=0.7393680566860967307007967974641360342502593994140625, r=0.01562501), Circle(x=0.57264705977377372558834167648456059396266937255859375, y=0.80989263449354031987326152375317178666591644287109375, r=0.01184154), Circle(x=-0.20882603562105661598735650841263122856616973876953125, y=0.97027272385723339898078165788319893181324005126953125, r=0.008974209), Circle(x=0.424209419172794033325857299132621847093105316162109375, y=0.89909879687231797351643081128713674843311309814453125, r=0.006801179), Circle(x=0.110286511239986950716485125667531974613666534423828125, y=-0.98936943284950518151532605770626105368137359619140625, r=0.005154330), Circle(x=-0.675947897554812104914390147314406931400299072265625, y=-0.732007983290130948006435573915950953960418701171875, r=0.003906252), Circle(x=0.9684947036287188115721846770611591637134552001953125, y=-0.233445824486877284709152036157320253551006317138671875, r=0.002960385), Circle(x=-0.446946708195753383829895710732671432197093963623046875, y=-0.8895709389850099402252681102254427969455718994140625, r=0.002243553), Circle(x=-0.55448923632514379722380226667155511677265167236328125, y=0.82695865409689961467165630892850458621978759765625, r=0.001700295), Circle(x=-0.446946708195759379034228686577989719808101654052734375, y=-0.88957093898502126450011928682215511798858642578125, r=0.001288583)]
    # circles = [Circle(x=-0.4621705, y=0, r=0.7578583), Circle(x=0.6256425, y=0.2326335, r=0.5743492), Circle(x=0.3799522, y=-0.6394088, r=0.4352753), Circle(x=0.09435549, y=0.8280994, r=0.3298770), Circle(x=-0.1911341, y=-0.8601548, r=0.2500000), Circle(x=-0.3621675, y=0.8498134, r=0.1894646), Circle(x=0.8355603, y=-0.3944838, r=0.1435873), Circle(x=-0.52249459542098286579658861228381283581256866455078125, y=-0.79951779020669666575571454814053140580654144287109375, r=0.1088188), Circle(x=0.478108268017147286510493131572729907929897308349609375, y=0.83226584615214205253863610778353177011013031005859375, r=0.08246926), Circle(x=-0.5923398972211406654508891733712516725063323974609375, y=0.77278031110463363262397251673974096775054931640625, r=0.06250001), Circle(x=0.06440370617754369308993744880353915505111217498779296875, y=-0.9739136913108425996910000321804545819759368896484375, r=0.04736615), Circle(x=0.945172075466456984571550492546521127223968505859375, y=-0.266710364048800219194390592747367918491363525390625, r=0.03589683), Circle(x=-0.64607838948470541762247876249602995812892913818359375, y=-0.7476118730071712459306354503496550023555755615234375, r=0.02720471), Circle(x=0.81929930229130787378011291366419754922389984130859375, y=-0.5526431474161856982618701294995844364166259765625, r=0.02061732), Circle(x=-0.66298605878645922029335224578971974551677703857421875, y=0.73858704278622078209792789493803866207599639892578125, r=0.01562501), Circle(x=0.569612049146654531028843848616816103458404541015625, y=0.81232234708392148103683894078130833804607391357421875, r=0.01184154), Circle(x=-0.446152987284383739652326994473696686327457427978515625, y=-0.8865485790819789269079365112702362239360809326171875, r=0.008974209), Circle(x=-0.2089098823704004515011689591119647957384586334228515625, y=0.971964861975724403464482747949659824371337890625, r=0.006801179), Circle(x=0.421736682759198122649735296363360248506069183349609375, y=0.90055484730787005442920190034783445298671722412109375, r=0.005154330), Circle(x=-0.67710409240092073179795306714368052780628204345703125, y=-0.7307063094435901628997953594080172479152679443359375, r=0.003906252), Circle(x=0.11458748311946047315768026919613475911319255828857421875, y=-0.9894749187324034522816873504780232906341552734375, r=0.002960385), Circle(x=0.96865022960213431613141210618778131902217864990234375, y=-0.2327812901864864947309996523472364060580730438232421875, r=0.002243553), Circle(x=-0.55390670905321937045329150350880809128284454345703125, y=0.8267531015774716163235780186369083821773529052734375, r=0.001700295), Circle(x=0.42173668275920039860693577793426811695098876953125, y=0.90055484730787405123209055091137997806072235107421875, r=0.001288583)]
    
    # works for p = 0.67, find all
    # circles = [Circle(x=-0.4908906, y=0, r=0.67), Circle(x=0.5804569, y=0.2405622, r=0.5484186), Circle(x=0.208870012530215920509846228014794178307056427001953125, y=-0.624393270985229786873560442472808063030242919921875, r=0.4489), Circle(x=-0.0687488043892642564092199108927161432802677154541015625, y=0.77310537179094873661000519859953783452510833740234375, r=0.3674405), Circle(x=-0.393608215151188989278097096757846884429454803466796875, y=-0.791659521973363755620312076644040644168853759765625, r=0.300763), Circle(x=0.7684885770609046762302796196308918297290802001953125, y=-0.42347591574216758569804142098291777074337005615234375, r=0.2461851), Circle(x=-0.5598258887884235424081680321251042187213897705078125, y=0.74140036004681342252098374956403858959674835205078125, r=0.2015112), Circle(x=0.3929231971960387426179295289330184459686279296875, y=0.8388412309319217552427971895667724311351776123046875, r=0.1649440), Circle(x=-0.719611742351108585324936939286999404430389404296875, y=-0.6575628068270826798169537141802720725536346435546875, r=0.1350125), Circle(x=0.9519120239052198950702177171478979289531707763671875, y=-0.2013058037931811650178559602863970212638378143310546875, r=0.1105125), Circle(x=0.6797276989002438707387909744284115731716156005859375, y=-0.69765922557720283503357450172188691794872283935546875, r=0.09045838), Circle(x=-0.10946213991471363813356987293445854447782039642333984375, y=-0.9722348976327073177827742256340570747852325439453125, r=0.07404338), Circle(x=-0.7605164806089919071752092349925078451633453369140625, y=0.62843717938068355266523212776519358158111572265625, r=0.06060712), Circle(x=0.1816497487910462005356038162062759511172771453857421875, y=-0.158394717928385109217970239114947617053985595703125, r=0.04960906), Circle(x=0.57507623236683491629861464389250613749027252197265625, y=0.80242357988825607773009096490568481385707855224609375, r=0.04060677), Circle(x=0.05021068790505144041613760919062769971787929534912109375, y=0.413950344752257615166257664895965717732906341552734375, r=0.03323807), Circle(x=0.26069828357698343612725011553266085684299468994140625, y=0.956030152369351515773132632602937519550323486328125, r=0.02720653), Circle(x=-0.421038711816131117071648759520030580461025238037109375, y=0.8995030400165087147712483783834613859653472900390625, r=0.02226951), Circle(x=0.539648777974118143418991166981868445873260498046875, y=-0.31223545371566785444628067125449888408184051513671875, r=0.01822838), Circle(x=0.614181769376461073051132188993506133556365966796875, y=0.78848548708470478363352640371886081993579864501953125, r=0.01492057), Circle(x=-0.03952937922512121671214657681048265658318996429443359375, y=-0.9988086069087371843266964788199402391910552978515625, r=0.01221301)]

    # not quite 0.66, find all
    # circles = [Circle(x=-0.4830411, y=0, r=0.66), Circle(x=0.5318464, y=0.2416313, r=0.5361865), Circle(x=0.2861341, y=-0.6065730, r=0.4356), Circle(x=-0.03615046, y=0.7759997, r=0.3538831), Circle(x=-0.3231259, y=-0.7974621, r=0.287496), Circle(x=0.8054224, y=-0.3762863, r=0.2335629), Circle(x=-0.5249223, y=0.7555362, r=0.1897474), Circle(x=0.4046515, y=0.8577455, r=0.1541515), Circle(x=-0.6913100, y=-0.6914375, r=0.1252333), Circle(x=-0.01136641, y=-0.9794021, r=0.1017400), Circle(x=0.9682765, y=-0.1338024, r=0.08265395), Circle(x=-0.7370363, y=0.6390875, r=0.06714839), Circle(x=0.7482672, y=-0.6419571, r=0.05455161), Circle(x=0.5791013, y=0.7920574, r=0.04431794), Circle(x=-0.3782377, y=0.9100824, r=0.03600406), Circle(x=0.1705255, y=-0.1736458, r=0.02924984), Circle(x=-0.7942498, y=-0.5902347, r=0.02376268), Circle(x=-0.8009848, y=0.5868187, r=0.01930489), Circle(x=0.9929994, y=-0.04188113, r=0.01568337), Circle(x=0.2742226, y=0.9559963, r=0.01274123), Circle(x=0.62783818683958070305806131727877072989940643310546875, y=0.77326822321228105838741839761496521532535552978515625, r=0.01035102), Circle(x=0.71628027964776486147258083292399533092975616455078125, y=-0.69301439361814309858544902454013936221599578857421875, r=0.008409211), Circle(x=-0.81731057784901739449168189821648411452770233154296875, y=-0.57198732142082653329140384812490083277225494384765625, r=0.006831675), Circle(x=-0.81871758378737402583880111706093885004520416259765625, y=0.5706265440066895511250777417444624006748199462890625, r=0.005550079), Circle(x=0.025271821720933902721828445692153763957321643829345703125, y=0.424918884588280476588550982341985218226909637451171875, r=0.004508906), Circle(x=-0.70887689170958678719358658781857229769229888916015625, y=0.70214588860013826820960503027890808880329132080078125, r=0.003663052), Circle(x=0.99854446726559464853067993317381478846073150634765625, y=-0.024729054068863824678015106428574654273688793182373046875, r=0.002975878), Circle(x=0.6389450658617759071233876966289244592189788818359375, y=0.76799714141834407588049771220539696514606475830078125, r=0.002417615), Circle(x=-0.6115306319420650282836504629813134670257568359375, y=-0.7895569141375948785110949756926856935024261474609375, r=0.001964079), Circle(x=-0.823424844433121361220173639594577252864837646484375, y=-0.56623809834514970962260349551797844469547271728515625, r=0.001595626), Circle(x=-0.8237548305303843410030140148592181503772735595703125, y=0.5659288806692226447836446823203004896640777587890625, r=0.001296292), Circle(x=0.71190112698686214276477812745724804699420928955078125, y=-0.701426679930230445592087562545202672481536865234375, r=0.001053113)]

    # works for p = 0.76, find one
    # circles = [Circle(x=-0.4835255, y=0, r=0.66), Circle(x=0.5318361, y=0.2461311, r=0.5361865), Circle(x=0.2903324, y=-0.6025555, r=0.4356), Circle(x=-0.03487531, y=0.7734550, r=0.3538831), Circle(x=-0.3268369, y=-0.7899857, r=0.287496), Circle(x=0.8097788, y=-0.3717931, r=0.2335629), Circle(x=-0.5255267, y=0.7548169, r=0.1897474), Circle(x=0.4004226, y=0.8674718, r=0.1541515), Circle(x=-0.67945382451224645148357694779406301677227020263671875, y=-0.6751750376275313936247357560205273330211639404296875, r=0.1252333), Circle(x=-0.024939131909031464484627349520451389253139495849609375, y=-0.95977248830864458906120262327021919190883636474609375, r=0.1017400), Circle(x=0.96042654915217517963554882953758351504802703857421875, y=-0.12630901746618850012282564421184360980987548828125, r=0.08265395), Circle(x=-0.7379269962124757054056090055382810533046722412109375, y=0.63629507516187100435445245238952338695526123046875, r=0.06714839), Circle(x=0.74725443286440584866880953995860181748867034912109375, y=-0.63242648475445495392932571121491491794586181640625, r=0.05455161), Circle(x=0.57292595114667521638551761498092673718929290771484375, y=0.79745691569950449828496630289009772241115570068359375, r=0.04431794), Circle(x=-0.3800292752701974752227442877483554184436798095703125, y=0.90688579706415783920903095349785871803760528564453125, r=0.03600406), Circle(x=0.1701712341055969524017399407966877333819866180419921875, y=-0.16927768809544196937366677957470528781414031982421875, r=0.02924984), Circle(x=-0.7964434090271288457785203718231059610843658447265625, y=-0.5907472441865468937294281204231083393096923828125, r=0.02376268), Circle(x=-0.80251702513649281200969198835082352161407470703125, y=0.5854148964206424832212860565050505101680755615234375, r=0.01930489), Circle(x=0.9934478957365551199387709857546724379062652587890625, y=-0.03814209083888808748952214955352246761322021484375, r=0.01568337), Circle(x=0.0817895469029160937513012186173000372946262359619140625, y=-0.99129309792664332956491080039995722472667694091796875, r=0.01274123), Circle(x=0.6212917866577833958530163727118633687496185302734375, y=0.77872669842496000658371713143424130976200103759765625, r=0.01035102), Circle(x=0.7212511615752885685282080885372124612331390380859375, y=-0.68751476264394018045322809484787285327911376953125, r=0.008409211), Circle(x=0.27049894230017679230826388447894714772701263427734375, y=0.9590038643927643047248920993297360837459564208984375, r=0.006831675), Circle(x=-0.8179052782265145982165677196462638676166534423828125, y=-0.57155388699260523122092081393930129706859588623046875, r=0.005550079), Circle(x=-0.70975574740430091846832283408730290830135345458984375, y=0.70007895650846385660059922884101979434490203857421875, r=0.004508906), Circle(x=-0.81969319379565097616335833663470111787319183349609375, y=0.569900458619543304195076416363008320331573486328125, r=0.003663052), Circle(x=-0.352464891500763155551823047062498517334461212158203125, y=0.93384660870944535826509991238708607852458953857421875, r=0.002975878), Circle(x=-0.61599044900595334439685757388360798358917236328125, y=-0.78541558261101107607515814379439689218997955322265625, r=0.002417615), Circle(x=0.99848828905546505030343951148097403347492218017578125, y=-0.02056921686473318910959307004304719157516956329345703125, r=0.001964079), Circle(x=0.09651744527318030508755697383094229735434055328369140625, y=-0.99406620009273638505220560546149499714374542236328125, r=0.001595626), Circle(x=0.024447861688632895049710924695318681187927722930908203125, y=0.42330639558194727012363500762148760259151458740234375, r=0.001296292), Circle(x=0.6322286034724895475989114856929518282413482666015625, y=0.77369357982573283560867594133014790713787078857421875, r=0.001053113)]
    # # # # for square in get_all_uncovered_squares(circles):
    # # # #     print(square)
    # # # squares = list(get_all_uncovered_squares(circles))
    square = get_biggest_uncovered_square(circles)
    print(circles, len(circles), get_empty_area(circles), covers_unit_circle(circles))
    print(square)

    # draw_circles(circles, squares=[], title='Algorithm 11')
