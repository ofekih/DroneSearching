import sys
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

def get_empty_area(circles: list[Circle]) -> float:
    quad_segs = int(pi() / 2 / EPSILON())
    
    unit_circle = shapely.Point(0, 0).buffer(1, quad_segs)
    circle_polygons = [shapely.Point(float(circle.x), float(circle.y)).buffer(float(circle.r), quad_segs) for circle in circles]
    uncovered_polygons = unit_circle.difference(shapely.unary_union(circle_polygons))

    return uncovered_polygons.area

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

        if uncovered_polygons.area < EPSILON() ** 2:
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

def params_from_created_circles(circles: list[Circle]) -> list[float]:
    """Convert a list of Circle objects into optimization parameters.
    
    For the first circle, extracts the distance from origin on -x axis.
    For remaining circles, extracts angle (theta) and distance (d) from origin.
    
    Args:
        circles: List of Circle objects to convert into parameters
        
    Returns:
        List of parameters in the format [dx, theta1, d1, theta2, d2, ...]
    """
    params = []
    
    # First circle: distance from origin on -x axis
    params.append(float(-circles[0].x))  # Convert to positive distance
    
    # Remaining circles: extract theta and distance
    for circle in circles[1:]:
        x, y = circle.x, circle.y
        theta = float(atan(y/x) if x != 0 else pi()/2 if y > 0 else -pi()/2)
        if x < 0:  # atan gives angle in [-pi/2, pi/2], adjust for x < 0
            theta = theta + float(pi()) if y >= 0 else theta - float(pi())
        d = float((x*x + y*y).sqrt())
        params.extend([theta, d])
    
    return params

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
    
    obj_func = ObjectiveFunctionWrapper(p, k, pk)

    init_circles = [Circle(x=Decimal('-0.4831839'), y=Decimal('0'), r=Decimal('0.66')), Circle(x=Decimal('0.5326323'), y=Decimal('0.2403830'), r=Decimal('0.5361865')), Circle(x=Decimal('0.2863618'), y=Decimal('-0.6066478'), r=Decimal('0.4356')), Circle(x=Decimal('-0.03433004'), y=Decimal('0.7731869'), r=Decimal('0.3538831')), Circle(x=Decimal('-0.3239563'), y=Decimal('-0.7974954'), r=Decimal('0.287496')), Circle(x=Decimal('0.8081841'), y=Decimal('-0.3766654'), r=Decimal('0.2335629')), Circle(x=Decimal('-0.5244322'), y=Decimal('0.7548266'), r=Decimal('0.1897474')), Circle(x=Decimal('0.4069449'), y=Decimal('0.8574731'), r=Decimal('0.1541515')), Circle(x=Decimal('-0.6915893'), y=Decimal('-0.6915203'), r=Decimal('0.1252333')), Circle(x=Decimal('-0.01223258'), y=Decimal('-0.9787379'), r=Decimal('0.1017400')), Circle(x=Decimal('0.96013347214355981851241494950954802334308624267578125'), y=Decimal('-0.1325329110194929660426765849479124881327152252197265625'), r=Decimal('0.08265395')), Circle(x=Decimal('-0.73744977902050934392974568254430778324604034423828125'), y=Decimal('0.63656132198286474732640272122807800769805908203125'), r=Decimal('0.06714839')), Circle(x=Decimal('0.74338972522651214358546667426708154380321502685546875'), y=Decimal('-0.63669140381805855444241615259670652449131011962890625'), r=Decimal('0.05455161')), Circle(x=Decimal('0.58147709815490877804933234074269421398639678955078125'), y=Decimal('0.791391614189733250128711006254889070987701416015625'), r=Decimal('0.04431794')), Circle(x=Decimal('-0.37926889027240295870768704844522289931774139404296875'), y=Decimal('0.90724160408533227606397986164665780961513519287109375'), r=Decimal('0.03600406')), Circle(x=Decimal('0.1691632230171305761867728278957656584680080413818359375'), y=Decimal('-0.172936999541843550165509668659069575369358062744140625'), r=Decimal('0.02924984')), Circle(x=Decimal('-0.79566647221991448901690091588534414768218994140625'), y=Decimal('-0.5908503109011091769531276440829969942569732666015625'), r=Decimal('0.02376268')), Circle(x=Decimal('-0.80228867010446958119018745492212474346160888671875'), y=Decimal('0.58550236675686573306620630319230258464813232421875'), r=Decimal('0.01930489')), Circle(x=Decimal('0.99339431986336723667818660032935440540313720703125'), y=Decimal('-0.044826205813169171798815426654982729814946651458740234375'), r=Decimal('0.01568337')), Circle(x=Decimal('0.276650053580255128604648007240029983222484588623046875'), y=Decimal('0.9544370063530280834385166599531657993793487548828125'), r=Decimal('0.01274123')), Circle(x=Decimal('0.62976741321449380972552489765803329646587371826171875'), y=Decimal('0.771738701686883299402097691199742257595062255859375'), r=Decimal('0.01035102')), Circle(x=Decimal('0.71727387349150639739292500962619669735431671142578125'), y=Decimal('-0.69166122601558199800564352699439041316509246826171875'), r=Decimal('0.008409211')), Circle(x=Decimal('-0.817627265708325889903562710969708859920501708984375'), y=Decimal('-0.57170044192971392060798052625614218413829803466796875'), r=Decimal('0.006831675')), Circle(x=Decimal('-0.7089895127556216092301610842696391046047210693359375'), y=Decimal('0.70046125837105421840789176712860353291034698486328125'), r=Decimal('0.005550079')), Circle(x=Decimal('-0.81962028074573611835518249790766276419162750244140625'), y=Decimal('0.56984671027435596091237357541103847324848175048828125'), r=Decimal('0.004508906')), Circle(x=Decimal('-0.351636885135609034147563534133951179683208465576171875'), y=Decimal('0.934150772525544681457176920957863330841064453125'), r=Decimal('0.003663052')), Circle(x=Decimal('0.0263000539501630527239317558496622950769960880279541015625'), y=Decimal('0.422570237313190266714713061446673236787319183349609375'), r=Decimal('0.002975878')), Circle(x=Decimal('0.998452264319443560935951609280891716480255126953125'), y=Decimal('-0.027505928248592777241032791835095849819481372833251953125'), r=Decimal('0.002417615')), Circle(x=Decimal('0.64080589862540537904322945905732922255992889404296875'), y=Decimal('0.76649279704940409541080725830397568643093109130859375'), r=Decimal('0.001964079')), Circle(x=Decimal('0.265497143826182890880005516009987331926822662353515625'), y=Decimal('0.96307426927522088444533210349618457257747650146484375'), r=Decimal('0.001595626')), Circle(x=Decimal('0.7126111128538477057503541800542734563350677490234375'), y=Decimal('-0.70037356894279512165013557023485191166400909423828125'), r=Decimal('0.001296292')), Circle(x=Decimal('-0.61211447780576211386005525127984583377838134765625'), y=Decimal('-0.78950764636347214864287025193334557116031646728515625'), r=Decimal('0.001053113'))]

    init_params = params_from_created_circles(init_circles[:k])

    # init_params.extend([0, 0])

    # print(add_centroid_circles(p, pk, create_circles_from_params(init_params, k, p, pk)))
    print(len(init_params))
    print(init_params)


    # Default optimization parameters
    kwargs = {
        'bounds': bounds,
        'workers': -1,
        'updating': 'deferred',  # Required when using parallel workers
        'seed': 0,  # For reproducibility
        # 'atol': float(EPSILON() ** 2),
        'mutation': (0.5, 1.0),  # Allow mutation rate to adapt
        'x0': init_params,
    }
    
    # Update with any custom parameters
    if optimization_kwargs:
        kwargs.update(optimization_kwargs)
    
    if callback:
        kwargs['callback'] = callback
    
    result = optimize.differential_evolution(obj_func, **kwargs)
    return result

def place_algorithm_11(p: Decimal, pk: Callable[[Decimal, Decimal], Decimal] = default_pk,
                      initial_circles: int = 6,
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
    start_time = time.time()
    best_score = float('inf')
    iterations = 0

    def progress_callback(xk, _):
        nonlocal best_score, iterations
        iterations += 1
        score = float(objective_function_global(xk, p, initial_circles, pk))
        if score < best_score:
            best_score = score
            elapsed = time.time() - start_time
            print(f"Iteration {iterations}: New best score = {best_score:.6f} (elapsed: {elapsed:.2f}s)")
        return False

    # Run the optimization
    result = optimize_circle_placement(p, initial_circles, pk, progress_callback, optimization_kwargs)

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
    # main()
    getcontext().prec = 7
    # p = 2 ** (-1 / Decimal('2.5'))
    # p = Decimal('0.76')
    p = Decimal('0.66')
    initial_circles = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(initial_circles)
    circles = place_algorithm_11(p, initial_circles=initial_circles, pk=find_all_pk)
    # circles = [Circle(x=Decimal('-0.4660305'), y=Decimal('0'), r=Decimal('0.7578583')), Circle(x=Decimal('0.6219811'), y=Decimal('0.2293612'), r=Decimal('0.5743492')), Circle(x=Decimal('0.3799721'), y=Decimal('-0.6418602'), r=Decimal('0.4352753')), Circle(x=Decimal('0.09652912'), y=Decimal('0.8269215'), r=Decimal('0.3298770')), Circle(x=Decimal('-0.1954859'), y=Decimal('-0.8560651'), r=Decimal('0.2500000')), Circle(x=Decimal('-0.3625750'), y=Decimal('0.8458253'), r=Decimal('0.1894646')), Circle(x=Decimal('0.8360550'), y=Decimal('-0.3959104'), r=Decimal('0.1435873')), Circle(x=Decimal('-0.5261192'), y=Decimal('-0.8089158'), r=Decimal('0.1088188')), Circle(x=Decimal('0.480869862587108898299703696466167457401752471923828125'), y=Decimal('0.83068605528566052953465259633958339691162109375'), r=Decimal('0.08246926')), Circle(x=Decimal('-0.59274082610672007565000285467249341309070587158203125'), y=Decimal('0.77314890488008958246979318573721684515476226806640625'), r=Decimal('0.06250001')), Circle(x=Decimal('0.05955299818111030318856791154757956974208354949951171875'), y=Decimal('-0.97322066399223505772653197709587402641773223876953125'), r=Decimal('0.04736615')), Circle(x=Decimal('0.945071667223251754563762006000615656375885009765625'), y=Decimal('-0.267485111240606254767726568388752639293670654296875'), r=Decimal('0.03589683')), Circle(x=Decimal('-0.64508836223532295406357661704532802104949951171875'), y=Decimal('-0.7483326183722491808936183588230051100254058837890625'), r=Decimal('0.02720471')), Circle(x=Decimal('0.8187915289415081954160768873407505452632904052734375'), y=Decimal('-0.55365355153100981322467077916371636092662811279296875'), r=Decimal('0.02061732')), Circle(x=Decimal('-0.66262275981678431246990612635272555053234100341796875'), y=Decimal('0.7393680566860967307007967974641360342502593994140625'), r=Decimal('0.01562501')), Circle(x=Decimal('0.57264705977377372558834167648456059396266937255859375'), y=Decimal('0.80989263449354031987326152375317178666591644287109375'), r=Decimal('0.01184154')), Circle(x=Decimal('-0.20882603562105661598735650841263122856616973876953125'), y=Decimal('0.97027272385723339898078165788319893181324005126953125'), r=Decimal('0.008974209')), Circle(x=Decimal('0.424209419172794033325857299132621847093105316162109375'), y=Decimal('0.89909879687231797351643081128713674843311309814453125'), r=Decimal('0.006801179')), Circle(x=Decimal('0.110286511239986950716485125667531974613666534423828125'), y=Decimal('-0.98936943284950518151532605770626105368137359619140625'), r=Decimal('0.005154330')), Circle(x=Decimal('-0.675947897554812104914390147314406931400299072265625'), y=Decimal('-0.732007983290130948006435573915950953960418701171875'), r=Decimal('0.003906252')), Circle(x=Decimal('0.9684947036287188115721846770611591637134552001953125'), y=Decimal('-0.233445824486877284709152036157320253551006317138671875'), r=Decimal('0.002960385')), Circle(x=Decimal('-0.446946708195753383829895710732671432197093963623046875'), y=Decimal('-0.8895709389850099402252681102254427969455718994140625'), r=Decimal('0.002243553')), Circle(x=Decimal('-0.55448923632514379722380226667155511677265167236328125'), y=Decimal('0.82695865409689961467165630892850458621978759765625'), r=Decimal('0.001700295')), Circle(x=Decimal('-0.446946708195759379034228686577989719808101654052734375'), y=Decimal('-0.88957093898502126450011928682215511798858642578125'), r=Decimal('0.001288583'))]
    # circles = [Circle(x=Decimal('-0.4621705'), y=Decimal('0'), r=Decimal('0.7578583')), Circle(x=Decimal('0.6256425'), y=Decimal('0.2326335'), r=Decimal('0.5743492')), Circle(x=Decimal('0.3799522'), y=Decimal('-0.6394088'), r=Decimal('0.4352753')), Circle(x=Decimal('0.09435549'), y=Decimal('0.8280994'), r=Decimal('0.3298770')), Circle(x=Decimal('-0.1911341'), y=Decimal('-0.8601548'), r=Decimal('0.2500000')), Circle(x=Decimal('-0.3621675'), y=Decimal('0.8498134'), r=Decimal('0.1894646')), Circle(x=Decimal('0.8355603'), y=Decimal('-0.3944838'), r=Decimal('0.1435873')), Circle(x=Decimal('-0.52249459542098286579658861228381283581256866455078125'), y=Decimal('-0.79951779020669666575571454814053140580654144287109375'), r=Decimal('0.1088188')), Circle(x=Decimal('0.478108268017147286510493131572729907929897308349609375'), y=Decimal('0.83226584615214205253863610778353177011013031005859375'), r=Decimal('0.08246926')), Circle(x=Decimal('-0.5923398972211406654508891733712516725063323974609375'), y=Decimal('0.77278031110463363262397251673974096775054931640625'), r=Decimal('0.06250001')), Circle(x=Decimal('0.06440370617754369308993744880353915505111217498779296875'), y=Decimal('-0.9739136913108425996910000321804545819759368896484375'), r=Decimal('0.04736615')), Circle(x=Decimal('0.945172075466456984571550492546521127223968505859375'), y=Decimal('-0.266710364048800219194390592747367918491363525390625'), r=Decimal('0.03589683')), Circle(x=Decimal('-0.64607838948470541762247876249602995812892913818359375'), y=Decimal('-0.7476118730071712459306354503496550023555755615234375'), r=Decimal('0.02720471')), Circle(x=Decimal('0.81929930229130787378011291366419754922389984130859375'), y=Decimal('-0.5526431474161856982618701294995844364166259765625'), r=Decimal('0.02061732')), Circle(x=Decimal('-0.66298605878645922029335224578971974551677703857421875'), y=Decimal('0.73858704278622078209792789493803866207599639892578125'), r=Decimal('0.01562501')), Circle(x=Decimal('0.569612049146654531028843848616816103458404541015625'), y=Decimal('0.81232234708392148103683894078130833804607391357421875'), r=Decimal('0.01184154')), Circle(x=Decimal('-0.446152987284383739652326994473696686327457427978515625'), y=Decimal('-0.8865485790819789269079365112702362239360809326171875'), r=Decimal('0.008974209')), Circle(x=Decimal('-0.2089098823704004515011689591119647957384586334228515625'), y=Decimal('0.971964861975724403464482747949659824371337890625'), r=Decimal('0.006801179')), Circle(x=Decimal('0.421736682759198122649735296363360248506069183349609375'), y=Decimal('0.90055484730787005442920190034783445298671722412109375'), r=Decimal('0.005154330')), Circle(x=Decimal('-0.67710409240092073179795306714368052780628204345703125'), y=Decimal('-0.7307063094435901628997953594080172479152679443359375'), r=Decimal('0.003906252')), Circle(x=Decimal('0.11458748311946047315768026919613475911319255828857421875'), y=Decimal('-0.9894749187324034522816873504780232906341552734375'), r=Decimal('0.002960385')), Circle(x=Decimal('0.96865022960213431613141210618778131902217864990234375'), y=Decimal('-0.2327812901864864947309996523472364060580730438232421875'), r=Decimal('0.002243553')), Circle(x=Decimal('-0.55390670905321937045329150350880809128284454345703125'), y=Decimal('0.8267531015774716163235780186369083821773529052734375'), r=Decimal('0.001700295')), Circle(x=Decimal('0.42173668275920039860693577793426811695098876953125'), y=Decimal('0.90055484730787405123209055091137997806072235107421875'), r=Decimal('0.001288583'))]
    
    # works for p = 0.67, find all
    # circles = [Circle(x=Decimal('-0.4908906'), y=Decimal('0'), r=Decimal('0.67')), Circle(x=Decimal('0.5804569'), y=Decimal('0.2405622'), r=Decimal('0.5484186')), Circle(x=Decimal('0.208870012530215920509846228014794178307056427001953125'), y=Decimal('-0.624393270985229786873560442472808063030242919921875'), r=Decimal('0.4489')), Circle(x=Decimal('-0.0687488043892642564092199108927161432802677154541015625'), y=Decimal('0.77310537179094873661000519859953783452510833740234375'), r=Decimal('0.3674405')), Circle(x=Decimal('-0.393608215151188989278097096757846884429454803466796875'), y=Decimal('-0.791659521973363755620312076644040644168853759765625'), r=Decimal('0.300763')), Circle(x=Decimal('0.7684885770609046762302796196308918297290802001953125'), y=Decimal('-0.42347591574216758569804142098291777074337005615234375'), r=Decimal('0.2461851')), Circle(x=Decimal('-0.5598258887884235424081680321251042187213897705078125'), y=Decimal('0.74140036004681342252098374956403858959674835205078125'), r=Decimal('0.2015112')), Circle(x=Decimal('0.3929231971960387426179295289330184459686279296875'), y=Decimal('0.8388412309319217552427971895667724311351776123046875'), r=Decimal('0.1649440')), Circle(x=Decimal('-0.719611742351108585324936939286999404430389404296875'), y=Decimal('-0.6575628068270826798169537141802720725536346435546875'), r=Decimal('0.1350125')), Circle(x=Decimal('0.9519120239052198950702177171478979289531707763671875'), y=Decimal('-0.2013058037931811650178559602863970212638378143310546875'), r=Decimal('0.1105125')), Circle(x=Decimal('0.6797276989002438707387909744284115731716156005859375'), y=Decimal('-0.69765922557720283503357450172188691794872283935546875'), r=Decimal('0.09045838')), Circle(x=Decimal('-0.10946213991471363813356987293445854447782039642333984375'), y=Decimal('-0.9722348976327073177827742256340570747852325439453125'), r=Decimal('0.07404338')), Circle(x=Decimal('-0.7605164806089919071752092349925078451633453369140625'), y=Decimal('0.62843717938068355266523212776519358158111572265625'), r=Decimal('0.06060712')), Circle(x=Decimal('0.1816497487910462005356038162062759511172771453857421875'), y=Decimal('-0.158394717928385109217970239114947617053985595703125'), r=Decimal('0.04960906')), Circle(x=Decimal('0.57507623236683491629861464389250613749027252197265625'), y=Decimal('0.80242357988825607773009096490568481385707855224609375'), r=Decimal('0.04060677')), Circle(x=Decimal('0.05021068790505144041613760919062769971787929534912109375'), y=Decimal('0.413950344752257615166257664895965717732906341552734375'), r=Decimal('0.03323807')), Circle(x=Decimal('0.26069828357698343612725011553266085684299468994140625'), y=Decimal('0.956030152369351515773132632602937519550323486328125'), r=Decimal('0.02720653')), Circle(x=Decimal('-0.421038711816131117071648759520030580461025238037109375'), y=Decimal('0.8995030400165087147712483783834613859653472900390625'), r=Decimal('0.02226951')), Circle(x=Decimal('0.539648777974118143418991166981868445873260498046875'), y=Decimal('-0.31223545371566785444628067125449888408184051513671875'), r=Decimal('0.01822838')), Circle(x=Decimal('0.614181769376461073051132188993506133556365966796875'), y=Decimal('0.78848548708470478363352640371886081993579864501953125'), r=Decimal('0.01492057')), Circle(x=Decimal('-0.03952937922512121671214657681048265658318996429443359375'), y=Decimal('-0.9988086069087371843266964788199402391910552978515625'), r=Decimal('0.01221301'))]

    # not quite 0.66, find all
    # circles = [Circle(x=Decimal('-0.4830411'), y=Decimal('0'), r=Decimal('0.66')), Circle(x=Decimal('0.5318464'), y=Decimal('0.2416313'), r=Decimal('0.5361865')), Circle(x=Decimal('0.2861341'), y=Decimal('-0.6065730'), r=Decimal('0.4356')), Circle(x=Decimal('-0.03615046'), y=Decimal('0.7759997'), r=Decimal('0.3538831')), Circle(x=Decimal('-0.3231259'), y=Decimal('-0.7974621'), r=Decimal('0.287496')), Circle(x=Decimal('0.8054224'), y=Decimal('-0.3762863'), r=Decimal('0.2335629')), Circle(x=Decimal('-0.5249223'), y=Decimal('0.7555362'), r=Decimal('0.1897474')), Circle(x=Decimal('0.4046515'), y=Decimal('0.8577455'), r=Decimal('0.1541515')), Circle(x=Decimal('-0.6913100'), y=Decimal('-0.6914375'), r=Decimal('0.1252333')), Circle(x=Decimal('-0.01136641'), y=Decimal('-0.9794021'), r=Decimal('0.1017400')), Circle(x=Decimal('0.9682765'), y=Decimal('-0.1338024'), r=Decimal('0.08265395')), Circle(x=Decimal('-0.7370363'), y=Decimal('0.6390875'), r=Decimal('0.06714839')), Circle(x=Decimal('0.7482672'), y=Decimal('-0.6419571'), r=Decimal('0.05455161')), Circle(x=Decimal('0.5791013'), y=Decimal('0.7920574'), r=Decimal('0.04431794')), Circle(x=Decimal('-0.3782377'), y=Decimal('0.9100824'), r=Decimal('0.03600406')), Circle(x=Decimal('0.1705255'), y=Decimal('-0.1736458'), r=Decimal('0.02924984')), Circle(x=Decimal('-0.7942498'), y=Decimal('-0.5902347'), r=Decimal('0.02376268')), Circle(x=Decimal('-0.8009848'), y=Decimal('0.5868187'), r=Decimal('0.01930489')), Circle(x=Decimal('0.9929994'), y=Decimal('-0.04188113'), r=Decimal('0.01568337')), Circle(x=Decimal('0.2742226'), y=Decimal('0.9559963'), r=Decimal('0.01274123')), Circle(x=Decimal('0.62783818683958070305806131727877072989940643310546875'), y=Decimal('0.77326822321228105838741839761496521532535552978515625'), r=Decimal('0.01035102')), Circle(x=Decimal('0.71628027964776486147258083292399533092975616455078125'), y=Decimal('-0.69301439361814309858544902454013936221599578857421875'), r=Decimal('0.008409211')), Circle(x=Decimal('-0.81731057784901739449168189821648411452770233154296875'), y=Decimal('-0.57198732142082653329140384812490083277225494384765625'), r=Decimal('0.006831675')), Circle(x=Decimal('-0.81871758378737402583880111706093885004520416259765625'), y=Decimal('0.5706265440066895511250777417444624006748199462890625'), r=Decimal('0.005550079')), Circle(x=Decimal('0.025271821720933902721828445692153763957321643829345703125'), y=Decimal('0.424918884588280476588550982341985218226909637451171875'), r=Decimal('0.004508906')), Circle(x=Decimal('-0.70887689170958678719358658781857229769229888916015625'), y=Decimal('0.70214588860013826820960503027890808880329132080078125'), r=Decimal('0.003663052')), Circle(x=Decimal('0.99854446726559464853067993317381478846073150634765625'), y=Decimal('-0.024729054068863824678015106428574654273688793182373046875'), r=Decimal('0.002975878')), Circle(x=Decimal('0.6389450658617759071233876966289244592189788818359375'), y=Decimal('0.76799714141834407588049771220539696514606475830078125'), r=Decimal('0.002417615')), Circle(x=Decimal('-0.6115306319420650282836504629813134670257568359375'), y=Decimal('-0.7895569141375948785110949756926856935024261474609375'), r=Decimal('0.001964079')), Circle(x=Decimal('-0.823424844433121361220173639594577252864837646484375'), y=Decimal('-0.56623809834514970962260349551797844469547271728515625'), r=Decimal('0.001595626')), Circle(x=Decimal('-0.8237548305303843410030140148592181503772735595703125'), y=Decimal('0.5659288806692226447836446823203004896640777587890625'), r=Decimal('0.001296292')), Circle(x=Decimal('0.71190112698686214276477812745724804699420928955078125'), y=Decimal('-0.701426679930230445592087562545202672481536865234375'), r=Decimal('0.001053113'))]

    # works for p = 0.76, find one
    # circles = [Circle(x=Decimal('-0.4835255'), y=Decimal('0'), r=Decimal('0.66')), Circle(x=Decimal('0.5318361'), y=Decimal('0.2461311'), r=Decimal('0.5361865')), Circle(x=Decimal('0.2903324'), y=Decimal('-0.6025555'), r=Decimal('0.4356')), Circle(x=Decimal('-0.03487531'), y=Decimal('0.7734550'), r=Decimal('0.3538831')), Circle(x=Decimal('-0.3268369'), y=Decimal('-0.7899857'), r=Decimal('0.287496')), Circle(x=Decimal('0.8097788'), y=Decimal('-0.3717931'), r=Decimal('0.2335629')), Circle(x=Decimal('-0.5255267'), y=Decimal('0.7548169'), r=Decimal('0.1897474')), Circle(x=Decimal('0.4004226'), y=Decimal('0.8674718'), r=Decimal('0.1541515')), Circle(x=Decimal('-0.67945382451224645148357694779406301677227020263671875'), y=Decimal('-0.6751750376275313936247357560205273330211639404296875'), r=Decimal('0.1252333')), Circle(x=Decimal('-0.024939131909031464484627349520451389253139495849609375'), y=Decimal('-0.95977248830864458906120262327021919190883636474609375'), r=Decimal('0.1017400')), Circle(x=Decimal('0.96042654915217517963554882953758351504802703857421875'), y=Decimal('-0.12630901746618850012282564421184360980987548828125'), r=Decimal('0.08265395')), Circle(x=Decimal('-0.7379269962124757054056090055382810533046722412109375'), y=Decimal('0.63629507516187100435445245238952338695526123046875'), r=Decimal('0.06714839')), Circle(x=Decimal('0.74725443286440584866880953995860181748867034912109375'), y=Decimal('-0.63242648475445495392932571121491491794586181640625'), r=Decimal('0.05455161')), Circle(x=Decimal('0.57292595114667521638551761498092673718929290771484375'), y=Decimal('0.79745691569950449828496630289009772241115570068359375'), r=Decimal('0.04431794')), Circle(x=Decimal('-0.3800292752701974752227442877483554184436798095703125'), y=Decimal('0.90688579706415783920903095349785871803760528564453125'), r=Decimal('0.03600406')), Circle(x=Decimal('0.1701712341055969524017399407966877333819866180419921875'), y=Decimal('-0.16927768809544196937366677957470528781414031982421875'), r=Decimal('0.02924984')), Circle(x=Decimal('-0.7964434090271288457785203718231059610843658447265625'), y=Decimal('-0.5907472441865468937294281204231083393096923828125'), r=Decimal('0.02376268')), Circle(x=Decimal('-0.80251702513649281200969198835082352161407470703125'), y=Decimal('0.5854148964206424832212860565050505101680755615234375'), r=Decimal('0.01930489')), Circle(x=Decimal('0.9934478957365551199387709857546724379062652587890625'), y=Decimal('-0.03814209083888808748952214955352246761322021484375'), r=Decimal('0.01568337')), Circle(x=Decimal('0.0817895469029160937513012186173000372946262359619140625'), y=Decimal('-0.99129309792664332956491080039995722472667694091796875'), r=Decimal('0.01274123')), Circle(x=Decimal('0.6212917866577833958530163727118633687496185302734375'), y=Decimal('0.77872669842496000658371713143424130976200103759765625'), r=Decimal('0.01035102')), Circle(x=Decimal('0.7212511615752885685282080885372124612331390380859375'), y=Decimal('-0.68751476264394018045322809484787285327911376953125'), r=Decimal('0.008409211')), Circle(x=Decimal('0.27049894230017679230826388447894714772701263427734375'), y=Decimal('0.9590038643927643047248920993297360837459564208984375'), r=Decimal('0.006831675')), Circle(x=Decimal('-0.8179052782265145982165677196462638676166534423828125'), y=Decimal('-0.57155388699260523122092081393930129706859588623046875'), r=Decimal('0.005550079')), Circle(x=Decimal('-0.70975574740430091846832283408730290830135345458984375'), y=Decimal('0.70007895650846385660059922884101979434490203857421875'), r=Decimal('0.004508906')), Circle(x=Decimal('-0.81969319379565097616335833663470111787319183349609375'), y=Decimal('0.569900458619543304195076416363008320331573486328125'), r=Decimal('0.003663052')), Circle(x=Decimal('-0.352464891500763155551823047062498517334461212158203125'), y=Decimal('0.93384660870944535826509991238708607852458953857421875'), r=Decimal('0.002975878')), Circle(x=Decimal('-0.61599044900595334439685757388360798358917236328125'), y=Decimal('-0.78541558261101107607515814379439689218997955322265625'), r=Decimal('0.002417615')), Circle(x=Decimal('0.99848828905546505030343951148097403347492218017578125'), y=Decimal('-0.02056921686473318910959307004304719157516956329345703125'), r=Decimal('0.001964079')), Circle(x=Decimal('0.09651744527318030508755697383094229735434055328369140625'), y=Decimal('-0.99406620009273638505220560546149499714374542236328125'), r=Decimal('0.001595626')), Circle(x=Decimal('0.024447861688632895049710924695318681187927722930908203125'), y=Decimal('0.42330639558194727012363500762148760259151458740234375'), r=Decimal('0.001296292')), Circle(x=Decimal('0.6322286034724895475989114856929518282413482666015625'), y=Decimal('0.77369357982573283560867594133014790713787078857421875'), r=Decimal('0.001053113'))]
    # # # # for square in get_all_uncovered_squares(circles):
    # # # #     print(square)
    # # # squares = list(get_all_uncovered_squares(circles))
    square = get_biggest_uncovered_square(circles)
    print(circles, len(circles), get_empty_area(circles), covers_unit_circle(circles))
    print(square)

    # draw_circles(circles, squares=[], title='Algorithm 11')
