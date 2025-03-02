import sys
from typing import Callable, Any, Optional
import argparse
import time
import math
from dataclasses import dataclass

import shapely
from utils import PRECISION, Circle, covers_unit_circle, draw_circles, binary_search, get_biggest_semicovered_square, get_biggest_uncovered_square, get_distance_traveled

from scipy import optimize
from scipy.optimize._optimize import OptimizeResult
import itertools

PkFunction = Callable[[float, int], float]
CirclePlacerFunction = Callable[[float, PkFunction], list[Circle]]

def dummy_placement_algorithm(_p: float, _pk: PkFunction) -> list[Circle]:
    return []

def default_pk(p: float, k: int) -> float:
    """Default pk function that returns p^k"""
    return p ** k

def find_all_pk(p: float, k: int) -> float:
    """pk function that returns p^((k + 1) / 2)"""
    return p ** ((k + 1) / 2)

def get_configuration(args: argparse.Namespace) -> tuple[PkFunction, int]:
    """Get pk function and c_multiplier based on arguments."""
    if args.find_all:
        return find_all_pk, 2
    return default_pk, 1

def place_algorithm_4(p: float, pk: PkFunction = default_pk) -> list[Circle]:
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

def place_algorithm_5(p: float, pk: PkFunction = default_pk) -> list[Circle]:
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

def place_algorithm_5_5(p: float, pk: PkFunction = default_pk, final_optimization: bool = True) -> list[Circle]:
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

def place_algorithm_6(p: float, pk: PkFunction = default_pk, final_optimization: bool = True) -> list[Circle]:
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
            return circles # failure

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

def place_algorithm_10(p: float, pk: PkFunction = default_pk, circle_placement_algorithm: CirclePlacerFunction = dummy_placement_algorithm) -> list[Circle]:
    circles = circle_placement_algorithm(p, pk)

    return add_intelligent_circles(p, pk, circles)[0]

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

def add_centroid_circles(p: float, pk: PkFunction, circles: list[Circle]) -> tuple[list[Circle], float]:
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
                biggest_uncovered_square = get_biggest_semicovered_square(circles)
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

def get_circle_centers(x1: float, y1: float, x2: float, y2: float, r: float):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    if dist > 2 * r:
        return
    if dist == 0 and r != 0:
        return
    if dist == 0 and r == 0:
        return

    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    if dist == 2 * r:
        yield (mid_x, mid_y)
        return

    a = (r**2) - (dist/2)**2
    if a < 0:
      # Numerical instability could cause slight negative due to floating point errors.
      # Should happen when dist is very close to 2 * r. Just treat it as zero.
      a = 0

    a = math.sqrt(a)

    # Calculate the coordinates of the intersection points.
    dx = x2 - x1
    dy = y2 - y1

    x3 = mid_x + (a * dy / dist)
    y3 = mid_y - (a * dx / dist)
    x4 = mid_x - (a * dy / dist)
    y4 = mid_y + (a * dx / dist)

    yield (x3, y3)
    yield (x4, y4)

def intelligently_minimize(largest_geom: shapely.geometry.base.BaseGeometry, r: float) -> Circle:
    convex_hull = largest_geom.convex_hull

    # Get vertices of convex hull
    all_vertices: list[tuple[float, float]] = list(convex_hull.exterior.coords)[:-1] # type: ignore

    vertices: list[tuple[float, float]] = []
    for i in range(len(all_vertices)):
        x1, y1 = all_vertices[i]
        x2, y2 = all_vertices[(i + 1) % len(all_vertices)]
        x3, y3 = all_vertices[(i + 2) % len(all_vertices)]

        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x3 - x2
        dy2 = y3 - y2

        dot = dx1 * dx2 + dy1 * dy2
        det = dx1 * dy2 - dy1 * dx2
        angle = math.atan2(det, dot)

        if max(abs(angle), math.pi - abs(angle)) <= math.pi * 0.9:
            vertices.append((x2, y2))

    if len(vertices) > 10:
        return Circle(largest_geom.centroid.x, largest_geom.centroid.y, r)

    max_area = 0
    max_circle = None
    for (x1, y1), (x2, y2) in itertools.combinations(vertices, 2):
        for (x0, y0) in get_circle_centers(x1, y1, x2, y2, r):
            circle = Circle(x0, y0, r)
            circle_polygon = PRECISION.get_circle_polygon(circle)

            # Calculate area of the intersection
            intersection = largest_geom.intersection(circle_polygon)
            if intersection.area > max_area:
                max_area = intersection.area
                max_circle = circle
        
    return max_circle if max_circle else Circle(largest_geom.centroid.x, largest_geom.centroid.y, r)

def add_intelligent_circles(p: float, pk: PkFunction, circles: list[Circle]) -> tuple[list[Circle], float]:
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
                biggest_uncovered_square = get_biggest_semicovered_square(circles)
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
        
        new_circle: Circle = intelligently_minimize(largest_geom, current_radius) # type: ignore
        circles.append(new_circle)
        k += 1

        uncovered_polygons = uncovered_polygons.difference(PRECISION.get_circle_polygon(new_circle)) # type: ignore

    return circles, uncovered_polygons.area

# Algorithm 11 helper functions
def create_circles_from_params(params: list[float], k: int, p: float, pk: PkFunction) -> list[Circle]:
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

def objective_function_global(x: list[float], p: float, k: int, pk: PkFunction) -> float:
    """Calculate the objective function value (remaining area) for given parameters"""
    circles = create_circles_from_params(x, k, p, pk)
    # _, remaining_area = add_centroid_circles(p, pk, circles)
    _, remaining_area = add_intelligent_circles(p, pk, circles)
    return remaining_area

@dataclass
class ObjectiveFunctionWrapper:
    """A pickleable wrapper class for the objective function."""
    p: float
    k: int
    pk: PkFunction
    
    def __call__(self, x: list[float]) -> float:
        return objective_function_global(x, self.p, self.k, self.pk)

def optimize_circle_placement(p: float, k: int, pk: PkFunction,
                          callback: Optional[Callable[[list[float], float], bool]] = None,
                          optimization_kwargs: Optional[dict[str, Any]] = None,
                          x0: Optional[list[float]] = None) -> OptimizeResult:
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
        'workers': -1,  # Use all available cores
        'updating': 'deferred',  # Use deferred updating,
        'popsize': 15,  # Population size
        'maxiter': 2000,
        'atol': PRECISION.epsilon / 10,
    }
    
    # Update with any custom parameters
    if optimization_kwargs:
        kwargs.update(optimization_kwargs)
    
    if callback:
        kwargs['callback'] = callback

    if x0 is not None:
        kwargs['x0'] = x0
    
    return optimize.differential_evolution(obj_func, **kwargs) # type: ignore

def place_algorithm_11(p: float, pk: PkFunction = default_pk,
                      initial_circles: int = 6,
                      optimization_kwargs: Optional[dict[str, Any]] = None,
                      initial_guess: Optional[list[Circle]] = None) -> list[Circle]:
    """Algorithm that uses differential evolution to optimize circle placement."""
    start_time = time.time()
    best_score = float('inf')
    iterations = 0

    print(p)

    def progress_callback(xk: list[float], _: float) -> bool:
        nonlocal best_score, iterations
        iterations += 1
        score = objective_function_global(xk, p, initial_circles, pk)
        if score < best_score:
            best_score = score
            elapsed = time.time() - start_time
            print(f"Iteration {iterations}: New best score = {best_score:.6g} (elapsed: {elapsed:.2f}s)")
        return False

    # Convert initial guess to optimization parameters if provided
    x0 = None
    if initial_guess is not None:
        x0 = params_from_created_circles(initial_guess[:initial_circles])

    # Run the optimization
    result: OptimizeResult = optimize_circle_placement(p, initial_circles, pk, progress_callback, optimization_kwargs, x0)

    if not result.success:
        print(f"Warning: Optimization may not have converged: {result.message}")
    
    total_time = time.time() - start_time
    print(f"Optimization complete in {result.nit} iterations ({total_time:.2f}s)")
    
    # Create the final circles using the optimized parameters
    circles = create_circles_from_params(result.x, initial_circles, p, pk)
    
    # Add centroid circles
    # circles, area = add_centroid_circles(p, pk, circles)
    circles, area = add_intelligent_circles(p, pk, circles)

    print(f'Final Remaining Area: {area:.6f}')
    
    return [Circle(float(circle.x), float(circle.y), float(circle.r)) for circle in circles]

def get_placement_algorithm(algorithm: float) -> CirclePlacerFunction:
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
        circle_placement_algorithm: CirclePlacerFunction = lambda p, pk: place_algorithm_5(p, pk)[:-1]
        return lambda p, pk: place_algorithm_10(p, pk, circle_placement_algorithm)
    elif algorithm == 11:
        initial_guess_find_one = [Circle(x=-0.4652323341815545, y=0.0, r=0.7606738281250001), Circle(x=0.631835807925796, y=0.1903730016493503, r=0.5786246727943422), Circle(x=0.3697940080061505, y=-0.6721376127407843, r=0.4401446449020478), Circle(x=0.10868020756478226, y=0.809927396170051, r=0.3348065119663595), Circle(x=-0.2268096324157326, y=-0.8507008005358745, r=0.2546785511386293), Circle(x=-0.35105515226951206, y=0.8525717091300288, r=0.19372730843594976), Circle(x=0.8334512813738311, y=-0.45141887403603526, r=0.14736329332032652), Circle(x=0.5066812387531501, y=0.8205098167780042, r=0.11209540045508003), Circle(x=-0.5338611091463583, y=-0.8110230105587689, r=0.0852680373793706), Circle(x=-0.5799133384414544, y=0.7846827534845661, r=0.06486116441007143), Circle(x=0.024957526866519295, y=-0.980222221404978, r=0.049338190228454044), Circle(x=-0.6306083956676077, y=-0.7532837868134962, r=0.03753027003383761), Circle(x=0.9308286536798448, y=-0.3196167621135757, r=0.028548294177204232), Circle(x=-0.6507673738092681, y=0.7465983474462492, r=0.02171594021821259), Circle(x=0.621415822507479, y=0.774187508417866, r=0.01651874737712142), Circle(x=-0.1931950036775884, y=0.9728756836537937, r=0.012565378803184755), Circle(x=-0.6723563363140392, y=-0.7368493091564825, r=0.00955815479605928), Circle(x=-0.6750275665055643, y=0.7343573392884171, r=0.0072706381985297415), Circle(x=0.9514513927696395, y=-0.2968022336890622, r=0.0055305841913874726), Circle(x=-0.18133794081650303, y=0.9813524299000489, r=0.004206970648630317), Circle(x=0.9554387725018909, y=-0.2904077680461287, r=0.0032001324681031375), Circle(x=-0.6814658353052274, y=-0.7300857539758098, r=0.0024342570150191183), Circle(x=-0.682339445375422, y=0.7298309511390368, r=0.0018516756022547286), Circle(x=0.6379088284879326, y=0.7690965724631914, r=0.0014085211688127694), Circle(x=0.07213390677064563, y=-0.9969650230993562, r=0.0010714251894759087), Circle(x=-0.6842523100838022, y=-0.7289071056215444, r=0.000815005100428193), Circle(x=-0.6843349897161745, y=0.7287749660796549, r=0.0006199530496841137), Circle(x=-0.17762853730071212, y=0.9838848899039228, r=0.0004715820595609831), Circle(x=-0.6851032171052384, y=0.7282938748064527, r=0.00035872013052132485), Circle(x=-0.6850662861629557, y=-0.7282949943196, r=0.00027286901490915585), Circle(x=-0.06988362735558888, y=-0.6499857099214075, r=0.00020756431814764528), Circle(x=0.9576159033083701, y=-0.287901726864696, r=0.00015788874446752475), Circle(x=-0.6854133856566726, y=-0.7281474087016567, r=0.00012010183567196199)]
        initial_guess_find_all = [Circle(x=-0.471474198161091, y=0.0, r=0.6610867748260498), Circle(x=0.5344356300402844, y=0.2516804101514229, r=0.5375114297981989), Circle(x=0.2931742180401616, y=-0.601069880168715, r=0.4370357238499083), Circle(x=-0.04714303745451637, y=0.7758512566130683, r=0.35534169755742995), Circle(x=-0.3142139475405279, y=-0.8046332150144365, r=0.288918537163704), Circle(x=0.8118390255823599, y=-0.36471140452782463, r=0.234911696799455), Circle(x=-0.5387703656339764, y=0.7587953426460567, r=0.19100022392101332), Circle(x=0.3969877576912187, y=0.8670636387453872, r=0.1552970160060666), Circle(x=-0.6766721557954213, y=-0.6974955984567275, r=0.12626772202299602), Circle(x=-0.0007235223709840529, y=-0.9626115150116027, r=0.10266480345156001), Circle(x=0.9633341312578282, y=-0.11891457640707916, r=0.08347392111681462), Circle(x=-0.7444685494440092, y=0.6225383563303094, r=0.06787034380194211), Circle(x=0.7550232014179635, y=-0.6321053383253897, r=0.05518350529319908), Circle(x=0.5758672010215882, y=0.8014051020093613, r=0.04486818669036109), Circle(x=-0.7899071318400825, y=-0.5926293697629225, r=0.036481085537877225), Circle(x=-0.39170629655318034, y=0.9053792971775639, r=0.02966176483142391), Circle(x=0.1805013539778567, y=-0.16761877785458698, r=0.024117163180388503), Circle(x=-0.8112484919241976, y=0.5743617397272538, r=0.019609000448054782), Circle(x=0.2632988745338545, y=0.9590790103386507, r=0.015943537624876596), Circle(x=0.9953073223054228, y=-0.033096345759572744, r=0.012963250863767102), Circle(x=-0.8224547412002893, y=-0.5636849690645723, r=0.010540061867747447), Circle(x=0.7250581514342697, y=-0.6844248514518876, r=0.008569833704788797), Circle(x=-0.7196911569465999, y=0.6888864505391404, r=0.006967895506616191), Circle(x=-0.8278587049546778, y=0.558450093823494, r=0.005665403724694405), Circle(x=0.6194732654145584, y=0.7835784766107168, r=0.004606383567793822), Circle(x=0.0991572723911561, y=-0.9939795089945883, r=0.0037453234764457137), Circle(x=-0.3697771729955276, y=0.927787999834231, r=0.0030452192564445306), Circle(x=0.9992132615438009, y=-0.01950720668937055, r=0.0024759838177237856), Circle(x=-0.8309274328131553, y=-0.5554443592754847, r=0.002013154176881096), Circle(x=0.3070732302519927, y=0.739540504000506, r=0.0016368401565805075), Circle(x=0.5321008454033898, y=0.7897217135307991, r=0.0013308696020219149), Circle(x=0.7208652595980174, y=-0.6925910798291882, r=0.001082093380019574), Circle(x=0.2043602343467095, y=-0.172881259224821, r=0.0008798202929146962), Circle(x=0.7999448298492995, y=-0.599568618464722, r=0.0007153576226577594), Circle(x=-0.8321882995536307, y=0.5541992080067899, r=0.000581637559869487), Circle(x=-0.7146097042370902, y=0.6837115548508587, r=0.00047291346361004845), Circle(x=-0.8324932451267179, y=-0.5538971821394532, r=0.0003845128985718126), Circle(x=-0.36744323682740365, y=0.9299420037489814, r=0.00031263683642978343), Circle(x=0.1029697232468548, y=-0.994617923617098, r=0.00025419639199585563), Circle(x=-0.8326543609898214, y=0.5537375454529609, r=0.0002066800778871848), Circle(x=0.7203326234148276, y=-0.6935891006027869, r=0.0001680458729569585)]

        return lambda p, pk: place_algorithm_11(p, pk, initial_circles=6, initial_guess=initial_guess_find_one if pk == default_pk else initial_guess_find_all)
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
    place_algorithm: Callable[[float, PkFunction], list[Circle]],
    pk: PkFunction
) -> Callable[[float], tuple[bool, list[Circle]]]:
    """Create an evaluation function for binary search."""
    def evaluate(p: float) -> tuple[bool, list[Circle]]:
        circles = place_algorithm(p, pk)
        return covers_unit_circle(circles), circles
    return evaluate

def run_search(evaluator: Callable[[float], tuple[bool, list[Circle]]], debug: bool = False) -> tuple[float, list[Circle]]:
    """Run binary search with the given evaluator."""
    # return binary_search(0.5, 1, evaluator, debug=debug)
    return binary_search(0.66, 0.6621735496520996, evaluator, debug=debug)

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
    
    calc_precision = (precision + 2) * 2
    PRECISION.set_precision(calc_precision)
    
    pk, c_multiplier = get_configuration(argparse.Namespace(find_all=find_all))
    place_algorithm = get_placement_algorithm(algorithm)
    evaluator = create_evaluator(place_algorithm, pk)
    
    start_time = time.time()
    p, circles = run_search(evaluator, debug=debug)
    elapsed_time = time.time() - start_time
    
    c = calculate_result(p, c_multiplier)
    ct = get_distance_traveled(circles, debug=debug)
    return p, c, ct, circles, elapsed_time

def main() -> None:
    """Main execution function for command-line usage."""
    args = parse_args()
    
    p, c, ct, circles, elapsed_time = run_simulation(
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
    print(f"CPU Time: {elapsed_time:.3f} seconds")

    print(circles)
    
    draw_circles(circles,
        title=f"Algorithm {args.algorithm}" + (" (Find All)" if args.find_all else ""),
        p=p,
        c=c,
        ct=ct,
        cpu_time=elapsed_time)

if __name__ == '__main__':
    main()
    exit(0)
    # # p = 2 ** (-1 / 2.5)

    PRECISION.set_precision(8)
    print(PRECISION.epsilon)
    p = 0.761484375
    # p = 0.665
    # p = 0.66
    # initial_circles = 2
    # initial_circles = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    # print(initial_circles)
    # x0 = [Circle(x=-0.4484065647207323, y=0.0, r=0.76), Circle(x=0.5601502470074224, y=0.3576865370211815, r=0.5776), Circle(x=0.5085024698409785, y=-0.5423395822010191, r=0.43897600000000003), Circle(x=-0.08623281286331508, y=-0.8548665419624422, r=0.33362176), Circle(x=0.005611148628313943, y=0.845671058086685, r=0.2535525376), Circle(x=-0.36929874870667445, y=0.8483471293981908, r=0.19269992857600002), Circle(x=0.9109084345662557, y=-0.20157729210020672, r=0.14645194571776002), Circle(x=-0.504228918956284, y=-0.8149227497911052, r=0.1113034787454976), Circle(x=0.3021577501867461, y=0.9262540702371836, r=0.08459064384657819), Circle(x=-0.6012817587761183, y=0.77304297986932, r=0.06428888932339942), Circle(x=0.26754890384408, y=-0.9408068775967778, r=0.048859555885783564), Circle(x=-0.6314869169678121, y=-0.7542786634108117, r=0.037133262473195504), Circle(x=0.9853587681711702, y=-0.05326624279204786, r=0.028221279479628585), Circle(x=0.9207169673688986, y=-0.3621657909073574, r=0.021448172404517726), Circle(x=-0.6695299587329187, y=0.7338448039781391, r=0.01630061102743347), Circle(x=-0.674232691709327, y=-0.7313005216874521, r=0.012388464380849439), Circle(x=-0.42246009690914194, y=-0.8990886224749783, r=0.009415232929445573), Circle(x=0.3923821065912721, y=0.9148484785610943, r=0.007155577026378636), Circle(x=0.3210642414130638, y=-0.9432280565483033, r=0.005438238540047763), Circle(x=-0.21732469079958655, y=0.9729726424290217, r=0.0041330612904363), Circle(x=0.9972202401594649, y=-0.023752180701944764, r=0.003141126580731588), Circle(x=-0.687473258757824, y=0.7234982981943359, r=0.002387256201356007), Circle(x=-0.6882868713814427, y=-0.7230054226437646, r=0.0018143147130305654), Circle(x=0.22786445607561054, y=0.9715703546152582, r=0.0013788791819032296), Circle(x=0.9205916643858245, y=-0.38595229982605295, r=0.0010479481782464546)]
    # x0 = [Circle(x=-0.4830411, y=0, r=0.66), Circle(x=0.5318464, y=0.2416313, r=0.5361865), Circle(x=0.2861341, y=-0.6065730, r=0.4356), Circle(x=-0.03615046, y=0.7759997, r=0.3538831), Circle(x=-0.3231259, y=-0.7974621, r=0.287496), Circle(x=0.8054224, y=-0.3762863, r=0.2335629), Circle(x=-0.5249223, y=0.7555362, r=0.1897474), Circle(x=0.4046515, y=0.8577455, r=0.1541515), Circle(x=-0.6913100, y=-0.6914375, r=0.1252333), Circle(x=-0.01136641, y=-0.9794021, r=0.1017400), Circle(x=0.9682765, y=-0.1338024, r=0.08265395), Circle(x=-0.7370363, y=0.6390875, r=0.06714839), Circle(x=0.7482672, y=-0.6419571, r=0.05455161), Circle(x=0.5791013, y=0.7920574, r=0.04431794), Circle(x=-0.3782377, y=0.9100824, r=0.03600406), Circle(x=0.1705255, y=-0.1736458, r=0.02924984), Circle(x=-0.7942498, y=-0.5902347, r=0.02376268), Circle(x=-0.8009848, y=0.5868187, r=0.01930489), Circle(x=0.9929994, y=-0.04188113, r=0.01568337), Circle(x=0.2742226, y=0.9559963, r=0.01274123), Circle(x=0.62783818683958070305806131727877072989940643310546875, y=0.77326822321228105838741839761496521532535552978515625, r=0.01035102), Circle(x=0.71628027964776486147258083292399533092975616455078125, y=-0.69301439361814309858544902454013936221599578857421875, r=0.008409211), Circle(x=-0.81731057784901739449168189821648411452770233154296875, y=-0.57198732142082653329140384812490083277225494384765625, r=0.006831675), Circle(x=-0.81871758378737402583880111706093885004520416259765625, y=0.5706265440066895511250777417444624006748199462890625, r=0.005550079), Circle(x=0.025271821720933902721828445692153763957321643829345703125, y=0.424918884588280476588550982341985218226909637451171875, r=0.004508906), Circle(x=-0.70887689170958678719358658781857229769229888916015625, y=0.70214588860013826820960503027890808880329132080078125, r=0.003663052), Circle(x=0.99854446726559464853067993317381478846073150634765625, y=-0.024729054068863824678015106428574654273688793182373046875, r=0.002975878), Circle(x=0.6389450658617759071233876966289244592189788818359375, y=0.76799714141834407588049771220539696514606475830078125, r=0.002417615), Circle(x=-0.6115306319420650282836504629813134670257568359375, y=-0.7895569141375948785110949756926856935024261474609375, r=0.001964079), Circle(x=-0.823424844433121361220173639594577252864837646484375, y=-0.56623809834514970962260349551797844469547271728515625, r=0.001595626), Circle(x=-0.8237548305303843410030140148592181503772735595703125, y=0.5659288806692226447836446823203004896640777587890625, r=0.001296292), Circle(x=0.71190112698686214276477812745724804699420928955078125, y=-0.701426679930230445592087562545202672481536865234375, r=0.001053113)]
    # circles = place_algorithm_11(p, initial_circles=initial_circles, pk=default_pk, initial_guess=x0)
    # # circles = [Circle(x=-0.4660305, y=0, r=0.7578583), Circle(x=0.6219811, y=0.2293612, r=0.5743492), Circle(x=0.3799721, y=-0.6418602, r=0.4352753), Circle(x=0.09652912, y=0.8269215, r=0.3298770), Circle(x=-0.1954859, y=-0.8560651, r=0.2500000), Circle(x=-0.3625750, y=0.8458253, r=0.1894646), Circle(x=0.8360550, y=-0.3959104, r=0.1435873), Circle(x=-0.5261192, y=-0.8089158, r=0.1088188), Circle(x=0.480869862587108898299703696466167457401752471923828125, y=0.83068605528566052953465259633958339691162109375, r=0.08246926), Circle(x=-0.59274082610672007565000285467249341309070587158203125, y=0.77314890488008958246979318573721684515476226806640625, r=0.06250001), Circle(x=0.05955299818111030318856791154757956974208354949951171875, y=-0.97322066399223505772653197709587402641773223876953125, r=0.04736615), Circle(x=0.945071667223251754563762006000615656375885009765625, y=-0.267485111240606254767726568388752639293670654296875, r=0.03589683), Circle(x=-0.64508836223532295406357661704532802104949951171875, y=-0.7483326183722491808936183588230051100254058837890625, r=0.02720471), Circle(x=0.8187915289415081954160768873407505452632904052734375, y=-0.55365355153100981322467077916371636092662811279296875, r=0.02061732), Circle(x=-0.66262275981678431246990612635272555053234100341796875, y=0.7393680566860967307007967974641360342502593994140625, r=0.01562501), Circle(x=0.57264705977377372558834167648456059396266937255859375, y=0.80989263449354031987326152375317178666591644287109375, r=0.01184154), Circle(x=-0.20882603562105661598735650841263122856616973876953125, y=0.97027272385723339898078165788319893181324005126953125, r=0.008974209), Circle(x=0.424209419172794033325857299132621847093105316162109375, y=0.89909879687231797351643081128713674843311309814453125, r=0.006801179), Circle(x=0.110286511239986950716485125667531974613666534423828125, y=-0.98936943284950518151532605770626105368137359619140625, r=0.005154330), Circle(x=-0.675947897554812104914390147314406931400299072265625, y=-0.732007983290130948006435573915950953960418701171875, r=0.003906252), Circle(x=0.9684947036287188115721846770611591637134552001953125, y=-0.233445824486877284709152036157320253551006317138671875, r=0.002960385), Circle(x=-0.446946708195753383829895710732671432197093963623046875, y=-0.8895709389850099402252681102254427969455718994140625, r=0.002243553), Circle(x=-0.55448923632514379722380226667155511677265167236328125, y=0.82695865409689961467165630892850458621978759765625, r=0.001700295), Circle(x=-0.446946708195759379034228686577989719808101654052734375, y=-0.88957093898502126450011928682215511798858642578125, r=0.001288583)]
    # # circles = [Circle(x=-0.4621705, y=0, r=0.7578583), Circle(x=0.6256425, y=0.2326335, r=0.5743492), Circle(x=0.3799522, y=-0.6394088, r=0.4352753), Circle(x=0.09435549, y=0.8280994, r=0.3298770), Circle(x=-0.1911341, y=-0.8601548, r=0.2500000), Circle(x=-0.3621675, y=0.8498134, r=0.1894646), Circle(x=0.8355603, y=-0.3944838, r=0.1435873), Circle(x=-0.52249459542098286579658861228381283581256866455078125, y=-0.79951779020669666575571454814053140580654144287109375, r=0.1088188), Circle(x=0.478108268017147286510493131572729907929897308349609375, y=0.83226584615214205253863610778353177011013031005859375, r=0.08246926), Circle(x=-0.5923398972211406654508891733712516725063323974609375, y=0.77278031110463363262397251673974096775054931640625, r=0.06250001), Circle(x=0.06440370617754369308993744880353915505111217498779296875, y=-0.9739136913108425996910000321804545819759368896484375, r=0.04736615), Circle(x=0.945172075466456984571550492546521127223968505859375, y=-0.266710364048800219194390592747367918491363525390625, r=0.03589683), Circle(x=-0.64607838948470541762247876249602995812892913818359375, y=-0.7476118730071712459306354503496550023555755615234375, r=0.02720471), Circle(x=0.81929930229130787378011291366419754922389984130859375, y=-0.5526431474161856982618701294995844364166259765625, r=0.02061732), Circle(x=-0.66298605878645922029335224578971974551677703857421875, y=0.73858704278622078209792789493803866207599639892578125, r=0.01562501), Circle(x=0.569612049146654531028843848616816103458404541015625, y=0.81232234708392148103683894078130833804607391357421875, r=0.01184154), Circle(x=-0.446152987284383739652326994473696686327457427978515625, y=-0.8865485790819789269079365112702362239360809326171875, r=0.008974209), Circle(x=-0.2089098823704004515011689591119647957384586334228515625, y=0.971964861975724403464482747949659824371337890625, r=0.006801179), Circle(x=0.421736682759198122649735296363360248506069183349609375, y=0.90055484730787005442920190034783445298671722412109375, r=0.005154330), Circle(x=-0.67710409240092073179795306714368052780628204345703125, y=-0.7307063094435901628997953594080172479152679443359375, r=0.003906252), Circle(x=0.11458748311946047315768026919613475911319255828857421875, y=-0.9894749187324034522816873504780232906341552734375, r=0.002960385), Circle(x=0.96865022960213431613141210618778131902217864990234375, y=-0.2327812901864864947309996523472364060580730438232421875, r=0.002243553), Circle(x=-0.55390670905321937045329150350880809128284454345703125, y=0.8267531015774716163235780186369083821773529052734375, r=0.001700295), Circle(x=0.42173668275920039860693577793426811695098876953125, y=0.90055484730787405123209055091137997806072235107421875, r=0.001288583)]

    # # works for p = 0.67, find all
    # circles = [Circle(x=-0.4908906, y=0, r=0.67), Circle(x=0.5804569, y=0.2405622, r=0.5484186), Circle(x=0.208870012530215920509846228014794178307056427001953125, y=-0.624393270985229786873560442472808063030242919921875, r=0.4489), Circle(x=-0.0687488043892642564092199108927161432802677154541015625, y=0.77310537179094873661000519859953783452510833740234375, r=0.3674405), Circle(x=-0.393608215151188989278097096757846884429454803466796875, y=-0.791659521973363755620312076644040644168853759765625, r=0.300763), Circle(x=0.7684885770609046762302796196308918297290802001953125, y=-0.42347591574216758569804142098291777074337005615234375, r=0.2461851), Circle(x=-0.5598258887884235424081680321251042187213897705078125, y=0.74140036004681342252098374956403858959674835205078125, r=0.2015112), Circle(x=0.3929231971960387426179295289330184459686279296875, y=0.8388412309319217552427971895667724311351776123046875, r=0.1649440), Circle(x=-0.719611742351108585324936939286999404430389404296875, y=-0.6575628068270826798169537141802720725536346435546875, r=0.1350125), Circle(x=0.9519120239052198950702177171478979289531707763671875, y=-0.2013058037931811650178559602863970212638378143310546875, r=0.1105125), Circle(x=0.6797276989002438707387909744284115731716156005859375, y=-0.69765922557720283503357450172188691794872283935546875, r=0.09045838), Circle(x=-0.10946213991471363813356987293445854447782039642333984375, y=-0.9722348976327073177827742256340570747852325439453125, r=0.07404338), Circle(x=-0.7605164806089919071752092349925078451633453369140625, y=0.62843717938068355266523212776519358158111572265625, r=0.06060712), Circle(x=0.1816497487910462005356038162062759511172771453857421875, y=-0.158394717928385109217970239114947617053985595703125, r=0.04960906), Circle(x=0.57507623236683491629861464389250613749027252197265625, y=0.80242357988825607773009096490568481385707855224609375, r=0.04060677), Circle(x=0.05021068790505144041613760919062769971787929534912109375, y=0.413950344752257615166257664895965717732906341552734375, r=0.03323807), Circle(x=0.26069828357698343612725011553266085684299468994140625, y=0.956030152369351515773132632602937519550323486328125, r=0.02720653), Circle(x=-0.421038711816131117071648759520030580461025238037109375, y=0.8995030400165087147712483783834613859653472900390625, r=0.02226951), Circle(x=0.539648777974118143418991166981868445873260498046875, y=-0.31223545371566785444628067125449888408184051513671875, r=0.01822838), Circle(x=0.614181769376461073051132188993506133556365966796875, y=0.78848548708470478363352640371886081993579864501953125, r=0.01492057), Circle(x=-0.03952937922512121671214657681048265658318996429443359375, y=-0.9988086069087371843266964788199402391910552978515625, r=0.01221301)]

    # # not quite 0.66, find all ()
    # circles = [Circle(x=-0.4833120005694204, y=0.0, r=0.66), Circle(x=0.5317358816274448, y=0.24214677801153794, r=0.5361865347059734), Circle(x=0.2866995324856751, y=-0.6057168722596091, r=0.43560000000000004), Circle(x=-0.03634378304984366, y=0.7760127386996647, r=0.3538831129059425), Circle(x=-0.3222367103102662, y=-0.7973038313796376, r=0.28749600000000003), Circle(x=0.806303934058124, y=-0.37620487369057065, r=0.23356285451792205), Circle(x=-0.5250033675341838, y=0.7553755491823693, r=0.18974736000000003), Circle(x=0.40442850342336056, y=0.8582933498770591, r=0.15415148398182857), Circle(x=-0.691059950219828, y=-0.6915931103271704, r=0.12523325760000004), Circle(x=-0.01074808016882956, y=-0.9791293856616984, r=0.10173997942800685), Circle(x=0.968105880099089, y=-0.13307310577236536, r=0.08265395001600002), Circle(x=-0.7372260813132544, y=0.6392121051535343, r=0.06714838642248452), Circle(x=0.7483569093084873, y=-0.6421822341018402, r=0.05455160701056002), Circle(x=0.5787911464684424, y=0.792475355647242, r=0.04431793503883979), Circle(x=-0.3783609154943018, y=0.9100056949885919, r=0.03600406062696961), Circle(x=0.17023035363473682, y=-0.17308240345680925, r=0.02924983712563426), Circle(x=-0.7940488543707009, y=-0.5903884590478149, r=0.023762680013799945), Circle(x=-0.8011533708056912, y=0.5869624229300081, r=0.019304892502918614), Circle(x=0.9929091004154476, y=-0.041200121839168505, r=0.015683368809107964), Circle(x=0.2737850848818797, y=0.9561115144916119, r=0.012741229051926286), Circle(x=0.6280379351115938, y=0.7739768544046264, r=0.010351023414011257), Circle(x=0.7153594036835527, y=-0.6940910186896739, r=0.008409211174271349), Circle(x=-0.8174807581765395, y=-0.5724208346499913, r=0.006831675453247431), Circle(x=-0.8187441103697091, y=0.5706992330471568, r=0.005550079375019091), Circle(x=0.02496122751359406, y=0.4249237094247865, r=0.004508905799143304), Circle(x=-0.7089524993528952, y=0.702159274516865, r=0.0036630523875126), Circle(x=0.9985467997821961, y=-0.024052259357123327, r=0.002975877827434581), Circle(x=-0.6108657412256391, y=-0.7897111771875888, r=0.0024176145757583162), Circle(x=0.6386706275767347, y=0.7683935632960457, r=0.0019640793661068233), Circle(x=-0.8232931912398008, y=-0.5664482596985936, r=0.0015956256200004887), Circle(x=-0.8237037371758822, y=0.5660758028031595, r=0.0012962923816305036), Circle(x=0.09069262756666946, y=-0.9953286259051966, r=0.0010531129092003226)]

    # works for 0.66109, find all (precision 8)
    circles = [Circle(x=-0.471474198161091, y=0.0, r=0.6610867748260498), Circle(x=0.5344356300402844, y=0.2516804101514229, r=0.5375114297981989), Circle(x=0.2931742180401616, y=-0.601069880168715, r=0.4370357238499083), Circle(x=-0.04714303745451637, y=0.7758512566130683, r=0.35534169755742995), Circle(x=-0.3142139475405279, y=-0.8046332150144365, r=0.288918537163704), Circle(x=0.8118390255823599, y=-0.36471140452782463, r=0.234911696799455), Circle(x=-0.5387703656339764, y=0.7587953426460567, r=0.19100022392101332), Circle(x=0.3969877576912187, y=0.8670636387453872, r=0.1552970160060666), Circle(x=-0.6766721557954213, y=-0.6974955984567275, r=0.12626772202299602), Circle(x=-0.0007235223709840529, y=-0.9626115150116027, r=0.10266480345156001), Circle(x=0.9633341312578282, y=-0.11891457640707916, r=0.08347392111681462), Circle(x=-0.7444685494440092, y=0.6225383563303094, r=0.06787034380194211), Circle(x=0.7550232014179635, y=-0.6321053383253897, r=0.05518350529319908), Circle(x=0.5758672010215882, y=0.8014051020093613, r=0.04486818669036109), Circle(x=-0.7899071318400825, y=-0.5926293697629225, r=0.036481085537877225), Circle(x=-0.39170629655318034, y=0.9053792971775639, r=0.02966176483142391), Circle(x=0.1805013539778567, y=-0.16761877785458698, r=0.024117163180388503), Circle(x=-0.8112484919241976, y=0.5743617397272538, r=0.019609000448054782), Circle(x=0.2632988745338545, y=0.9590790103386507, r=0.015943537624876596), Circle(x=0.9953073223054228, y=-0.033096345759572744, r=0.012963250863767102), Circle(x=-0.8224547412002893, y=-0.5636849690645723, r=0.010540061867747447), Circle(x=0.7250581514342697, y=-0.6844248514518876, r=0.008569833704788797), Circle(x=-0.7196911569465999, y=0.6888864505391404, r=0.006967895506616191), Circle(x=-0.8278587049546778, y=0.558450093823494, r=0.005665403724694405), Circle(x=0.6194732654145584, y=0.7835784766107168, r=0.004606383567793822), Circle(x=0.0991572723911561, y=-0.9939795089945883, r=0.0037453234764457137), Circle(x=-0.3697771729955276, y=0.927787999834231, r=0.0030452192564445306), Circle(x=0.9992132615438009, y=-0.01950720668937055, r=0.0024759838177237856), Circle(x=-0.8309274328131553, y=-0.5554443592754847, r=0.002013154176881096), Circle(x=0.3070732302519927, y=0.739540504000506, r=0.0016368401565805075), Circle(x=0.5321008454033898, y=0.7897217135307991, r=0.0013308696020219149), Circle(x=0.7208652595980174, y=-0.6925910798291882, r=0.001082093380019574), Circle(x=0.2043602343467095, y=-0.172881259224821, r=0.0008798202929146962), Circle(x=0.7999448298492995, y=-0.599568618464722, r=0.0007153576226577594), Circle(x=-0.8321882995536307, y=0.5541992080067899, r=0.000581637559869487), Circle(x=-0.7146097042370902, y=0.6837115548508587, r=0.00047291346361004845), Circle(x=-0.8324932451267179, y=-0.5538971821394532, r=0.0003845128985718126), Circle(x=-0.36744323682740365, y=0.9299420037489814, r=0.00031263683642978343), Circle(x=0.1029697232468548, y=-0.994617923617098, r=0.00025419639199585563), Circle(x=-0.8326543609898214, y=0.5537375454529609, r=0.0002066800778871848), Circle(x=0.7203326234148276, y=-0.6935891006027869, r=0.0001680458729569585)]

    # works for p = 0.76067, find one (precision 8)
    # circles = [Circle(x=-0.4652323341815545, y=0.0, r=0.7606738281250001), Circle(x=0.631835807925796, y=0.1903730016493503, r=0.5786246727943422), Circle(x=0.3697940080061505, y=-0.6721376127407843, r=0.4401446449020478), Circle(x=0.10868020756478226, y=0.809927396170051, r=0.3348065119663595), Circle(x=-0.2268096324157326, y=-0.8507008005358745, r=0.2546785511386293), Circle(x=-0.35105515226951206, y=0.8525717091300288, r=0.19372730843594976), Circle(x=0.8334512813738311, y=-0.45141887403603526, r=0.14736329332032652), Circle(x=0.5066812387531501, y=0.8205098167780042, r=0.11209540045508003), Circle(x=-0.5338611091463583, y=-0.8110230105587689, r=0.0852680373793706), Circle(x=-0.5799133384414544, y=0.7846827534845661, r=0.06486116441007143), Circle(x=0.024957526866519295, y=-0.980222221404978, r=0.049338190228454044), Circle(x=-0.6306083956676077, y=-0.7532837868134962, r=0.03753027003383761), Circle(x=0.9308286536798448, y=-0.3196167621135757, r=0.028548294177204232), Circle(x=-0.6507673738092681, y=0.7465983474462492, r=0.02171594021821259), Circle(x=0.621415822507479, y=0.774187508417866, r=0.01651874737712142), Circle(x=-0.1931950036775884, y=0.9728756836537937, r=0.012565378803184755), Circle(x=-0.6723563363140392, y=-0.7368493091564825, r=0.00955815479605928), Circle(x=-0.6750275665055643, y=0.7343573392884171, r=0.0072706381985297415), Circle(x=0.9514513927696395, y=-0.2968022336890622, r=0.0055305841913874726), Circle(x=-0.18133794081650303, y=0.9813524299000489, r=0.004206970648630317), Circle(x=0.9554387725018909, y=-0.2904077680461287, r=0.0032001324681031375), Circle(x=-0.6814658353052274, y=-0.7300857539758098, r=0.0024342570150191183), Circle(x=-0.682339445375422, y=0.7298309511390368, r=0.0018516756022547286), Circle(x=0.6379088284879326, y=0.7690965724631914, r=0.0014085211688127694), Circle(x=0.07213390677064563, y=-0.9969650230993562, r=0.0010714251894759087), Circle(x=-0.6842523100838022, y=-0.7289071056215444, r=0.000815005100428193), Circle(x=-0.6843349897161745, y=0.7287749660796549, r=0.0006199530496841137), Circle(x=-0.17762853730071212, y=0.9838848899039228, r=0.0004715820595609831), Circle(x=-0.6851032171052384, y=0.7282938748064527, r=0.00035872013052132485), Circle(x=-0.6850662861629557, y=-0.7282949943196, r=0.00027286901490915585), Circle(x=-0.06988362735558888, y=-0.6499857099214075, r=0.00020756431814764528), Circle(x=0.9576159033083701, y=-0.287901726864696, r=0.00015788874446752475), Circle(x=-0.6854133856566726, y=-0.7281474087016567, r=0.00012010183567196199)]

    # not quite for p = 0.76, find one (0.000294)
    # circles = [Circle(x=-0.4484065647207323, y=0.0, r=0.76), Circle(x=0.5601502470074224, y=0.3576865370211815, r=0.5776), Circle(x=0.5085024698409785, y=-0.5423395822010191, r=0.43897600000000003), Circle(x=-0.08623281286331508, y=-0.8548665419624422, r=0.33362176), Circle(x=0.005611148628313943, y=0.845671058086685, r=0.2535525376), Circle(x=-0.36929874870667445, y=0.8483471293981908, r=0.19269992857600002), Circle(x=0.9109084345662557, y=-0.20157729210020672, r=0.14645194571776002), Circle(x=-0.504228918956284, y=-0.8149227497911052, r=0.1113034787454976), Circle(x=0.3021577501867461, y=0.9262540702371836, r=0.08459064384657819), Circle(x=-0.6012817587761183, y=0.77304297986932, r=0.06428888932339942), Circle(x=0.26754890384408, y=-0.9408068775967778, r=0.048859555885783564), Circle(x=-0.6314869169678121, y=-0.7542786634108117, r=0.037133262473195504), Circle(x=0.9853587681711702, y=-0.05326624279204786, r=0.028221279479628585), Circle(x=0.9207169673688986, y=-0.3621657909073574, r=0.021448172404517726), Circle(x=-0.6695299587329187, y=0.7338448039781391, r=0.01630061102743347), Circle(x=-0.674232691709327, y=-0.7313005216874521, r=0.012388464380849439), Circle(x=-0.42246009690914194, y=-0.8990886224749783, r=0.009415232929445573), Circle(x=0.3923821065912721, y=0.9148484785610943, r=0.007155577026378636), Circle(x=0.3210642414130638, y=-0.9432280565483033, r=0.005438238540047763), Circle(x=-0.21732469079958655, y=0.9729726424290217, r=0.0041330612904363), Circle(x=0.9972202401594649, y=-0.023752180701944764, r=0.003141126580731588), Circle(x=-0.687473258757824, y=0.7234982981943359, r=0.002387256201356007), Circle(x=-0.6882868713814427, y=-0.7230054226437646, r=0.0018143147130305654), Circle(x=0.22786445607561054, y=0.9715703546152582, r=0.0013788791819032296), Circle(x=0.9205916643858245, y=-0.38595229982605295, r=0.0010479481782464546)]
    # # # # # # for square in get_all_uncovered_squares(circles):
    # # # # # #     print(square)
    # # # # # squares = list(get_all_uncovered_squares(circles))
    print(len(circles[:-35]))
    circles = add_intelligent_circles(circles[0].r, find_all_pk, circles[:-25])[0]

    square = get_biggest_uncovered_square(circles)
    print(circles, len(circles), get_empty_area(circles), covers_unit_circle(circles))
    print(square)


    draw_circles(circles, squares=[], title='Algorithm 11')

    # circles = place_algorithm_5(0.79)

    # circle_polygons = [PRECISION.get_circle_polygon(circle) for circle in circles]

    # circle_polygon_union = shapely.union_all(circle_polygons)

    # unit_circle_polygon = PRECISION.unit_circle_polygon

    # difference_polygons = unit_circle_polygon.difference(circle_polygon_union)

    # largest_geom = max(difference_polygons.geoms, key=lambda g: g.area) if hasattr(difference_polygons, 'geoms') else difference_polygons # type: ignore
    
    # new_circle = intelligently_minimize(largest_geom, 0.2)

    # draw_circles([new_circle], polygons=[largest_geom])
