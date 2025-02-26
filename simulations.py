from typing import Callable
from decimal import Decimal, getcontext
import argparse
import time

import shapely
from decimal_math import asin, atan, cos, log2, pi, sin
from utils import EPSILON, Circle, covers_unit_circle, draw_circles, binary_search, get_biggest_uncovered_square, get_distance_traveled

from scipy import optimize

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

def place_algorithm_11(p: Decimal, pk: Callable[[Decimal, Decimal], Decimal] = lambda p, k: p**k) -> list[Circle]:
    def create_circles_from_params(params: list[float], k: int) -> list[Circle]:
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
    
    num_calls = 0
    min_area = pi()

    def objective_function(x: list[float]):
        nonlocal num_calls, min_area
        k = (len(x) + 1) // 2  # Calculate k based on the length of x
        
        # Create circles using the helper function
        circles = create_circles_from_params(x, k)
        _, remaining_area = add_centroid_circles(p, pk, circles)
                
        num_calls += 1
        
        if remaining_area < min_area:
            min_area = remaining_area
            param_str = ", ".join([f"{val:.7f}" for val in x])
            print(f"Call {num_calls}: {param_str} ->\n\t{remaining_area:.6f}")

        return remaining_area

    def optimize_circle_placement(k: int):
        # Create bounds for optimization
        # First parameter (dx) is between 0 and 1
        bounds: list[tuple[float, float]] = [(0, 1)]
        
        # For each of the k-1 remaining circles, we need bounds for theta and d
        # theta can be between -pi and pi
        # d can be between 0 and 1
        for _ in range(1, k):
            bounds.extend([(-float(pi()), float(pi())), (0, 1)])
        
        result = optimize.differential_evolution(
            objective_function,
            bounds=bounds,
        )
        
        return result

    # Set the number of circles to optimize
    k = 2
    
    # Run the optimization
    result = optimize_circle_placement(k)

    print(f"Optimization complete in {result.nit} iterations")
    
    # Create the final circles using the optimized parameters
    circles = create_circles_from_params(result.x, k)
    
    # Print the positions of all circles
    for i, circle in enumerate(circles):
        print(f"Circle {i+1}: center=({circle.x:.7f}, {circle.y:.7f}), radius={circle.r:.7f}")
    
    # Add centroid circles
    result = add_centroid_circles(p, pk, circles)
    
    print(f'Final Remaining Area: {result[1]:.6f}')
    
    return circles

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
    elif algorithm == 11:
        return place_algorithm_11
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
    # main()
    getcontext().prec = 7
    circles = place_algorithm_11(Decimal('0.76'))
    # circles = [Circle(x=Decimal('-0.4559316'), y=Decimal('0'), r=Decimal('0.76')), Circle(x=Decimal('0.6326715'), y=Decimal('0.2425936'), r=Decimal('0.5776')), Circle(x=Decimal('0.3929132'), y=Decimal('-0.6341119'), r=Decimal('0.438976')), Circle(x=Decimal('0.08847175'), y=Decimal('0.8351524'), r=Decimal('0.3336218')), Circle(x=Decimal('-0.1790904'), y=Decimal('-0.8592810'), r=Decimal('0.2535525')), Circle(x=Decimal('-0.3697012'), y=Decimal('0.8570674'), r=Decimal('0.1926999')), Circle(x=Decimal('0.85096475187464515332891323851072229444980621337890625'), y=Decimal('-0.376728526573564204138477862215950153768062591552734375'), r=Decimal('0.1464519')), Circle(x=Decimal('-0.51713609699463380930950506808585487306118011474609375'), y=Decimal('-0.8022813244003079713451143106794916093349456787109375'), r=Decimal('0.1113035')), Circle(x=Decimal('0.47166381955873504239207250066101551055908203125'), y=Decimal('0.83887634093329010998019157341332174837589263916015625'), r=Decimal('0.08459064')), Circle(x=Decimal('-0.59740654310518037650723499609739519655704498291015625'), y=Decimal('0.770928885039575018112145698978565633296966552734375'), r=Decimal('0.06428889')), Circle(x=Decimal('0.07860861237381600030715844695805571973323822021484375'), y=Decimal('-0.9738927010103466397339389004628174006938934326171875'), r=Decimal('0.04885956')), Circle(x=Decimal('-0.6439489543326024634239956867531873285770416259765625'), y=Decimal('-0.749072819819898594317919560126028954982757568359375'), r=Decimal('0.03713326')), Circle(x=Decimal('0.95544327095467374011406036515836603939533233642578125'), y=Decimal('-0.25142437940531447981840074135106988251209259033203125'), r=Decimal('0.02822128')), Circle(x=Decimal('0.83188665918665039011870021568029187619686126708984375'), y=Decimal('-0.53567751202964541956674793254933319985866546630859375'), r=Decimal('0.02144817')), Circle(x=Decimal('-0.66721668392493793664499435180914588272571563720703125'), y=Decimal('0.7365367974490559799249922434682957828044891357421875'), r=Decimal('0.01630061')), Circle(x=Decimal('-0.437529007912684064773856107422034256160259246826171875'), y=Decimal('-0.89033107389682808463504670726251788437366485595703125'), r=Decimal('0.01238846')), Circle(x=Decimal('0.56174084806983370921074083526036702096462249755859375'), y=Decimal('0.8211210575510274889410311516257934272289276123046875'), r=Decimal('0.009415233')), Circle(x=Decimal('-0.681390175979557977115064204554073512554168701171875'), y=Decimal('-0.728452969060096933162640198133885860443115234375'), r=Decimal('0.007155577')), Circle(x=Decimal('0.129165254889673064564448168312082998454570770263671875'), y=Decimal('-0.98870711938976663102351949419244192540645599365234375'), r=Decimal('0.005438239')), Circle(x=Decimal('-0.68461116017606593597832898012711666524410247802734375'), y=Decimal('0.7265675664620612206334726579370908439159393310546875'), r=Decimal('0.004133061')), Circle(x=Decimal('0.416712036607134905796812063272227533161640167236328125'), y=Decimal('0.90649958832595889379746267877635546028614044189453125'), r=Decimal('0.003141127')), Circle(x=Decimal('0.82715638130172164377285071168444119393825531005859375'), y=Decimal('-0.55890149213198458966189718921668827533721923828125'), r=Decimal('0.002387256')), Circle(x=Decimal('0.9725047628775518315791259738034568727016448974609375'), y=Decimal('-0.2263283133827659054926328963119885884225368499755859375'), r=Decimal('0.001814315')), Circle(x=Decimal('-0.5611671620601785814841377941775135695934295654296875'), y=Decimal('0.82564893766047775525152019326924346387386322021484375'), r=Decimal('0.001378879')), Circle(x=Decimal('0.572538515154466409740052768029272556304931640625'), y=Decimal('0.8183285027203999195677397437975741922855377197265625'), r=Decimal('0.001047948'))]
    # # # # for square in get_all_uncovered_squares(circles):
    # # # #     print(square)
    # # # squares = list(get_all_uncovered_squares(circles))
    square = get_biggest_uncovered_square(circles)
    print(circles, len(circles))
    print(square)
    # # print(square, square.side_length ** 2   )
    
    # draw_circles(circles, squares=[])

    # unit_circle = shapely.set_precision(Point(0, 0).buffer(1, 32), 0.000000000001)

    # other_circles = [shapely.set_precision(Point(float(circle.x), float(circle.y)).buffer(float(circle.r), 32), 0.000000000001) for circle in circles]
    # unit_circle = unit_circle.difference(shapely.union_all(other_circles))

    # other_circle_union = shapely.unary_union(other_circles, axis=None)  # type: BaseGeometry

    # print(type(other_circle_union))

    # print(unit_circle)

    # print area
    # print(unit_circle.area, unit_circle.is_empty)

    # # centers = [shapely.centroid(p) for p in unit_circle.geoms]
    # # print(centers)
    # centers = [shapely.minimum_bounding_circle(p) for p in unit_circle.geoms]

    # print(centers)

    # draw_circles([], polygons=list(unit_circle.geoms) + centers, title="Uncovered Area")


    # circles = [Circle(x=Decimal('-0.25'), y=Decimal('-0.25'), r=Decimal('0.7825317')), Circle(x=Decimal('0.625'), y=Decimal('0.125'), r=Decimal('0.61235586'))]

    # unit_circle = shapely.Point(0, 0).buffer(1)
    # circle_polygons = [shapely.Point(float(circle.x), float(circle.y)).buffer(float(circle.r)) for circle in circles]
    # uncovered_polygons = unit_circle.difference(shapely.unary_union(circle_polygons))

    # draw_circles(circles, polygons=uncovered_polygons.geoms, title="Algorithm 11")