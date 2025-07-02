"""
Marco Polo Problem: Geometric Localization Algorithms

This module implements 8 algorithms for the Marco Polo problem, a combinatorial approach 
to geometric localization using probe-based searching with binary responses. The algorithms
locate points of interest (POIs) using circular probes and minimize different metrics:
- P(n): Number of probes issued
- D(n): Total distance traveled by the search point  
- R_max: Maximum number of POI responses

Algorithms range from hexagonal tilings to sophisticated optimization-based approaches,
demonstrating various trade-offs between probe efficiency and travel distance.
"""
from typing import Callable, Optional, Any
import argparse
import time
import math

from src.geometry_types import PRECISION, Circle
from src.geometry_algorithms import covers_unit_circle, add_intelligent_circles, get_intersections
from src.algorithm_plot import draw_circles
from src.algorithm_utils import binary_search, get_distance_traveled


# Pre-computed algorithm results
ALGORITHM_1 = [Circle(x=0, y=0, r=0.5), Circle(x=0.75, y=0.43301270189221935, r=0.5), Circle(x=-1.6653345369377348e-16, y=0.8660254037844386, r=0.5), Circle(x=-0.7500000000000001, y=0.4330127018922191, r=0.5), Circle(x=-0.7499999999999998, y=-0.4330127018922196, r=0.5), Circle(x=6.38378239159465e-16, y=-0.8660254037844385, r=0.5), Circle(x=0.7500000000000004, y=-0.4330127018922183, r=0.5)]
ALGORITHM_2 = [Circle(x=0, y=0, r=0.5), Circle(x=0.5, y=0.5, r=0.7071067811865475), Circle(x=-0.5, y=0.5, r=0.7071067811865475), Circle(x=-0.7499999999999998, y=-0.43301270189221935, r=0.5), Circle(x=6.38378239159465e-16, y=-0.8660254037844385, r=0.5), Circle(x=0.7500000000000004, y=-0.4330127018922183, r=0.5)]
ALGORITHM_3 = [Circle(x=0.28789865898868494, y=0.4527836361234821, r=0.843860972560833), Circle(x=-0.6618325909742372, y=0.23428465945130714, r=0.712101341011315), Circle(x=-0.36476973938250307, y=-0.7112276461444245, r=0.6009145301876817), Circle(x=0.5569297259284322, y=-0.6577923047806103, r=0.5070883198701132), Circle(x=0.8987720218026585, y=0.09539463542963203, r=0.42791204277983247)]
ALGORITHM_4 = [Circle(x=0.3240584078787189, y=0.468021961196084, r=0.8221566712745698), Circle(x=-0.6536636388727417, y=-0.3403333825180739, r=0.675941592121281), Circle(x=0.19819922134273138, y=-0.8073917009340685, r=0.5557298893544654), Circle(x=-0.6588517518809158, y=0.597628201790139, r=0.4568970359594523), Circle(x=0.8687374815033344, y=-0.32278331395764914, r=0.3756409461996407)]
ALGORITHM_5 = [Circle(x=0, y=0, r=0.8342831628142449), Circle(x=0.5155444723022481, y=0.4997583109672561, r=0.6960283957553398), Circle(x=-0.45191758342842164, y=0.6771821719669994, r=0.5806847714192898), Circle(x=-0.8658878771785593, y=-0.12466365083014752, r=0.4844555276977519), Circle(x=-0.44332373517785195, y=-0.8000676091658153, r=0.4041730898905245), Circle(x=0.23657851953538434, y=-0.9112245982264008, r=0.3371948037582729), Circle(x=0.7415164177729718, y=-0.609109793002744, r=0.2813159473639805), Circle(x=0.9600399845076252, y=-0.19586529211226786, r=0.23469715831690727)]
ALGORITHM_6 = [Circle(x=0, y=0, r=0.8124407481991511), Circle(x=0.5643208368822755, y=0.4958455706596195, r=0.6600599693343965), Circle(x=-0.3572231106256195, y=0.7647334661094909, r=0.5362596153423458), Circle(x=-0.8939478838207515, y=0.10507543879251385, r=0.4356791631177244), Circle(x=-0.7179732867681129, y=-0.599353148346416, r=0.3539635052581441), Circle(x=-0.2082732502738843, y=-0.9229095419396449, r=0.28757437504712074), Circle(x=0.27639770841541267, y=-0.8891745432525523, r=0.23363714042618605), Circle(x=0.6027932264816553, y=-0.6964993285330394, r=0.18981633317496074), Circle(x=0.782929757696528, y=-0.4724868908724668, r=0.15421452374508446), Circle(x=0.8646018242986668, y=-0.2839050168222197, r=0.12529016305463217), Circle(x=0.8920706471869477, y=-0.16435352894045435, r=0.10179083381409901), Circle(x=0.9950262135480192, y=-0.05553112863558944, r=0.08269902118374205)]
ALGORITHM_7 = [Circle(x=0.3766095638275147, y=0.4845356542723805, r=0.78955078125), Circle(x=-0.6275047422486887, y=-0.46649990626374743, r=0.6233904361724854), Circle(x=0.22055685720357815, y=-0.8420780260559434, r=0.49219840590376407), Circle(x=-0.8363946658436578, y=0.38655122619585, r=0.38861563591132153), Circle(x=0.8405974221134691, y=-0.4463745438181471, r=0.30683177893974944), Circle(x=-0.4600705903487553, y=0.8541928925132176, r=0.2422592707742065), Circle(x=0.08366242225629675, y=-0.23730948262677515, r=0.19127599650483001), Circle(x=-0.3797354768401737, y=0.1731748713449593, r=0.15102211247476083), Circle(x=0.4914502254478055, y=-0.35144440898159507, r=0.11923962689047278), Circle(x=-0.4652119846895424, y=0.5499473330977886, r=0.09414574056733128), Circle(x=0.9318232713465783, y=-0.12293596444208141, r=0.07433284301629624), Circle(x=0.3311617259784244, y=-0.3349595104849733, r=0.0586895542760503), Circle(x=-0.43723848007056154, y=0.34499464742122243, r=0.04633838342986979), Circle(x=-0.4359177913258304, y=0.4247192866145801, r=0.03658650683891575), Circle(x=0.983269323540918, y=-0.06016623510468254, r=0.028886905057874397), Circle(x=0.2576485140468424, y=-0.3384791427308925, r=0.022807678456339308), Circle(x=0.9872487090293736, y=-0.022826769121943327, r=0.018007820343701495), Circle(x=0.27047360937857473, y=-0.30859021040107315, r=0.014218088620979157), Circle(x=-0.4184952978014185, y=0.46201937613118615, r=0.01122590297857583), Circle(x=-0.41254822050693163, y=0.38983550014736806, r=0.008863420466971248), Circle(x=1.0018747269313437, y=-0.00674233527601123, r=0.006998120554244389), Circle(x=-0.40476827646332914, y=0.378147326534274, r=0.00552537155088534), Circle(x=-0.4220866421019454, y=0.3928031992667419, r=0.004362561424698044), Circle(x=-0.43134390662214, y=0.46376467139943706, r=0.003444463781121454), Circle(x=0.2816293232336179, y=-0.2991930013893405, r=0.002719579069371773), Circle(x=-0.4086454225500263, y=0.3996048490523598, r=0.0021472457788936313)]
ALGORITHM_8 = [Circle(x=-0.4652323341815545, y=0.0, r=0.7606738281250001), Circle(x=0.631835807925796, y=0.1903730016493503, r=0.5786246727943422), Circle(x=0.3697940080061505, y=-0.6721376127407843, r=0.4401446449020478), Circle(x=0.10868020756478226, y=0.809927396170051, r=0.3348065119663595), Circle(x=-0.2268096324157326, y=-0.8507008005358745, r=0.2546785511386293), Circle(x=-0.35105515226951206, y=0.8525717091300288, r=0.19372730843594976), Circle(x=0.8334512813738311, y=-0.45141887403603526, r=0.14736329332032652), Circle(x=0.5066812387531501, y=0.8205098167780042, r=0.11209540045508003), Circle(x=-0.5338611091463583, y=-0.8110230105587689, r=0.0852680373793706), Circle(x=-0.5799133384414544, y=0.7846827534845661, r=0.06486116441007143), Circle(x=0.024957526866519295, y=-0.980222221404978, r=0.049338190228454044), Circle(x=-0.6306083956676077, y=-0.7532837868134962, r=0.03753027003383761), Circle(x=0.9308286536798448, y=-0.3196167621135757, r=0.028548294177204232), Circle(x=-0.6507673738092681, y=0.7465983474462492, r=0.02171594021821259), Circle(x=0.621415822507479, y=0.774187508417866, r=0.01651874737712142), Circle(x=-0.1931950036775884, y=0.9728756836537937, r=0.012565378803184755), Circle(x=-0.6723563363140392, y=-0.7368493091564825, r=0.00955815479605928), Circle(x=-0.6750275665055643, y=0.7343573392884171, r=0.0072706381985297415), Circle(x=0.9514513927696395, y=-0.2968022336890622, r=0.0055305841913874726), Circle(x=-0.18133794081650303, y=0.9813524299000489, r=0.004206970648630317), Circle(x=0.9554387725018909, y=-0.2904077680461287, r=0.0032001324681031375), Circle(x=-0.6814658353052274, y=-0.7300857539758098, r=0.0024342570150191183), Circle(x=-0.682339445375422, y=0.7298309511390368, r=0.0018516756022547286), Circle(x=0.6379088284879326, y=0.7690965724631914, r=0.0014085211688127694), Circle(x=0.07213390677064563, y=-0.9969650230993562, r=0.0010714251894759087), Circle(x=-0.6842523100838022, y=-0.7289071056215444, r=0.000815005100428193), Circle(x=-0.6843349897161745, y=0.7287749660796549, r=0.0006199530496841137), Circle(x=-0.17762853730071212, y=0.9838848899039228, r=0.0004715820595609831), Circle(x=-0.6851032171052384, y=0.7282938748064527, r=0.00035872013052132485), Circle(x=-0.6850662861629557, y=-0.7282949943196, r=0.00027286901490915585), Circle(x=-0.06988362735558888, y=-0.6499857099214075, r=0.00020756431814764528), Circle(x=0.9576159033083701, y=-0.287901726864696, r=0.00015788874446752475), Circle(x=-0.6854133856566726, y=-0.7281474087016567, r=0.00012010183567196199)]

ALGORITHMS: list[list[Circle]] = [[], ALGORITHM_1, ALGORITHM_2, ALGORITHM_3, ALGORITHM_4, ALGORITHM_5, ALGORITHM_6, ALGORITHM_7, ALGORITHM_8]


# Type definitions and placement functions
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


def place_algorithm_1():
    """Algorithm 1: Hexagonal Algorithm (Hexagonal)
    Uses a tiling of the search area with 7 hexagons of radius n/2.
    Probes 6 of the 7 hexagons with radius-n/2 probes (POI must be in last if others fail).
    Worst-case: P(n) ≤ 6⌈log n⌉ probes, D(n) ≤ 10.39n distance, R_max ≤ ⌈log n⌉ responses.
    """
    r = 0.5
    circles = [Circle(0, 0, r)]

    theta = 0
    theta_step = 2 * math.asin(r)
    while theta < 2 * math.pi:
        current_coord = (math.cos(theta), math.sin(theta))
        theta += theta_step
        next_coord = (math.cos(theta), math.sin(theta))

        circles.append(Circle(
            (current_coord[0] + next_coord[0]) / 2,
            (current_coord[1] + next_coord[1]) / 2,
            r
        ))

    return circles


def place_algorithm_2():
    """Algorithm 2: Modified Hexagonal Algorithm (Hexagonal)
    First probes upper two quadrants with radius n/√2 probes (eliminating 3 hexagons).
    Then probes 3 of the remaining 4 hexagons as in Algorithm 1.
    Better trade-off: P(n) ≤ 5⌈log n⌉ probes, D(n) ≤ 8.81n distance, R_max ≤ 2⌈log n⌉ responses.
    """
    circles = [
        Circle(0, 0, 0.5),
        Circle(0.5, 0.5, 1/math.sqrt(2)),
        Circle(-0.5, 0.5, 1/math.sqrt(2)),
    ]

    r = 0.5
    theta = math.pi
    theta_step = 2 * math.asin(r)
    while theta < 2 * math.pi:
        current_coord = (math.cos(theta), math.sin(theta))
        theta += theta_step
        next_coord = (math.cos(theta), math.sin(theta))

        circles.append(Circle(
            (current_coord[0] + next_coord[0]) / 2,
            (current_coord[1] + next_coord[1]) / 2,
            r
        ))

    return circles


def place_algorithm_3(p: float, pk: PkFunction = default_pk) -> list[Circle]:
    """Algorithm 3: Progressive Chord-Based Shrinking (Chord-Based)
    Places probe diameters as chords of the search circle in monotonic counterclockwise order.
    Uses progressively shrinking probes with ρ₁ ≈ 0.844 to avoid uncovered areas.
    Performance: P(n) < 4.08⌈log n⌉ probes, D(n) ≤ 6.95n distance.
    """
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


def place_algorithm_4(p: float, pk: PkFunction = default_pk) -> list[Circle]:
    """Algorithm 4: Reordered Chord Placement (Chord-Based)
    Non-monotonic version of Algorithm 3 with optimized probe placement.
    Places two largest probes side by side, alternates remaining probes to minimize overlap.
    Improved performance: P(n) < 3.54⌈log n⌉ probes, D(n) ≤ 9.31n distance.
    """
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

    return sorted(circles, key=lambda c: c.r, reverse=True)


def place_algorithm_5_5(p: float, pk: PkFunction = default_pk, final_optimization: bool = True) -> list[Circle]:
    """Algorithm 5: Central + Chords (Higher-Count Monotonic-Path)
    Places a central probe, then places remaining probes along perimeter.
    Uses chord-based placement for up to 8 probes per recursive level.
    Performance: P(n) < 3.83⌈log n⌉ probes, D(n) ≤ 6.72n distance.
    """
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
    """Algorithm 6: Central + Optimized Chords (Higher-Count Monotonic-Path)
    Advanced version of Algorithm 5 with geometric optimization for probe positioning.
    Balances coverage rate of inner and outer circumferences for optimal probe placement.
    Best distance performance: P(n) < 3.34⌈log n⌉ probes, D(n) ≤ 6.02n distance.
    """
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


def place_algorithm_8(p: float, pk: PkFunction,
                      initial_circles: int = 6,
                      optimization_kwargs: Optional[dict[str, Any]] = None,
                      initial_guess: Optional[list[Circle]] = None) -> list[Circle]:
    """Algorithm 8: Differential Evolution Optimization (Darting)
    Uses differential evolution algorithm to optimize placement of initial 6 probes.
    Applies greedy gap-filling method for remaining probes.
    Best probe performance: P(n) < 2.53⌈log n⌉ probes, D(n) ≤ 45.4n distance.
    """
    from src.algorithm_utils import optimize_circle_placement, create_circles_from_params, params_from_created_circles, objective_function_global
    
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
    result = optimize_circle_placement(p, initial_circles, pk, progress_callback, optimization_kwargs, x0)

    if not result.success:
        print(f"Warning: Optimization may not have converged: {result.message}")
    
    total_time = time.time() - start_time
    print(f"Optimization complete in {result.nit} iterations ({total_time:.2f}s)")
    
    # Create the final circles using the optimized parameters
    circles = create_circles_from_params(result.x, initial_circles, p, pk)
    
    # Add intelligent circles
    circles, area = add_intelligent_circles(p, pk, circles)

    print(f'Final Remaining Area: {area:.6f}')
    
    return [Circle(float(circle.x), float(circle.y), float(circle.r)) for circle in circles]


def get_configuration(args: argparse.Namespace) -> tuple[PkFunction, int]:
    """Get pk function and c_multiplier based on arguments."""
    if args.find_all:
        return find_all_pk, 2
    return default_pk, 1


def place_algorithm_7(p: float, pk: PkFunction = default_pk, circle_placement_algorithm: CirclePlacerFunction = dummy_placement_algorithm) -> list[Circle]:
    """Algorithm 7: Darting Non-Monotonic (Darting)
    Starts with Algorithm 4 (minus final probe), then greedily fills gaps.
    Uses computer-assisted probe placement to efficiently cover search area.
    Performance: P(n) < 2.93⌈log n⌉ probes, D(n) ≤ 25.8n distance.
    """
    circles = circle_placement_algorithm(p, pk)
    return add_intelligent_circles(p, pk, circles)[0]


def get_placement_algorithm(algorithm: float) -> CirclePlacerFunction:
    """Get the appropriate placement algorithm based on argument."""
    if algorithm == 3:
        return place_algorithm_3
    elif algorithm == 4:
        return place_algorithm_4
    elif algorithm == 5:
        return lambda p, pk: place_algorithm_5_5(p, pk, final_optimization=False)
    elif algorithm == 5.75:
        return place_algorithm_5_5
    elif algorithm == 6:
        return lambda p, pk: place_algorithm_6(p, pk, final_optimization=False)
    elif algorithm == 6.5:
        return place_algorithm_6
    elif algorithm == 7:
        circle_placement_algorithm: CirclePlacerFunction = lambda p, pk: place_algorithm_4(p, pk)[:-1]
        return lambda p, pk: place_algorithm_7(p, pk, circle_placement_algorithm)
    elif algorithm == 8:
        initial_guess_find_one = [Circle(x=-0.4652323341815545, y=0.0, r=0.7606738281250001), Circle(x=0.631835807925796, y=0.1903730016493503, r=0.5786246727943422), Circle(x=0.3697940080061505, y=-0.6721376127407843, r=0.4401446449020478), Circle(x=0.10868020756478226, y=0.809927396170051, r=0.3348065119663595), Circle(x=-0.2268096324157326, y=-0.8507008005358745, r=0.2546785511386293), Circle(x=-0.35105515226951206, y=0.8525717091300288, r=0.19372730843594976), Circle(x=0.8334512813738311, y=-0.45141887403603526, r=0.14736329332032652), Circle(x=0.5066812387531501, y=0.8205098167780042, r=0.11209540045508003), Circle(x=-0.5338611091463583, y=-0.8110230105587689, r=0.0852680373793706), Circle(x=-0.5799133384414544, y=0.7846827534845661, r=0.06486116441007143), Circle(x=0.024957526866519295, y=-0.980222221404978, r=0.049338190228454044), Circle(x=-0.6306083956676077, y=-0.7532837868134962, r=0.03753027003383761), Circle(x=0.9308286536798448, y=-0.3196167621135757, r=0.028548294177204232), Circle(x=-0.6507673738092681, y=0.7465983474462492, r=0.02171594021821259), Circle(x=0.621415822507479, y=0.774187508417866, r=0.01651874737712142), Circle(x=-0.1931950036775884, y=0.9728756836537937, r=0.012565378803184755), Circle(x=-0.6723563363140392, y=-0.7368493091564825, r=0.00955815479605928), Circle(x=-0.6750275665055643, y=0.7343573392884171, r=0.0072706381985297415), Circle(x=0.9514513927696395, y=-0.2968022336890622, r=0.0055305841913874726), Circle(x=-0.18133794081650303, y=0.9813524299000489, r=0.004206970648630317), Circle(x=0.9554387725018909, y=-0.2904077680461287, r=0.0032001324681031375), Circle(x=-0.6814658353052274, y=-0.7300857539758098, r=0.0024342570150191183), Circle(x=-0.682339445375422, y=0.7298309511390368, r=0.0018516756022547286), Circle(x=0.6379088284879326, y=0.7690965724631914, r=0.0014085211688127694), Circle(x=0.07213390677064563, y=-0.9969650230993562, r=0.0010714251894759087), Circle(x=-0.6842523100838022, y=-0.7289071056215444, r=0.000815005100428193), Circle(x=-0.6843349897161745, y=0.7287749660796549, r=0.0006199530496841137), Circle(x=-0.17762853730071212, y=0.9838848899039228, r=0.0004715820595609831), Circle(x=-0.6851032171052384, y=0.7282938748064527, r=0.00035872013052132485), Circle(x=-0.6850662861629557, y=-0.7282949943196, r=0.00027286901490915585), Circle(x=-0.06988362735558888, y=-0.6499857099214075, r=0.00020756431814764528), Circle(x=0.9576159033083701, y=-0.287901726864696, r=0.00015788874446752475), Circle(x=-0.6854133856566726, y=-0.7281474087016567, r=0.00012010183567196199)]
        initial_guess_find_all = [Circle(x=-0.473348414042532, y=0.0, r=0.6600893225800246), Circle(x=0.5343362143380699, y=0.24656189453562138, r=0.5362953873992997), Circle(x=0.30887579035559376, y=-0.6042218420332874, r=0.4357179137841558), Circle(x=-0.02083056251695819, y=0.7816097718162912, r=0.3540028589711956), Circle(x=-0.2825993952802953, y=-0.8047699691777478, r=0.28761274254576497), Circle(x=0.8234521696950104, y=-0.3638099536002596, r=0.2336735073696885), Circle(x=-0.4819141385282013, y=0.7828925135194501, r=0.18985010039241704), Circle(x=0.42451191741767436, y=0.8624873992880191, r=0.15424538718455608), Circle(x=-0.6396626599933831, y=-0.717876149449555, r=0.12531802415978022), Circle(x=-0.7141108348709663, y=0.6536564578703262, r=0.10181573313774724), Circle(x=0.011497666689047358, y=-0.9794447962833417, r=0.08272108967469649), Circle(x=0.9669304188180297, y=-0.12604033783935764, r=0.06720747831488413), Circle(x=-0.7659661432432211, y=-0.6078147256378162, r=0.05460330804645187), Circle(x=0.7662631254351678, y=-0.6220325194407933, r=0.04436293883318356), Circle(x=0.5917477912367785, y=0.7891413138234734, r=0.036043060619010824), Circle(x=0.17794651711818557, y=-0.1753295451104614, r=0.029283502242055205), Circle(x=-0.8093312265078569, y=0.5775022342433332, r=0.023791639467713618), Circle(x=0.9907735862946158, y=-0.04833171432876694, r=0.019329727157728852), Circle(x=-0.3481236078443935, y=0.9329748969841338, r=0.01570460717931126), Circle(x=-0.8183997876874411, y=-0.5694315222410222, r=0.012759346505201944), Circle(x=0.10159215419301831, y=-0.9905356984107909, r=0.01036644351437696), Circle(x=0.29195174733693935, y=0.9523927772427866, r=0.008422308391182555), Circle(x=0.7424365276740426, y=-0.6653533506288015, r=0.006842778676969178), Circle(x=0.998409169436137, y=-0.025644201706744713, r=0.00555947584049575), Circle(x=0.6293945722912323, y=0.7760138333188272, r=0.004516845141445621), Circle(x=-0.8279175154834418, y=0.5582362778620299, r=0.0036697506414528525), Circle(x=-0.8279993535205726, y=-0.557918318322183, r=0.0029815212496157157), Circle(x=0.029031208725864243, y=0.429970752047181, r=0.0024223632149542244), Circle(x=-0.12579272532573907, y=-0.5623803944100751, r=0.0019680703419167863), Circle(x=0.2077477551096776, y=-0.17976048763837638, r=0.0015989760936019044), Circle(x=-0.8311161830993191, y=-0.5554576959452511, r=0.001299102218785689), Circle(x=-0.8312157917442254, y=0.5552239664973123, r=0.0010554670464473352), Circle(x=0.7397890993915478, y=-0.6722214140176546, r=0.0008575235035604522), Circle(x=0.11203009116679852, y=-0.9933994962653329, r=0.0006967025276949609), Circle(x=-0.8322754377350393, y=0.554102013467376, r=0.0005660421085616682), Circle(x=-0.8321747928137314, y=-0.5541625610606129, r=0.0004598858995459575), Circle(x=0.7392365419924447, y=-0.673171068294957, r=0.00037363835199224034), Circle(x=0.11294993185685866, y=-0.9934815223719788, r=0.00030356577189539634), Circle(x=0.6334732851191984, y=0.7736354184970712, r=0.00024663468665647476), Circle(x=0.9997223780252019, y=-0.02007338565641066, r=0.00020038052472891446), Circle(x=-0.12473831665444662, y=-0.560616794077429, r=0.00016280092323980904), Circle(x=-0.8326245762235442, y=-0.5538059164646535, r=0.000132269044826539), Circle(x=0.7390719884053104, y=-0.6736009652364928, r=0.00010746315113676814)]
        return lambda p, pk: place_algorithm_8(p, pk, initial_circles=12, initial_guess=initial_guess_find_one if pk == default_pk else initial_guess_find_all)
    else:
        raise ValueError("Invalid algorithm selection. Choose 1, 2, 3, 4, 5, 5.75, 6, 6.5, 7, or 8.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Marco Polo Problem: Geometric Localization Algorithms')
    parser.add_argument('--algorithm', type=float, required=True, choices=[1, 2, 3, 4, 5, 5.75, 6, 6.5, 7, 8],
                       help='Algorithm choice: '
                            '1: Hexagonal Algorithm (Hexagonal) - 7 hexagons with radius n/2, P(n) ≤ 6⌈log n⌉; '
                            '2: Modified Hexagonal Algorithm (Hexagonal) - Quadrant probes then hexagons, P(n) ≤ 5⌈log n⌉; '
                            '3: Progressive Chord-Based Shrinking (Chord-Based) - Monotonic counterclockwise chords, P(n) < 4.08 log n; '
                            '4: Reordered Chord Placement (Chord-Based) - Non-monotonic optimized placement, P(n) < 3.54 log n; '
                            '5: Central + Chords (Higher-Count Monotonic-Path) - Central probe + perimeter chords, P(n) < 3.83 log n; '
                            '5.75: Central + Chords w/ Final Adjustment; '
                            '6: Central + Optimized Chords (Higher-Count Monotonic-Path) - Geometric optimization, P(n) < 3.34 log n; '
                            '6.5: Central + Optimized Chords w/ Final Adjustment; '
                            '7: Darting Non-Monotonic (Darting) - Algorithm 4 + greedy gap filling, P(n) < 2.93 log n; '
                            '8: Differential Evolution Optimization (Darting) - Optimized placement + greedy filling, P(n) < 2.53 log n')
    parser.add_argument('--find-all', action='store_true', help='Use p^((k+1)/2) for radius calculation instead of p^k')
    parser.add_argument('--precision', type=int, default=5, help='Decimal precision for calculations (minimum 1, actual precision is 2x+4)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output to see intermediate steps')

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
    return binary_search(0.5, 1, evaluator, debug=debug)


def calculate_result(p: float, c_multiplier: int) -> float:
    """Calculate final result using p and c_multiplier."""
    return c_multiplier / math.log2(1 / p)


def run_simulation(
    algorithm: float = 7.0,
    find_all: bool = False,
    precision: int = 5,
    debug: bool = False
) -> tuple[float, float, float, list[Circle], float]:
    """
    Run the Marco Polo problem simulation with the specified algorithm.
    
    Args:
        algorithm (float): Algorithm choice (1-8)
        find_all (bool): Whether to use p^((k+1)/2) for radius calculation instead of p^k
        precision (int): Decimal precision for calculations (minimum 1, actual precision is 2x+4)
        debug (bool): Enable debug output to see intermediate steps
    
    Returns:
        tuple[float, float, float, list[Circle], float]: (p value, c value, ct value, list of circles, CPU time)
        where:
        - p: optimal parameter value (ρ₁ for progressive shrinking algorithms)
        - c: efficiency coefficient (probes per ⌈log n⌉)  
        - ct: total distance traveled by the search point
        - circles: list of probe positions and radii
        - CPU time: execution time in seconds
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
    
    if args.algorithm not in [1.0, 2.0]:
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
    else:
        circles = place_algorithm_1() if args.algorithm == 1 else place_algorithm_2()
        p = c = ct = elapsed_time = None


    print(circles)
    
    draw_circles(circles,
        title=f"Algorithm {args.algorithm}" + (" (Find All)" if args.find_all else ""),
        p=p,
        c=c,
        ct=ct,
        cpu_time=elapsed_time)


if __name__ == '__main__':
    main()
