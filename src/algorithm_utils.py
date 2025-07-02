"""
Utility functions for the drone searching algorithm including binary search, distance calculations, and optimization.
"""
import math
from typing import Callable, Optional, Any
from dataclasses import dataclass

from scipy import optimize
from scipy.optimize._optimize import OptimizeResult

from .geometry_types import Circle, PRECISION


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


def get_distance_traveled(circles: list[Circle], debug: bool = False):
    # D(n) = max(dist to get to kth circle + D(r_k * n))
    # max(d_k / (1 - r_k))

    # circles.sort(key=lambda circle: circle.r, reverse=True)
    distance = 0
    current_point = (0, 0)

    max_ct = 0

    for circle in circles:
        x, y, r = circle
        distance_to_circle: float = math.sqrt((x - current_point[0]) ** 2 + (y - current_point[1]) ** 2)
        
        if circle == circles[-1]:
            # Don't need to necessarily travel to center of the last circle, since we are guaranteed that it is there.
            # Only need to travel to first probe point of the next layer

            # -1 * sqrt(x^2 + y^2) * r is for getting to the first probe of the next guy.
            # the second sqrt(x^2 + y^2) * r is for the next layer not needing to
            # traverse that distance to get to its first probe
            distance_to_circle -= 2 * math.sqrt(circles[0].x**2 + circles[0].y**2) * r

        distance += distance_to_circle

        ct = distance / (1 - r)

        if debug:
            print(f"Circle {circles.index(circle) + 1}: {distance}, {r} => {ct}") 

        max_ct = max(max_ct, ct)
        current_point = (x, y)

    return max_ct


# Optimization utilities

PkFunction = Callable[[float, int], float]


@dataclass
class ObjectiveFunctionWrapper:
    """A pickleable wrapper class for the objective function."""
    p: float
    k: int
    pk: PkFunction
    
    def __call__(self, x: list[float]) -> float:
        return objective_function_global(x, self.p, self.k, self.pk)


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
    from .geometry_algorithms import add_intelligent_circles
    circles = create_circles_from_params(x, k, p, pk)
    _, remaining_area = add_intelligent_circles(p, pk, circles)
    return remaining_area


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
