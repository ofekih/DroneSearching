"""
Utility functions for the delta searching algorithm including binary search and distance calculations.
"""
from typing import Callable
import math

from geometry_types import Circle, PRECISION


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
