"""
Circle placement algorithms for the drone searching problem.
"""
import math
from typing import Callable

from .geometry_types import Circle, PRECISION
from .geometry_algorithms import get_intersections


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

    return sorted(circles, key=lambda c: c.r, reverse=True)


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
