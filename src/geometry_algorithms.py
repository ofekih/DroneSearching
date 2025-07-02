"""
Geometric algorithms for coverage checking and intersection calculations.
"""
from collections import deque
from typing import Generator, Callable
import math
import itertools
import shapely

from .geometry_types import Circle, Square, HorizontalLine, UNIT_CIRCLE, PRECISION


def get_horizontal_line(circle: Circle, y: float) -> HorizontalLine | None:
    """Get the horizontal line that intersects the circle at y."""
    if abs(y - circle.y) > circle.r:
        return None

    delta = (circle.r ** 2 - (y - circle.y) ** 2) ** 0.5
    return HorizontalLine(circle.x - delta, circle.x + delta)


def get_line_union(lines: list[HorizontalLine]) -> list[HorizontalLine]:
    """Merge horizontal lines into a minimal set of non-overlapping lines."""
    if not lines:
        return []

    # Sort by start coordinate
    sorted_lines = sorted(lines, key=lambda line: line.start)
    merged_lines: list[HorizontalLine] = []
    current = sorted_lines[0]

    for line in sorted_lines[1:]:
        if line.start <= current.end:
            # Lines overlap, update end point
            current = HorizontalLine(current.start, max(current.end, line.end))
        else:
            # No overlap, add current line and start new one
            merged_lines.append(current)
            current = line

    merged_lines.append(current)
    return merged_lines


def do_circles_cover_unit_circle(circles: list[Circle], y: float) -> bool:
    lines = [line for circle in circles for line in [get_horizontal_line(circle, y)] if line is not None]
    union = get_line_union(lines)

    unit_circle_line = get_horizontal_line(UNIT_CIRCLE, y)

    if unit_circle_line is None:
        return True
    
    return any(union_line.start <= unit_circle_line.start and union_line.end >= unit_circle_line.end for union_line in union)


def covers_unit_circle_2(circles: list[Circle]) -> bool:
    y = -1.0
    while y < 1:
        if not do_circles_cover_unit_circle(circles, y):
            return False
        y += PRECISION.epsilon

    return True


def is_point_covered(circle: Circle, x: float, y: float) -> bool:
    return (x - circle.x) ** 2 + (y - circle.y) ** 2 <= circle.r ** 2


def is_point_covered_by_any(circles: list[Circle], x: float, y: float) -> bool:
    return any(is_point_covered(circle, x, y) for circle in circles)


def is_fully_covered(square: Square, circle: Circle) -> bool:
    return is_point_covered(circle, square.x, square.y) and \
            is_point_covered(circle, square.x + square.side_length, square.y) and \
            is_point_covered(circle, square.x, square.y + square.side_length) and \
            is_point_covered(circle, square.x + square.side_length, square.y + square.side_length)


def is_square_covered(circles: list[Circle], square: Square) -> bool:
    x, y = square.x, square.y
    side_length = square.side_length
    corners = [(x, y), (x + side_length, y), (x, y + side_length), (x + side_length, y + side_length)]
    
    num_outside_unit_circle = sum(1 for corner in corners if not is_point_covered(UNIT_CIRCLE, *corner))
    
    if num_outside_unit_circle == 4:
        return True
        
    # Check if any point inside unit circle is not covered
    for corner in corners:
        if is_point_covered(UNIT_CIRCLE, *corner) and not is_point_covered_by_any(circles, *corner):
            return False
            
    # Check if square is entirely covered by any circle
    if any(is_fully_covered(square, circle) for circle in circles):
        return True

    # Recurse into four sub-quadrants
    new_side_length = square.side_length / 2

    if new_side_length < PRECISION.epsilon:
        return True

    subsquares = [
        Square(x, y, new_side_length),
        Square(x + new_side_length, y, new_side_length),
        Square(x, y + new_side_length, new_side_length),
        Square(x + new_side_length, y + new_side_length, new_side_length)
    ]
    return all(is_square_covered(circles, subsquare) for subsquare in subsquares)


def get_all_uncovered_squares(circles: list[Circle]) -> Generator[Square, None, None]:
    def get_uncovered_squares(square: Square) -> Generator[Square, None, None]:
        x, y = square.x, square.y
        side_length = square.side_length
        corners = [(x, y), (x + side_length, y), (x, y + side_length), (x + side_length, y + side_length)]

        num_outside_unit_circle = sum(1 for corner in corners if not is_point_covered(UNIT_CIRCLE, *corner))

        if num_outside_unit_circle == 4:
            return
        
        num_uncovered_corners = sum(1 for corner in corners if is_point_covered(UNIT_CIRCLE, *corner) and not is_point_covered_by_any(circles, *corner))

        if num_uncovered_corners > 3:
            yield square
            return

        if num_uncovered_corners == 0 and any(is_fully_covered(square, circle) for circle in circles):
            return

        new_side_length = square.side_length / 2

        if new_side_length < PRECISION.epsilon:
            return

        subsquares = [
            Square(x, y, new_side_length),
            Square(x + new_side_length, y, new_side_length),
            Square(x, y + new_side_length, new_side_length),
            Square(x + new_side_length, y + new_side_length, new_side_length)
        ]
        for subsquare in subsquares:
            yield from get_uncovered_squares(subsquare)

    yield from get_uncovered_squares(Square(-1.0, -1.0, 1.0))
    yield from get_uncovered_squares(Square(-1.0, 0.0, 1.0))
    yield from get_uncovered_squares(Square(0.0, -1.0, 1.0))
    yield from get_uncovered_squares(Square(0.0, 0.0, 1.0))


def get_biggest_uncovered_square(circles: list[Circle]):
    # use BFS instead of DFS, first square found is guaranteed to be the biggest

    q = deque([Square(-1.0, -1.0, 1.0), Square(-1.0, 0.0, 1.0),
               Square(0.0, -1.0, 1.0), Square(0.0, 0.0, 1.0)])
    
    while q:
        square = q.popleft()

        x, y = square.x, square.y
        side_length = square.side_length
        corners = [(x, y), (x + side_length, y), (x, y + side_length), (x + side_length, y + side_length)]

        num_outside_unit_circle = sum(1 for corner in corners if not is_point_covered(UNIT_CIRCLE, *corner))

        if num_outside_unit_circle == 4:
            continue

        num_uncovered_corners = sum(1 for corner in corners if is_point_covered(UNIT_CIRCLE, *corner) and not is_point_covered_by_any(circles, *corner))

        if num_uncovered_corners > 3:
            return square
        
        if num_uncovered_corners == 0 and any(is_fully_covered(square, circle) for circle in circles):
            continue

        new_side_length = square.side_length / 2

        if new_side_length < PRECISION.epsilon:
            continue

        subsquares = [
            Square(x, y, new_side_length),
            Square(x + new_side_length, y, new_side_length),
            Square(x, y + new_side_length, new_side_length),
            Square(x + new_side_length, y + new_side_length, new_side_length)
        ]

        q.extend(subsquares)

    return None


def get_biggest_semicovered_square(circles: list[Circle]):
    # use BFS instead of DFS, first square found is guaranteed to be the biggest

    q = deque([Square(-1.0, -1.0, 1.0), Square(-1.0, 0.0, 1.0),
               Square(0.0, -1.0, 1.0), Square(0.0, 0.0, 1.0)])
    
    while q:
        square = q.popleft()

        x, y = square.x, square.y
        side_length = square.side_length
        corners = [(x, y), (x + side_length, y), (x, y + side_length), (x + side_length, y + side_length)]

        num_outside_unit_circle = sum(1 for corner in corners if not is_point_covered(UNIT_CIRCLE, *corner))

        if num_outside_unit_circle == 4:
            continue

        num_uncovered_corners = sum(1 for corner in corners if is_point_covered(UNIT_CIRCLE, *corner) and not is_point_covered_by_any(circles, *corner))

        if num_uncovered_corners > 0:
            return square
        
        if num_uncovered_corners == 0 and any(is_fully_covered(square, circle) for circle in circles):
            continue

        new_side_length = square.side_length / 2

        if new_side_length < PRECISION.epsilon:
            continue

        subsquares = [
            Square(x, y, new_side_length),
            Square(x + new_side_length, y, new_side_length),
            Square(x, y + new_side_length, new_side_length),
            Square(x + new_side_length, y + new_side_length, new_side_length)
        ]

        q.extend(subsquares)

    return None


def covers_unit_circle(circles: list[Circle]) -> bool:
    # (x, y) are the bottom left coordinates of the square
    return all(is_square_covered(circles, Square(x, y, 1.0)) 
              for x, y in [(-1.0, -1.0), (-1.0, 0.0), (0.0, -1.0), (0.0, 0.0)])


def covers_unit_circle_3(circles: list[Circle]) -> bool:
    circle_polygons = [PRECISION.get_circle_polygon(circle) for circle in circles]
    diff = PRECISION.unit_circle_polygon.difference(shapely.union_all(circle_polygons)) # type: ignore

    return diff.area < PRECISION.epsilon


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


def rotate_circles(circles: list[Circle]):
    # rotate the circles such that the first circle intersects (1, 0)

    first_circle = circles[0]
    intersection_with_unit_circle = get_intersections(first_circle, UNIT_CIRCLE)
    if intersection_with_unit_circle is None:
        return circles
    
    # get the upper intersection point
    x, y = max(intersection_with_unit_circle, key=lambda point: point[1])
    
    # get angle to rotate, angle from x y to origin
    angle = math.atan2(y, x)

    rotated_circles: list[Circle] = []
    for circle in circles:
        x, y = circle.x, circle.y
        r = circle.r

        new_x = x * math.cos(angle) - y * math.sin(angle)
        new_y = x * math.sin(angle) + y * math.cos(angle)

        rotated_circles.append(Circle(new_x, -new_y, r))
    
    return rotated_circles


# Coverage analysis utilities

PkFunction = Callable[[float, int], float]


def get_empty_area(circles: list[Circle]) -> float:
    """Calculate the uncovered area within the unit circle."""
    circle_polygons = [PRECISION.get_circle_polygon(circle) for circle in circles]
    uncovered_polygons = PRECISION.unit_circle_polygon.difference(shapely.union_all(circle_polygons)) # type: ignore
    return uncovered_polygons.area


def get_circle_centers(x1: float, y1: float, x2: float, y2: float, r: float):
    """Get possible circle centers for a circle of radius r passing through two points."""
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
    """Find optimal circle placement to minimize uncovered area."""
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


def add_centroid_circles(p: float, pk: PkFunction, circles: list[Circle]) -> tuple[list[Circle], float]:
    """Add circles at centroids of uncovered regions."""
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
        
        largest_geom = max(uncovered_polygons.geoms, key=lambda g: g.area) if hasattr(uncovered_polygons, 'geoms') else uncovered_polygons # type: ignore
        centroid = largest_geom.centroid # type: ignore
        new_circle = Circle(centroid.x, centroid.y, current_radius) # type: ignore
        circles.append(new_circle)
        k += 1

        uncovered_polygons = uncovered_polygons.difference(PRECISION.get_circle_polygon(new_circle)) # type: ignore

    return circles, uncovered_polygons.area


def add_intelligent_circles(p: float, pk: PkFunction, circles: list[Circle]) -> tuple[list[Circle], float]:
    """Add circles using intelligent placement to minimize uncovered area."""
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
        
        largest_geom = max(uncovered_polygons.geoms, key=lambda g: g.area) if hasattr(uncovered_polygons, 'geoms') else uncovered_polygons # type: ignore
        
        new_circle: Circle = intelligently_minimize(largest_geom, current_radius) # type: ignore
        circles.append(new_circle)
        k += 1

        uncovered_polygons = uncovered_polygons.difference(PRECISION.get_circle_polygon(new_circle)) # type: ignore

    return circles, uncovered_polygons.area
