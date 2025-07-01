"""
Geometric data structures and types for the delta searching algorithm.
"""
from dataclasses import dataclass
from typing import NamedTuple, TypedDict
from shapely import Polygon
import shapely
import math


class Circle(NamedTuple):
    x: float
    y: float
    r: float


class Square(NamedTuple):
    x: float
    y: float
    side_length: float


class HorizontalLine(NamedTuple):
    start: float
    end: float


class CirclesPlotKwargs(TypedDict, total=False):
    title: str | None
    p: float
    c: float
    cpu_time: float


UNIT_CIRCLE = Circle(0.0, 0.0, 1.0)


@dataclass
class Precision:
    precision: int = 7
    epsilon: float = 1e-3
    unit_circle_polygon: Polygon = shapely.Point(0.0, 0.0).buffer(1.0)

    def __post_init__(self):
        self.unit_circle_polygon = self.get_circle_polygon(UNIT_CIRCLE)

    def set_precision(self, precision: int) -> None:
        if precision == self.precision:
            return

        self.precision = precision
        self.epsilon = 1 / 10 ** (precision // 2)
        self.unit_circle_polygon = self.get_circle_polygon(UNIT_CIRCLE)

    def get_circle_polygon(self, circle: Circle) -> Polygon:
        quad_segs = min(math.ceil(circle.r * math.pi / 2 / self.epsilon), 2 ** 20)
        return shapely.Point(circle.x, circle.y).buffer(circle.r, quad_segs=quad_segs)


PRECISION = Precision()
