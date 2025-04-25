from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Generator

type CoordinateType = int | float

@dataclass(frozen=True)
class Point:
	"""
	Represents a point in d-dimensional space.

	Attributes:
		coordinates: A tuple containing the coordinates of the point.
					 The length of the tuple determines the dimension 'd'.
					 Assumes d >= 1.
	"""
	coordinates: tuple[CoordinateType, ...]

	def __post_init__(self):
		"""Validates that at least one coordinate is provided."""
		# Keep this check as it ensures the fundamental structure (d>=1)
		if not self.coordinates:
			raise ValueError("Point must have at least one coordinate (d >= 1).")

	@property
	def dimension(self) -> int:
		"""Returns the dimension 'd' of the point."""
		return len(self.coordinates)

	def __getitem__(self, index: int) -> CoordinateType:
		"""Allows accessing coordinates by index."""
		# Keep bounds check for valid indexing
		if not 0 <= index < self.dimension:
			raise IndexError(f"Index {index} out of bounds for dimension {self.dimension}.")
		return self.coordinates[index]

	def __len__(self) -> int:
		"""Returns the dimension 'd' of the point."""
		return self.dimension

	def __repr__(self) -> str:
		"""Returns a string representation of the point."""
		coord_str = ', '.join(map(str, self.coordinates))
		return f"Point({coord_str})"

	def distance_to(self, other: Point) -> CoordinateType:
		"""
		Calculates the Manhattan (L1) distance to another point.

		Args:
			other: The other Point object. Assumes it has the same dimension.

		Returns:
			The L1 distance between the two points.
		"""
		return sum(abs(c1 - c2) for c1, c2 in zip(self.coordinates, other.coordinates))

	def offset(self, offset: CoordinateType, dim: int) -> Point:
		"""
		Returns a new Point with the specified offset applied to the given dimension.
		"""
		new_coords = list(self.coordinates)
		new_coords[dim] += offset
		return Point(tuple(new_coords))

	def interpolate(self, other: Point, t: float) -> Point:
		"""
		Interpolates between this point and another point.

		Args:
			other: The other Point object. Assumes it has the same dimension.
			t: The interpolation parameter, between 0 and 1.

		Returns:
			A new Point representing the interpolated position.
		"""
		if not 0 <= t <= 1:
			raise ValueError("Interpolation parameter t must be between 0 and 1.")
		new_coords = tuple((1 - t) * c1 + t * c2 for c1, c2 in zip(self.coordinates, other.coordinates))
		return Point(new_coords)

	def shares_any_coordinate(self, other: Point) -> bool:
		"""
		Checks if this point shares any coordinate with another point.

		Args:
			other: The other Point object. Assumes it has the same dimension.

		Returns:
			True if any coordinate matches, False otherwise.
		"""
		return any(c1 == c2 for c1, c2 in zip(self.coordinates, other.coordinates))

def generate_gray_codes(n: int) -> Generator[tuple[int, ...], None, None]:
    """
    Generates binary tuples (using 0s and 1s) of length 'n'
    following the standard reflected binary Gray code sequence.

    Example n=3: (0,0,0), (0,0,1), (0,1,1), (0,1,0), (1,1,0), (1,1,1), (1,0,1), (1,0,0)
    """
    if n <= 0:
        # Base case for recursion or invalid input
        if n == 0:
             yield () # Yield empty tuple for 0 dimension
        return

    if n == 1:
        yield (0,)
        yield (1,)
        return

    # Recursively generate Gray code for n-1 bits
    # We need to store the result to iterate over it twice
    prev_gray_codes = list(generate_gray_codes(n - 1))

    # Yield first half: prepend 0 to G(n-1)
    for code in prev_gray_codes:
        yield (0,) + code

    # Yield second half: prepend 1 to reversed G(n-1)
    for code in reversed(prev_gray_codes):
        yield (1,) + code

@dataclass(frozen=True)
class Hypercube:
	"""
	Represents an axis-aligned hypercube with equal side lengths
	in d-dimensional space.

	Defined by its center point and the length of its sides.

	Attributes:
		center: The Point representing the center of the hypercube. Coordinates
				are float to allow for centers between integer coordinates.
		side_length: The length of each side of the hypercube (must be non-negative).
	"""
	center: Point # Center coordinates are float
	side_length: CoordinateType

	def __post_init__(self):
		"""Validates the hypercube definition."""
		# Dimension check is implicitly handled by Point validation for center
		if self.dimension < 1:
			raise ValueError("Hypercube dimension must be at least 1.")

		# Validate side length
		if self.side_length < 0:
			raise ValueError(f"side_length cannot be negative ({self.side_length}).")

	@property
	def dimension(self) -> int:
		"""Returns the dimension 'd' of the hypercube."""
		return self.center.dimension

	@property
	def min_corner(self) -> Point:
		"""Calculates the corner with the minimum coordinates."""
		half_side = self.side_length / 2.0
		min_coords = tuple(c - half_side for c in self.center.coordinates)
		return Point(coordinates=min_coords)

	@property
	def max_corner(self) -> Point:
		"""Calculates the corner with the maximum coordinates."""
		half_side = self.side_length / 2.0
		max_coords = tuple(c + half_side for c in self.center.coordinates)
		return Point(coordinates=max_coords)

	@property
	def volume(self) -> float:
		"""Calculates the volume (or area, or length) of the hypercube."""
		# Use math.pow for potential floating point precision with large dimensions
		return math.pow(self.side_length, self.dimension)

	def contains_point(self, point: Point) -> bool:
		"""
		Checks if a given point lies inside or on the boundary of the hypercube.

		Args:
			point: The Point to check. Assumes it has the same dimension as the hypercube.
				   Can have int or float coordinates.

		Returns:
			True if the point is contained within the hypercube, False otherwise.
		"""
		half_side = self.side_length / 2
		for i in range(self.dimension):
			min_bound = self.center.coordinates[i] - half_side
			max_bound = self.center.coordinates[i] + half_side
			# Compare point coordinate (int or float) with float bounds
			if not (min_bound <= point.coordinates[i] <= max_bound):
				return False
		return True
	
	def contains_hypercube(self, other: Hypercube) -> bool:
		"""
		Checks if a given hypercube is completely contained within this hypercube.

		Args:
			other: The Hypercube to check. Assumes it has the same dimension as this hypercube.

		Returns:
			True if the other hypercube is contained within this hypercube, False otherwise.
		"""
		return self.contains_point(other.min_corner) and self.contains_point(other.max_corner)
	
	def __contains__(self, item: Point | Hypercube) -> bool:
		"""Allows using the 'in' operator to check if a point or hypercube is inside this hypercube."""
		if isinstance(item, Point):
			return self.contains_point(item)
		
		return self.contains_hypercube(item)

	def __repr__(self) -> str:
		"""Returns a string representation of the hypercube."""
		return f"Hypercube(center={self.center!r}, side_length={self.side_length})"

	def orthant_from_code(self, code: tuple[int, ...]) -> Hypercube:
		"""
		Returns a new Hypercube representing the orthant defined by the given code.
		The code is a tuple of 0s and 1s, where each element specifies which half of the hypercube to take.
		"""
		if len(code) != self.dimension:
			raise ValueError("Code length must match the dimension of the hypercube.")
		
		new_center = Point(tuple(self.center.coordinates[i] + self.side_length / 4 * (2 * code[i] - 1) for i in range(self.dimension)))
		return Hypercube(new_center, self.side_length / 2)

	@property
	def orthants(self) -> Generator[Hypercube, None, None]:
		"""
		Generates all orthants of the hypercube.
		"""
		for code in generate_gray_codes(self.dimension):
			yield self.orthant_from_code(code)