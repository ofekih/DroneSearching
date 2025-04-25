from dataclasses import dataclass
import math

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

	def distance_to(self, other: 'Point') -> CoordinateType:
		"""
		Calculates the Manhattan (L1) distance to another point.

		Args:
			other: The other Point object. Assumes it has the same dimension.

		Returns:
			The L1 distance between the two points.
		"""
		return sum(abs(c1 - c2) for c1, c2 in zip(self.coordinates, other.coordinates))

	def offset(self, offset: CoordinateType, dim: int) -> 'Point':
		"""
		Returns a new Point with the specified offset applied to the given dimension.
		"""
		new_coords = list(self.coordinates)
		new_coords[dim] += offset
		return Point(tuple(new_coords))


# --- Updated Hypercube Class ---

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

	def contains(self, point: Point) -> bool:
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

	def __repr__(self) -> str:
		"""Returns a string representation of the hypercube."""
		return f"Hypercube(center={self.center!r}, side_length={self.side_length})"
