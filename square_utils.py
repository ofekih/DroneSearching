from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Generator, Iterable

CoordinateType = int | float

@dataclass(frozen=True)
class Point:
	"""
	Represents a point in d-dimensional space.

	Attributes:
		coordinates: A tuple containing the coordinates of the point.
					 The length of the tuple determines the dimension 'd'.
					 Allows d >= 0.
	"""
	coordinates: tuple[CoordinateType, ...]

	def __post_init__(self):
		"""Validates the coordinates."""
		# No validation needed for empty tuple (0-d point)
		pass

	@property
	def dimension(self) -> int:
		"""Returns the dimension 'd' of the point."""
		return len(self.coordinates)

	def __getitem__(self, index: int) -> CoordinateType:
		"""Allows accessing coordinates by index."""
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
		if 0 <= dim < self.dimension:
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

	def num_shared_coordinates(self, other: Point) -> int:
		"""
		Counts the number of coordinates that match between this point and another point.

		Args:
			other: The other Point object. Assumes it has the same dimension.

		Returns:
			The number of matching coordinates.
		"""
		return sum(1 for c1, c2 in zip(self.coordinates, other.coordinates) if math.isclose(c1, c2))

	def shares_any_coordinate(self, other: Point) -> bool:
		"""
		Checks if this point shares any coordinate with another point.

		Args:
			other: The other Point object. Assumes it has the same dimension.

		Returns:
			True if any coordinate matches, False otherwise.
		"""
		return self.num_shared_coordinates(other) > 0

	def __eq__(self, other: object) -> bool:
		"""Compares two Points for equality."""
		if not isinstance(other, Point):
			return False
		
		return all(math.isclose(c1, c2) for c1, c2 in zip(self.coordinates, other.coordinates))

	@staticmethod
	def origin(dimensions: int) -> Point:
		return Point(tuple(0 for _ in range(dimensions)))

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
		if self.dimension < 0:
			raise ValueError("Hypercube dimension must be non-negative.")

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
		if self.dimension == 0:
			return self.center
		half_side = self.side_length / 2.0
		min_coords = tuple(c - half_side for c in self.center.coordinates)
		return Point(coordinates=min_coords)

	@property
	def max_corner(self) -> Point:
		"""Calculates the corner with the maximum coordinates."""
		if self.dimension == 0:
			return self.center
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
		if self.dimension == 0:
			return point == self.center
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
		if self.dimension == 0:
			return other == self
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
		if self.dimension == 0:
			return self
		if len(code) != self.dimension:
			raise ValueError("Code length must match the dimension of the hypercube.")
		
		new_center = Point(tuple(self.center.coordinates[i] + self.side_length / 4 * (2 * code[i] - 1) for i in range(self.dimension)))
		return Hypercube(new_center, self.side_length / 2)

	def offset(self, offset: CoordinateType, dim: int) -> Hypercube:
		"""
		Returns a new Hypercube with the specified offset applied to the given dimension.
		"""
		return Hypercube(self.center.offset(offset, dim), self.side_length)

	@property
	def orthants(self) -> Generator[Hypercube, None, None]:
		"""
		Generates all orthants of the hypercube.
		"""
		if self.dimension == 0:
			yield self
		else:
			for code in generate_gray_codes(self.dimension):
				yield self.orthant_from_code(code)

	@property
	def neighbors(self) -> Generator[Hypercube, None, None]:
		"""
		Generates all neighbors of the hypercube.
		"""
		if self.dimension == 0:
			return
		for dim in range(self.dimension):
			for offset in (-1, 1):
				new_center = self.center.offset(offset * self.side_length, dim)
				yield Hypercube(new_center, self.side_length)

	def __eq__(self, other: object) -> bool:
		"""Compares two Hypercubes for equality."""
		if not isinstance(other, Hypercube):
			return False
		return self.center == other.center and math.isclose(self.side_length, other.side_length)

class ProjectionManager:
	"""
	Manages coordinate transformations from a projected lower-dimensional space
	back to an original higher-dimensional space as dimensions are fixed.

	Allows creating points and hypercubes using coordinates relative to the
	current projection dimension, and automatically maps them back to the
	original d-dimensional space.
	"""
	def __init__(self, original_dimension: int):
		"""
		Initializes the ProjectionManager.

		Args:
			original_dimension: The starting number of dimensions (d).
		"""
		if original_dimension < 0:
			raise ValueError("original_dimension must be a non-negative integer.")

		self._original_dimension: int = original_dimension
		# Stores fixed dimensions: {original_index: fixed_value}
		self._fixed_dimensions: dict[int, float] = {}
		# Stores the indices of dimensions that are still variable
		self._update_variable_dimensions()

	def _update_variable_dimensions(self):
		"""Helper method to recalculate the variable dimension indices."""
		fixed_indices = set(self._fixed_dimensions.keys())
		self._variable_dimension_indices: list[int] = sorted([
			i for i in range(self._original_dimension) if i not in fixed_indices
		])

	@property
	def original_dimension(self) -> int:
		"""Returns the initial dimension d."""
		return self._original_dimension

	@property
	def current_dimension(self) -> int:
		"""Returns the dimension of the current projection space."""
		return len(self._variable_dimension_indices)

	@property
	def fixed_dimensions(self) -> dict[int, float]:
		"""Returns a copy of the fixed dimensions map."""
		return self._fixed_dimensions.copy()

	@property
	def variable_dimension_indices(self) -> list[int]:
		"""Returns a copy of the list of variable dimension indices."""
		return self._variable_dimension_indices.copy()

	def fix_coordinate(self, dim_index: int, value: CoordinateType):
		"""
		Fixes a dimension to a specific value, reducing the current projection dimension by one.

		Args:
			dim_index: The index of the variable dimension to fix (0-indexed in the current projection space).
			value: The value to fix the coordinate at.

		Raises:
			ValueError: If the dimension index is invalid.
		"""
		if not (0 <= dim_index < self.current_dimension):
			raise ValueError(f"dim_index {dim_index} is out of bounds for current dimension {self.current_dimension}.")

		original_dim_index = self._variable_dimension_indices[dim_index]
		self._fixed_dimensions[original_dim_index] = float(value)
		self._update_variable_dimensions()  # Update the list of variable dims

	def fix_coordinates(self, values: Iterable[tuple[int, CoordinateType]]):
		"""
		Fixes multiple dimensions to specific values, reducing the current projection dimension by the number of fixed dimensions.

		Args:
			values: An iterable of tuples, where each tuple contains the index of the variable dimension to fix
					and the value to fix it at.

		Raises:
			ValueError: If any dimension index is invalid.
		"""
		for dim_index, value in sorted(values, key=lambda x: x[0], reverse=True):
			self.fix_coordinate(dim_index, value)

	def Point(self, p: Point) -> Point:
		"""
		Creates a Point in the original d-dimensional space from coordinates
		provided in the current projection space.

		Args:
			p: A Point object in the *current* projection dimension.

		Returns:
			A Point object with self.original_dimension coordinates.

		Raises:
			ValueError: If the number of provided coordinates does not match
						the current projection dimension.
		"""
		coords = p.coordinates

		if len(coords) != self.current_dimension:
			raise ValueError(f"Expected {self.current_dimension} coordinates for the current projection "
							 f"dimension, but received {len(coords)}.")

		# Initialize coordinates for the original d-dimensional space
		origin_coords = [0.0] * self._original_dimension

		# Fill in fixed coordinates
		for index, value in self._fixed_dimensions.items():
			origin_coords[index] = value

		# Fill in variable coordinates from input *coords
		for i, var_coord in enumerate(coords):
			original_index = self._variable_dimension_indices[i]
			origin_coords[original_index] = float(var_coord)

		return Point(tuple(origin_coords))

	def Hypercube(self, center: Point, side_length: CoordinateType) -> Hypercube:
		"""
		Creates a Hypercube in the original d-dimensional space from a center point
		defined in the current projection space and a side length.

		The center of the resulting d-dimensional hypercube will have coordinates matching
		the input center for the variable dimensions and the fixed values for the
		fixed dimensions.

		Args:
			center: The center point of the hypercube in the *current* projection space.
					Its dimension must match self.current_dimension.
			side_length: The side length of the hypercube.

		Returns:
			A Hypercube object in the original d-dimensional space.

		Raises:
			ValueError: If the dimension of center does not match the current
						projection dimension.
			ValueError: If side_length is negative.
		"""
		# Create the Hypercube in the original dimension
		return Hypercube(center=self.Point(center), side_length=side_length)
	
	def InsetHypercube(self, center: Point, side_length: CoordinateType) -> Hypercube:
		centered_hypercube = self.Hypercube(center, side_length)

		for dim in self._fixed_dimensions:
			centered_hypercube = centered_hypercube.offset(-math.copysign(side_length / 2 - 0.5, centered_hypercube.center[dim]), dim)


		return centered_hypercube

	def __repr__(self) -> str:
		"""Returns a string representation of the ProjectionManager's state."""
		fixed_str = ", ".join(f"{k}={v:.4f}".rstrip('0').rstrip('.')
							  for k, v in sorted(self._fixed_dimensions.items()))
		return (f"ProjectionManager(orig_dim={self._original_dimension}, "
				f"current_dim={self.current_dimension}, "
				f"fixed={{{fixed_str}}})")

HypercubeGetter = Callable[[ProjectionManager, Point, CoordinateType], Hypercube]
