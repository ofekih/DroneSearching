import itertools
import random
from typing import Callable, Literal, NamedTuple

from square_utils import HypercubeGetter, Point, Hypercube, ProjectionManager, CoordinateType

class SimulationResult(NamedTuple):
	P: int # total number of probes
	D: float # total distance traveled
	num_responses: int # number of hiker responses
	area: Hypercube

def get_random_hiker_position(search_area: Hypercube):
	return Point(tuple(random.uniform(search_area.center.coordinates[i] - search_area.side_length / 2, search_area.center.coordinates[i] + search_area.side_length / 2) for i in range(search_area.dimension)))

def get_random_hiker_position_non_equal(search_area: Hypercube):
	# don't allow any two coordinates to be within 1 of each other
	# loop one coordinate at a time
	coordinates: list[float] = []
	for i in range(search_area.dimension):
		while True:
			coordinate = random.uniform(search_area.center.coordinates[i] - search_area.side_length / 2, search_area.center.coordinates[i] + search_area.side_length / 2)
			if all(abs(abs(coordinate) - abs(c)) > 1 for c in coordinates):
				coordinates.append(coordinate)
				break
	return Point(tuple(coordinates))


KidnapperAlgorithm = Callable[[Hypercube, Point, Point], SimulationResult]
HikerGenerator = Callable[[Hypercube], Point]

def simple_hypercube_search(search_area: Hypercube, hiker: Point, drone: Point) -> SimulationResult:
	if search_area.side_length <= 1:
		return SimulationResult(0, drone.distance_to(search_area.center), 0, search_area)

	distance_traveled = 0
	num_probes = 0
	num_responses = 0

	orthant_iter = search_area.orthants
	correct_orthant = next(orthant_iter)

	for probe in orthant_iter:
		distance_traveled += drone.distance_to(probe.center)
		drone = probe.center

		num_probes += 1
		if hiker in probe:
			num_responses += 1
			correct_orthant = probe
			break

	result = simple_hypercube_search(correct_orthant, hiker, drone)

	return SimulationResult(num_probes + result.P, distance_traveled + result.D, num_responses + result.num_responses, result.area)

def domino_2d_search(search_area: Hypercube, hiker: Point, drone: Point) -> SimulationResult:
	if search_area.dimension != 2:
		raise ValueError("Domino 2D search only works in 2 dimensions.")

	if search_area.side_length <= 1:
		return SimulationResult(0, drone.distance_to(search_area.center), 0, search_area)

	def domino_2d_reduction(area: Hypercube, empty_adjacent: Hypercube) -> SimulationResult:
		nonlocal drone

		if area.side_length <= 1:
			return SimulationResult(0, drone.distance_to(area.center), 0, area)

		candidates = list(area.orthants)

		# move drone to halfway between the two areas
		center = area.center.interpolate(empty_adjacent.center, 0.5)
		distance_traveled = drone.distance_to(center)
		drone = center

		# probe the center
		num_responses = 0
		probe = Hypercube(center, area.side_length)
		if hiker in probe:
			num_responses += 1
			empty_adjacent_candidates = [o for o in empty_adjacent.orthants if o in probe]
			candidates = [c for c in candidates if c in probe]
		else:
			empty_adjacent_candidates = [c for c in candidates if c in probe]
			candidates = [c for c in candidates if c not in probe]

		probe = candidates[0]
		distance_traveled += drone.distance_to(probe.center)
		drone = probe.center
		if hiker in probe:
			num_responses += 1
			candidate = candidates[0]
		else:
			candidate = candidates[1]

		# correct empty adjacent candidate should have one dimension in common
		empty_adjacent_candidate = empty_adjacent_candidates[0] if empty_adjacent_candidates[0].center.shares_any_coordinate(candidate.center) else empty_adjacent_candidates[1]

		result = domino_2d_reduction(candidate, empty_adjacent_candidate)

		return SimulationResult(2 + result.P, distance_traveled + result.D, num_responses + result.num_responses, result.area)

	num_probes = 0
	distance_traveled = 0
	num_responses = 0

	orthant_iter = search_area.orthants
	correct_orthant = next(orthant_iter)
	empty_adjacent_orthant = None
	for probe in orthant_iter:
		distance_traveled += drone.distance_to(probe.center)
		drone = probe.center

		num_probes += 1
		if hiker in probe:
			num_responses += 1
			correct_orthant = probe
			break

		empty_adjacent_orthant = probe

	if empty_adjacent_orthant:
		result = domino_2d_reduction(correct_orthant, empty_adjacent_orthant)
	else:
		result = domino_2d_search(correct_orthant, hiker, drone)

	return SimulationResult(num_probes + result.P, distance_traveled + result.D, num_responses + result.num_responses, result.area)

def domino_3d_search(search_area: Hypercube, hiker: Point, drone: Point) -> SimulationResult:
	if search_area.dimension != 3:
		raise ValueError("Domino 3D search only works in 3 dimensions.")

	if search_area.side_length <= 1:
		return SimulationResult(0, drone.distance_to(search_area.center), 0, search_area)

	def domino_3d_reduction(area: Hypercube, empty_adjacent_3: tuple[Hypercube, Hypercube, Hypercube]) -> SimulationResult:
		nonlocal drone

		if area.side_length <= 1:
			return SimulationResult(0, drone.distance_to(area.center), 0, area)

		candidates = list(area.orthants)
		known_empty = list(empty_adjacent_3)

		directly_adjacent = [o for o in empty_adjacent_3 if o.center.shares_any_coordinate(area.center)]
		centers = [area.center.interpolate(o.center, 0.5) for o in directly_adjacent]
		closest_center = min(centers, key=lambda c: c.distance_to(drone))

		probe = Hypercube(closest_center, area.side_length)
		distance_traveled = drone.distance_to(probe.center)
		drone = probe.center

		num_responses = 0
		if hiker in probe:
			num_responses += 1
			candidates = [c for c in candidates if c in probe]
		else:
			candidates = [c for c in candidates if c not in probe]
			known_empty.append(probe)

		other_center = max(centers, key=lambda c: c.distance_to(drone))
		probe = Hypercube(other_center, area.side_length)
		distance_traveled += drone.distance_to(probe.center)
		drone = probe.center

		if hiker in probe:
			num_responses += 1
			candidates = [c for c in candidates if c in probe]
		else:
			candidates = [c for c in candidates if c not in probe]
			known_empty.append(probe)

		# only two candidates left
		probe = candidates[0]
		distance_traveled += drone.distance_to(probe.center)
		drone = probe.center

		if hiker in probe:
			num_responses += 1
			candidate = candidates[0]
		else:
			candidate = candidates[1]

		empty_adjacent_candidates = [n for n in candidate.neighbors if any(n in e for e in known_empty)]
		# should have two.
		final_empty_candidate = [n1 for n1 in empty_adjacent_candidates[0].neighbors if any(n1 in e for e in known_empty) and n1 in empty_adjacent_candidates[1].neighbors][0]

		result = domino_3d_reduction(candidate, (final_empty_candidate, empty_adjacent_candidates[0], empty_adjacent_candidates[1]))

		return SimulationResult(3 + result.P, distance_traveled + result.D, num_responses + result.num_responses, result.area)

	def domino_2d_reduction(area: Hypercube, empty_adjacent: Hypercube) -> SimulationResult:
		nonlocal drone

		if area.side_length <= 1:
			return SimulationResult(0, drone.distance_to(area.center), 0, area)

		candidates = list(area.orthants)
		num_responses = 0

		# move drone to halfway between the two areas
		center = area.center.interpolate(empty_adjacent.center, 0.5)
		distance_traveled = drone.distance_to(center)
		drone = center

		probe = Hypercube(center, area.side_length)
		if hiker in probe:
			num_responses += 1
			empty_adjacent_candidates = [o for o in empty_adjacent.orthants if o in probe]
			candidates = [c for c in candidates if c in probe]
		else:
			empty_adjacent_candidates = [c for c in candidates if c in probe]
			candidates = [c for c in candidates if c not in probe]

		probe = candidates[0]
		distance_traveled += drone.distance_to(probe.center)
		drone = probe.center

		if hiker in probe:
			num_responses += 1
			# recurse to 2d reduction
			empty_adjacent_candidate = [e for e in empty_adjacent_candidates if e in probe.neighbors][0]
			result = domino_2d_reduction(probe, empty_adjacent_candidate)
			return SimulationResult(2 + result.P, distance_traveled + result.D, num_responses + result.num_responses, result.area)

		new_empty_candidates = [probe]

		# find candidate which does not share a coordinate
		correct_candidate = [c for c in candidates if not c in probe.neighbors][0]

		for probe in candidates[1:]:
			if probe == correct_candidate:
				continue

			distance_traveled += drone.distance_to(probe.center)
			drone = probe.center

			if hiker in probe:
				num_responses += 1
				correct_candidate = probe
				break

			new_empty_candidates.append(probe)

		new_empty_candidate = [e for e in new_empty_candidates if e in correct_candidate.neighbors][0]
		old_empty_1 = [e for e in empty_adjacent_candidates if e in correct_candidate.neighbors][0]
		old_empty_2 = [e for e in empty_adjacent_candidates if e in new_empty_candidate.neighbors][0]

		result = domino_3d_reduction(correct_candidate, (old_empty_1, old_empty_2, new_empty_candidate))

		return SimulationResult(3 + result.P, distance_traveled + result.D, num_responses + result.num_responses, result.area)

	num_probes = 0
	distance_traveled = 0
	num_responses = 0

	all_orthants = list(search_area.orthants)
	orthant_ordering = [all_orthants.pop(0)]
	orthant_ordering += [n for n in orthant_ordering[0].neighbors if n in all_orthants]
	orthant_ordering += [o for o in all_orthants if o not in orthant_ordering]
	correct_orthant = orthant_ordering.pop()

	empty_orthants: list[Hypercube] = []
	for probe in orthant_ordering:
		distance_traveled += drone.distance_to(probe.center)
		drone = probe.center

		num_probes += 1
		if hiker in probe:
			num_responses += 1
			correct_orthant = probe
			break

		empty_orthants.append(probe)

	if len(empty_orthants) == 0:
		result = domino_3d_search(correct_orthant, hiker, drone)
	else:
		adj_empty_orthants = [e for e in empty_orthants if e in correct_orthant.neighbors]
		final_empty_orthant = [] if len(adj_empty_orthants) == 1 else [e for e in adj_empty_orthants[0].neighbors if e in adj_empty_orthants[1].neighbors and e in empty_orthants]

		if len(final_empty_orthant) > 0:
			result = domino_3d_reduction(correct_orthant, (final_empty_orthant[0], adj_empty_orthants[0], adj_empty_orthants[1]))
		else:
			result = domino_2d_reduction(correct_orthant, adj_empty_orthants[0])

	return SimulationResult(num_probes + result.P, distance_traveled + result.D, num_responses + result.num_responses, result.area)

def midpoint_1d(a: CoordinateType, b: CoordinateType, _dim: int) -> CoordinateType:
	return (a + b) / 2

def midpoint_volume(a: CoordinateType, b: CoordinateType, dim: int) -> CoordinateType:
	return ((a ** dim + b ** dim) / 2) ** (1 / dim)

def naive_central_binary_search(search_area: Hypercube, hiker: Point, drone: Point, hypercube_getter: HypercubeGetter = ProjectionManager.InsetHypercube, radius_calculator: Callable[[CoordinateType, CoordinateType, int], CoordinateType] = midpoint_1d) -> SimulationResult:
	# No optimizations!
	# step 0, check if we are done
	if search_area.side_length <= 1:
		return SimulationResult(0, drone.distance_to(search_area.center), 0, search_area)

	pm = ProjectionManager(search_area.dimension)
	current_radius = search_area.side_length / 2

	distance_traveled = 0
	num_probes = 0
	num_responses = 0

	for dims in range(search_area.dimension, 0, -1):
		new_search_area = Hypercube(Point.origin(dims), current_radius * 2)

		if new_search_area.side_length <= 1:
			true_area = hypercube_getter(pm, new_search_area.center, new_search_area.side_length)
			return SimulationResult(num_probes, distance_traveled + drone.distance_to(true_area.center), num_responses, true_area)

		# Step 1: Binary search in this dimension

		min_radius = 0
		max_radius = new_search_area.side_length / 2
		empty_regions: list[Hypercube] = []
		while min_radius + 1 < max_radius:
			radius = radius_calculator(min_radius, max_radius, dims)
			probe = hypercube_getter(pm, new_search_area.center, radius * 2)

			num_probes += 1
			distance_traveled += drone.distance_to(probe.center)
			drone = probe.center
			if hiker in probe:
				num_responses += 1
				max_radius = radius
			else:
				min_radius = radius
				empty_regions.append(probe)

		current_radius = (min_radius + max_radius) / 2
		offset_amount = min(current_radius, 1)
		side_length = 2 * max(max_radius - offset_amount, 0.5)

		# Step 2: Figure out which face the hiker is on
		for dim in range(dims):
			probe = hypercube_getter(pm, new_search_area.center.offset(offset_amount, dim), side_length)

			num_probes += 1
			distance_traveled += drone.distance_to(probe.center)
			drone = probe.center

			if hiker in probe:
				num_responses += 1
				pm.fix_coordinate(dim, current_radius)
				break

			# if last dimension, no need to check other side
			if dim == dims - 1:
				pm.fix_coordinate(dim, -current_radius)
				break

			probe = hypercube_getter(pm, new_search_area.center.offset(-offset_amount, dim), side_length)

			num_probes += 1
			distance_traveled += drone.distance_to(probe.center)
			drone = probe.center
			if hiker in probe:
				num_responses += 1
				pm.fix_coordinate(dim, -current_radius)
				break

	# Create the final hypercube
	guess = pm.Hypercube(Point.origin(0), 1)

	return SimulationResult(num_probes, distance_traveled + drone.distance_to(guess.center), num_responses, guess)

def naive_central_binary_search_just_one(search_area: Hypercube, hiker: Point, drone: Point) -> SimulationResult:
	return naive_central_binary_search(search_area, hiker, drone, ProjectionManager.Hypercube)

def naive_central_binary_search_volume(search_area: Hypercube, hiker: Point, drone: Point) -> SimulationResult:
	return naive_central_binary_search(search_area, hiker, drone, ProjectionManager.InsetHypercube, radius_calculator=midpoint_volume)

def naive_central_binary_search_volume_just_one(search_area: Hypercube, hiker: Point, drone: Point) -> SimulationResult:
	return naive_central_binary_search(search_area, hiker, drone, ProjectionManager.Hypercube, radius_calculator=midpoint_volume)

def central_binary_search(search_area: Hypercube, hiker: Point, drone: Point, hypercube_getter: HypercubeGetter = ProjectionManager.InsetHypercube, radius_calculator: Callable[[CoordinateType, CoordinateType, int], CoordinateType] = midpoint_1d) -> SimulationResult:
	# No optimizations!
	# step 0, check if we are done
	if search_area.side_length <= 1:
		return SimulationResult(0, drone.distance_to(search_area.center), 0, search_area)

	pm = ProjectionManager(search_area.dimension)
	current_radius: CoordinateType = search_area.side_length / 2

	distance_traveled = 0
	num_probes = 0
	num_responses = 0

	while pm.current_dimension > 0:
		new_search_area = Hypercube(Point.origin(pm.current_dimension), current_radius * 2)

		# Step 1: Binary search in this dimension

		min_radius = 0
		max_radius = new_search_area.side_length / 2
		empty_regions: list[Hypercube] = []
		while min_radius + 1 < max_radius:
			radius = radius_calculator(min_radius, max_radius, pm.current_dimension)
			probe = hypercube_getter(pm, new_search_area.center, radius * 2)

			num_probes += 1
			distance_traveled += drone.distance_to(probe.center)
			drone = probe.center
			if hiker in probe:
				num_responses += 1
				max_radius = radius
			else:
				min_radius = radius
				empty_regions.append(probe)

		if min_radius < 0.5:
			# probe with radius 0.5 and increase min_radius
			probe = hypercube_getter(pm, new_search_area.center, 1)

			num_probes += 1
			distance_traveled += drone.distance_to(probe.center)
			drone = probe.center
			if hiker in probe:
				num_responses += 1
				return SimulationResult(num_probes, distance_traveled, num_responses, probe)
			else:
				min_radius = 0.5

		current_radius = (min_radius + max_radius) / 2
		offset_amount: CoordinateType = max_radius - min_radius
		side_length: CoordinateType = 2 * min_radius

		# First, check if all the dimensions are positive
		probe_center = new_search_area.center
		for dim in range(pm.current_dimension):
			probe_center = probe_center.offset(offset_amount / 2, dim)

		probe = hypercube_getter(pm, probe_center, side_length + offset_amount)

		num_probes += 1
		distance_traveled += drone.distance_to(probe.center)
		drone = probe.center

		if hiker in probe:
			num_responses += 1
			all_positive = True
		else:
			all_positive = False

		positions = tuple(range(pm.current_dimension))
		found_hiker = False
		for num_dims in range(1, pm.current_dimension + 1):
			sign_combos: list[tuple[Literal[-1] | Literal[1], ...]] = []
			for signs in itertools.product((-1, 1), repeat=num_dims):
				if all_positive == all(s == 1 for s in signs):
					sign_combos.append(signs)

			for dims_to_check in itertools.combinations(positions, num_dims):
				# each one can be either 1 or -1
				for signs in sign_combos:
					probe_center = new_search_area.center
					for dim, sign in zip(dims_to_check, signs):
						probe_center = probe_center.offset(sign * offset_amount, dim)

					probe = hypercube_getter(pm, probe_center, side_length) # type: ignore (not sure why this is necessary)

					if pm.current_dimension == 2 and dims_to_check == (0,) and not all_positive:
						# must perform special probe
						probe_center = new_search_area.center.offset(-offset_amount / 2, 0).offset(offset_amount / 2, 1)
						probe = hypercube_getter(pm, probe_center, side_length + offset_amount)

					# check if this is the very last probe. if so, the hiker must be here, no need to actually probe
					if (num_dims == len(positions) and signs == sign_combos[-1]) \
						or (pm.current_dimension == 2 and dims_to_check == (1,)):
						pm.fix_coordinates(zip(dims_to_check, (sign * current_radius for sign in signs)))
						found_hiker = True

						if pm.current_dimension == 1 and dims_to_check == (1,):
							current_radius = max_radius # type: ignore (not sure why this is necessary)

						if current_radius <= 0.5:
							return SimulationResult(num_probes, distance_traveled + drone.distance_to(probe.center), num_responses, probe)

						break

					num_probes += 1
					distance_traveled += drone.distance_to(probe.center)
					drone = probe.center

					if hiker in probe:
						num_responses += 1
						pm.fix_coordinates(zip(dims_to_check, (sign * current_radius for sign in signs)))
						found_hiker = True

						if pm.current_dimension == 1 and not all_positive:
							current_radius = max_radius

						if current_radius <= 0.5:
							return SimulationResult(num_probes, distance_traveled, num_responses, probe)

						break

				if found_hiker:
					break

			if found_hiker:
				break

		# Create the final hypercube
	guess = pm.Hypercube(Point.origin(0), 1)

	return SimulationResult(num_probes, distance_traveled + drone.distance_to(guess.center), num_responses, guess)

def central_binary_search_just_one(search_area: Hypercube, hiker: Point, drone: Point) -> SimulationResult:
	return central_binary_search(search_area, hiker, drone, ProjectionManager.Hypercube)

def central_binary_search_volume(search_area: Hypercube, hiker: Point, drone: Point) -> SimulationResult:
	return central_binary_search(search_area, hiker, drone, ProjectionManager.InsetHypercube, radius_calculator=midpoint_volume)

def central_binary_search_volume_just_one(search_area: Hypercube, hiker: Point, drone: Point) -> SimulationResult:
	return central_binary_search(search_area, hiker, drone, ProjectionManager.Hypercube, radius_calculator=midpoint_volume)

# SHS: AlgorithmPair = (simple_hypercube_search, get_random_hiker_position)
# NCBS: AlgorithmPair = (naive_central_binary_search, get_random_hiker_position_non_equal)
# JO = Just One, only works if it is known that there is just one hiker
# NCBS_JO: AlgorithmPair = (naive_central_binary_search_just_one, get_random_hiker_position_non_equal)
# NCBS_V: AlgorithmPair = (naive_central_binary_search_volume, get_random_hiker_position_non_equal)
# NCBS_V_JO: AlgorithmPair = (naive_central_binary_search_volume_just_one, get_random_hiker_position_non_equal)
# CBS: AlgorithmPair = (central_binary_search, get_random_hiker_position)
# D2S: AlgorithmPair = (domino_2d_search, get_random_hiker_position)
# D3S: AlgorithmPair = (domino_3d_search, get_random_hiker_position)

CBSs: tuple[KidnapperAlgorithm,...] = (central_binary_search, central_binary_search_just_one, central_binary_search_volume, central_binary_search_volume_just_one)

ALGORITHMS_dD: tuple[KidnapperAlgorithm,...] = (simple_hypercube_search,) + CBSs
ALGORITHMS_1D: tuple[KidnapperAlgorithm,...] = (simple_hypercube_search, central_binary_search)
ALGORITHMS_2D: tuple[KidnapperAlgorithm,...] = (simple_hypercube_search, domino_2d_search) + CBSs
ALGORITHMS_3D: tuple[KidnapperAlgorithm,...] = (simple_hypercube_search, domino_3d_search) + CBSs

def get_algorithms(_dims: int) -> tuple[KidnapperAlgorithm,...]:
	match _dims:
		case 1:
			return ALGORITHMS_1D
		case 2:
			return ALGORITHMS_2D
		case 3:
			return ALGORITHMS_3D
		case _:
			return ALGORITHMS_dD
