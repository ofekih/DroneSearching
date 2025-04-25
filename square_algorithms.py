from typing import Callable

from actual_simulations import SimulationResult

from square_utils import Point, Hypercube


type KidnapperAlgorithm = Callable[[Hypercube, Point, Point], SimulationResult]

def simple_hypercube_search(search_area: Hypercube, hiker: Point, drone: Point) -> SimulationResult:
	if search_area.side_length <= 1:
		return SimulationResult(0, 0, 0)
	
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

	return SimulationResult(num_probes + result.P, distance_traveled + result.D, num_responses + result.num_responses)

def central_binary_search(search_area: Hypercube, hiker: Point, drone: Point) -> SimulationResult:
	# No optimizations!
	# step 0, check if we are done
	if search_area.side_length <= 1:
		return SimulationResult(0, 0, 0)
	
	# step 1, move to center
	distance_traveled = search_area.center.distance_to(drone)
	drone = search_area.center
	num_probes = 0
	num_responses = 0

	# step 2, binary search to get width-1 shell
	min_radius = 0
	max_radius = search_area.side_length / 2
	while min_radius + 1 < max_radius:
		radius = (min_radius + max_radius) / 2
		num_probes += 1
		probe = Hypercube(drone, radius)
		if hiker in probe:
			num_responses += 1
			min_radius = radius
		else:
			max_radius = radius

	# step 3, find which face the hiker is on
	for dim in range(search_area.dimension):
		# move one unit in each direction
		num_probes += 1
		probe = Hypercube(drone.offset(1, dim), max_radius - 1)
		if hiker in probe:
			num_responses += 1
			distance_traveled += 1
			drone = drone.offset(1, dim)
			break

		num_probes += 1
		probe = Hypercube(drone.offset(-1, dim), max_radius - 1)
		if hiker in probe:
			num_responses += 1
			distance_traveled += 1
			drone = drone.offset(-1, dim)
			break

		distance_traveled += 2

	# step 4, we know which face the hiker is on. update the hypercube to have 

	return SimulationResult(num_probes, distance_traveled, num_responses)

def domino_2d_search(search_area: Hypercube, hiker: Point, drone: Point) -> SimulationResult:
	if search_area.dimension != 2:
		raise ValueError("Domino 2D search only works in 2 dimensions.")
	
	if search_area.side_length <= 1:
		return SimulationResult(0, 0, 0)
	
	def domino_2d_reduction(area: Hypercube, empty_adjacent: Hypercube) -> SimulationResult:
		if area.side_length <= 1:
			return SimulationResult(0, 0, 0)
		
		nonlocal drone

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

		return SimulationResult(2 + result.P, distance_traveled + result.D, num_responses + result.num_responses)


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

	return SimulationResult(num_probes + result.P, distance_traveled + result.D, num_responses + result.num_responses)



ALGORITHMS: list[KidnapperAlgorithm] = [simple_hypercube_search]
ALGORITHMS_2D: list[KidnapperAlgorithm] = [simple_hypercube_search, domino_2d_search]

def get_algorithms(dims: int) -> list[KidnapperAlgorithm]:
	match dims:
		case 2:
			return ALGORITHMS_2D
		case _:
			return ALGORITHMS