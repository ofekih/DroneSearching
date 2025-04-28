from typing import Callable, NamedTuple

from square_utils import Point, Hypercube

class SimulationResult(NamedTuple):
	P: int # total number of probes
	D: float # total distance traveled
	num_responses: int # number of hiker responses
	area: Hypercube


type KidnapperAlgorithm = Callable[[Hypercube, Point, Point], SimulationResult]

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

# def central_binary_search(search_area: Hypercube, hiker: Point, drone: Point) -> SimulationResult:
# 	# No optimizations!
# 	# step 0, check if we are done
# 	if search_area.side_length <= 1:
# 		return SimulationResult(0, 0, 0, search_area)
	
# 	# step 1, move to center
# 	distance_traveled = search_area.center.distance_to(drone)
# 	drone = search_area.center
# 	num_probes = 0
# 	num_responses = 0

# 	# step 2, binary search to get width-1 shell
# 	min_radius = 0
# 	max_radius = search_area.side_length / 2
# 	while min_radius + 1 < max_radius:
# 		radius = (min_radius + max_radius) / 2
# 		num_probes += 1
# 		probe = Hypercube(drone, radius)
# 		if hiker in probe:
# 			num_responses += 1
# 			min_radius = radius
# 		else:
# 			max_radius = radius

# 	# step 3, find which face the hiker is on
# 	for dim in range(search_area.dimension):
# 		# move one unit in each direction
# 		num_probes += 1
# 		probe = Hypercube(drone.offset(1, dim), max_radius - 1)
# 		if hiker in probe:
# 			num_responses += 1
# 			distance_traveled += 1
# 			drone = drone.offset(1, dim)
# 			break

# 		num_probes += 1
# 		probe = Hypercube(drone.offset(-1, dim), max_radius - 1)
# 		if hiker in probe:
# 			num_responses += 1
# 			distance_traveled += 1
# 			drone = drone.offset(-1, dim)
# 			break

# 		distance_traveled += 2

# 	# step 4, we know which face the hiker is on. update the hypercube to have 

# 	return SimulationResult(num_probes, distance_traveled, num_responses)

def domino_2d_search(search_area: Hypercube, hiker: Point, drone: Point) -> SimulationResult:
	if search_area.dimension != 2:
		raise ValueError("Domino 2D search only works in 2 dimensions.")
	
	if search_area.side_length <= 1:
		return SimulationResult(0, drone.distance_to(search_area.center), 0, search_area)
	
	def domino_2d_reduction(area: Hypercube, empty_adjacent: Hypercube) -> SimulationResult:
		if area.side_length <= 1:
			return SimulationResult(0, drone.distance_to(area.center), 0, area)
		
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
		if area.side_length <= 1:
			return SimulationResult(0, drone.distance_to(area.center), 0, area)
		
		nonlocal drone

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
		if area.side_length <= 1:
			return SimulationResult(0, drone.distance_to(area.center), 0, area)
		
		nonlocal drone

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

ALGORITHMS: list[KidnapperAlgorithm] = [simple_hypercube_search]
ALGORITHMS_2D: list[KidnapperAlgorithm] = [simple_hypercube_search, domino_2d_search]
ALGORITHMS_3D: list[KidnapperAlgorithm] = [simple_hypercube_search, domino_3d_search]

def get_algorithms(dims: int) -> list[KidnapperAlgorithm]:
	match dims:
		case 2:
			return ALGORITHMS_2D
		case 3:
			return ALGORITHMS_3D
		case _:
			return ALGORITHMS
