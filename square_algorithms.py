from typing import Callable, Generator

from actual_simulations import SimulationResult

from square_utils import Point, Hypercube


type KidnapperAlgorithm = Callable[[Hypercube, Point, Point], SimulationResult]

def _generate_gray_codes(n: int) -> Generator[tuple[int, ...], None, None]:
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
    prev_gray_codes = list(_generate_gray_codes(n - 1))

    # Yield first half: prepend 0 to G(n-1)
    for code in prev_gray_codes:
        yield (0,) + code

    # Yield second half: prepend 1 to reversed G(n-1)
    for code in reversed(prev_gray_codes):
        yield (1,) + code

def simple_hypercube_search(hypercube: Hypercube, hiker: Point, drone: Point) -> SimulationResult:
	if hypercube.side_length <= 1:
		return SimulationResult(0, 0, 0)
	
	# check half-sized hypercubes
	new_side_length = hypercube.side_length / 2
	offset = new_side_length / 2

	distance_traveled = 0
	num_probes = 0
	num_responses = 0

	def new_center_from_code(code: tuple[int, ...]) -> Point:
		return Point(tuple(hypercube.center.coordinates[i] + offset * (2 * code[i] - 1) for i in range(hypercube.dimension)))

	gray_code_iter = _generate_gray_codes(hypercube.dimension)
	correct_code = next(gray_code_iter)

	for gray_code in gray_code_iter:
		new_center = new_center_from_code(gray_code)
		distance_traveled += drone.distance_to(new_center)
		drone = new_center

		num_probes += 1
		probe = Hypercube(new_center, new_side_length)
		print(f'Probing {probe} (distance from probe to hiker: {probe.center.distance_to(hiker)})')
		if probe.contains(hiker):
			num_responses += 1
			correct_code = gray_code
			print(f'Found hiker in {probe}')
			break

	new_center = new_center_from_code(correct_code)
	distance_traveled += drone.distance_to(new_center)
	drone = new_center

	result = simple_hypercube_search(Hypercube(new_center, new_side_length), hiker, drone)

	return SimulationResult(num_probes + result.P, distance_traveled + result.D, num_responses + result.num_responses)

def central_binary_search(hypercube: Hypercube, hiker: Point, drone: Point) -> SimulationResult:
	# No optimizations!
	# step 0, check if we are done
	if hypercube.side_length <= 1:
		return SimulationResult(0, 0, 0)
	
	# step 1, move to center
	distance_traveled = hypercube.center.distance_to(drone)
	drone = hypercube.center
	num_probes = 0
	num_responses = 0

	# step 2, binary search to get width-1 shell
	min_radius = 0
	max_radius = hypercube.side_length / 2
	while min_radius + 1 < max_radius:
		radius = (min_radius + max_radius) / 2
		num_probes += 1
		probe = Hypercube(drone, radius)
		if probe.contains(hiker):
			num_responses += 1
			min_radius = radius
		else:
			max_radius = radius

	# step 3, find which face the hiker is on
	for dim in range(hypercube.dimension):
		# move one unit in each direction
		num_probes += 1
		probe = Hypercube(drone.offset(1, dim), max_radius - 1)
		if probe.contains(hiker):
			num_responses += 1
			distance_traveled += 1
			drone = drone.offset(1, dim)
			break

		num_probes += 1
		probe = Hypercube(drone.offset(-1, dim), max_radius - 1)
		if probe.contains(hiker):
			num_responses += 1
			distance_traveled += 1
			drone = drone.offset(-1, dim)
			break

		distance_traveled += 2

	# step 4, we know which face the hiker is on. update the hypercube to have 

	return SimulationResult(num_probes, distance_traveled, num_responses)


ALGORITHMS: list[KidnapperAlgorithm] = [simple_hypercube_search]
