import random

from square_utils import CoordinateType, Point, Hypercube

from square_algorithms import get_algorithms

# set random seed
# random.seed(0)

def verify_algorithms(min_dim: int = 1, max_dim: int = 10, num_iterations: int = 1000, n: CoordinateType = 2 ** 10):
	for dims in range(min_dim, max_dim + 1):
		search_area = Hypercube(Point(tuple(0 for _ in range(dims))), side_length=n / 2)
		drone = search_area.center

		for _ in range(num_iterations):
			hiker = get_random_hiker_position(search_area)

			for algorithm in get_algorithms(dims):
				result = algorithm(search_area, hiker, drone)
				assert hiker in result.area, f'{algorithm.__name__} failed to find hiker in {dims} dimensions'
				assert result.area.side_length <= 1, f'{algorithm.__name__} final search area too large in {dims} dimensions'

	print('All algorithms verified')

def get_random_hiker_position(search_area: Hypercube):
	return Point(tuple(random.uniform(search_area.center.coordinates[i] - search_area.side_length / 2, search_area.center.coordinates[i] + search_area.side_length / 2) for i in range(search_area.dimension)))


def run_simulation(n: CoordinateType, dims: int, num_iterations: int = 1):
	search_area = Hypercube(Point(tuple(0 for _ in range(dims))), side_length=n / 2)
	drone = search_area.center

	for _ in range(num_iterations):
		hiker = get_random_hiker_position(search_area)

		for algorithm in get_algorithms(dims):
			result = algorithm(search_area, hiker, drone)
			print(f'{algorithm.__name__}: {result}')


if __name__ == '__main__':
	verify_algorithms(1, 6, n = 2**6)
	# run_simulation(16, 6)
	run_simulation(2**10, 3)
