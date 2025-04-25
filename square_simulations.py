import random

from square_utils import CoordinateType, Point, Hypercube

from square_algorithms import ALGORITHMS

# set random seed
# random.seed(0)

def get_random_hiker_position(search_area: Hypercube):
	return Point(tuple(random.uniform(search_area.center.coordinates[i] - search_area.side_length / 2, search_area.center.coordinates[i] + search_area.side_length / 2) for i in range(search_area.dimension)))


def run_simulation(n: CoordinateType, dims: int, num_iterations: int = 1):
	search_area = Hypercube(Point(tuple(0 for _ in range(dims))), side_length=n / 2)
	drone = search_area.center

	for _ in range(num_iterations):
		hiker = get_random_hiker_position(search_area)
		print(f'Hiker: {hiker}')

		for algorithm in ALGORITHMS:
			result = algorithm(search_area, hiker, drone)
			print(f'{algorithm.__name__}: {result}')


if __name__ == '__main__':
	run_simulation(16, 6)
