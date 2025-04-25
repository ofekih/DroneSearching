import math
import random
from typing import NamedTuple

from algorithms import ALGORITHMS
from utils import PRECISION, Circle
import multiprocessing
from pathlib import Path
import csv
import time
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle as plt_Circle

DATA_DIRECTORY = Path(__file__).parent / "data"
DATA_DIRECTORY.mkdir(exist_ok=True)

class SimulationResult(NamedTuple):
	P: int # total number of probes
	D: float # total distance traveled
	num_responses: int # number of hiker responses

def save_simulation_results(algorithm: int, n: float, results: list[SimulationResult]):
	with open(DATA_DIRECTORY / f"algorithm_{algorithm}.csv", "a", newline="") as f:
		writer = csv.writer(f)
		writer.writerows([(n, result.P, result.D, result.num_responses) for result in results])

class Position(NamedTuple):
	x: float
	y: float

# function that draws:
# hiker with X
# drone with D
# probes with circles
# given the search area
def draw_step(probes: list[Circle], search_area: Circle, hiker: Position, drone: Position):	
	fig, ax = plt.subplots(figsize=(10, 10))
	
	# Draw search area
	search_circle = plt_Circle((search_area.x, search_area.y), search_area.r, 
							   fill=False, color='blue', linestyle='--', alpha=0.5)
	ax.add_patch(search_circle)
	
	# Draw probes
	for probe in probes:
		# Scale and translate probe according to search area
		probe_circle = plt_Circle((probe.x, probe.y), probe.r,
								  fill=False, color='green', alpha=0.7)
		ax.add_patch(probe_circle)
	
	# Draw hiker with X
	ax.plot(hiker.x, hiker.y, 'rx', markersize=10, markeredgewidth=3, label='Hiker')
	
	# Draw drone with D
	ax.plot(drone.x, drone.y, 'bo', markersize=8, label='Drone')
	ax.text(drone.x, drone.y, 'D', color='white', ha='center', va='center', fontweight='bold')
	
	# Set limits and labels
	max_radius = search_area.r * 1.2
	ax.set_xlim(search_area.x - max_radius, search_area.x + max_radius)
	ax.set_ylim(search_area.y - max_radius, search_area.y + max_radius)
	ax.set_aspect('equal')
	ax.set_title('Drone Search Visualization')
	ax.legend()
	ax.grid(True)
	
	plt.show()
	

def get_random_hiker_position(n: float):
	r: float = random.uniform(0, n)
	theta = random.uniform(0, 2 * math.pi)

	return Position(r * math.cos(theta), r * math.sin(theta))

def probe_query(probe: Circle, hiker: Position):
	return math.sqrt((probe.x - hiker.x)**2 + (probe.y - hiker.y)**2) < probe.r + PRECISION.epsilon

def scale_translate_and_rotate_probes(placement: list[Circle], search_area: Circle, drone: Position) -> list[Circle]:
	# # first, scale and translate the probes
	# scaled_and_translated_probes = [Circle(search_area.x + probe.x * search_area.r, search_area.y + probe.y * search_area.r, probe.r * search_area.r) for probe in placement]

	# first, rotate the probes, such that the drone is as close to the first probe as possible
	# find the angle between the first probe and the drone
	first_x, first_y, _ = placement[0]
	first_probe_angle = math.atan2(first_y, first_x)

	origin_x, origin_y, _ = search_area
	drone_angle = math.atan2(drone.y - origin_y, drone.x - origin_x)

	angle = drone_angle - first_probe_angle

	# next, rotate the probes
	rotated_probes = [Circle(
		x=probe.x * math.cos(angle) - probe.y * math.sin(angle),
		y=probe.x * math.sin(angle) + probe.y * math.cos(angle),
		r=probe.r
	) for probe in placement]

	# finally, scale and translate the probes
	scaled_and_translated_probes = [Circle(
		x=search_area.x + probe.x * search_area.r,
		y=search_area.y + probe.y * search_area.r,
		r=probe.r * search_area.r
	) for probe in rotated_probes]
	
	return scaled_and_translated_probes

def find_hiker(placement: list[Circle], search_area: Circle, hiker: Position, drone: Position = Position(0, 0)) -> SimulationResult:
	# drone distance is drone distance from center
	if search_area.r < 1.0 + PRECISION.epsilon:
		# if drone is far from search area, drone must travel to search area
		return SimulationResult(0, math.sqrt((drone.x - search_area.x)**2 + (drone.y - search_area.y)**2), 0)
	
	probes = scale_translate_and_rotate_probes(placement, search_area, drone)

	num_probes_done = 0
	total_distance_traveled = 0
	num_responses = 0
	found_hiker = False

	# draw_step(probes, search_area, hiker, drone)

	for probe in probes:
		if probe == probes[-1]:
			# last probe does not need to be performed
			found_hiker = True
		else:
			total_distance_traveled += math.sqrt((probe.x - drone.x)**2 + (probe.y - drone.y)**2)
			drone = Position(probe.x, probe.y)

			if probe_query(probe, hiker):
				found_hiker = True
				num_responses += 1

			num_probes_done += 1

		if found_hiker:
			# print probe index
			# print(num_probes_done, end=', ')

			remaining_work = find_hiker(placement, probe, hiker, drone)
			num_probes_done += remaining_work.P
			total_distance_traveled += remaining_work.D
			num_responses += remaining_work.num_responses

			
			return SimulationResult(num_probes_done, total_distance_traveled, num_responses)
	
	# raise error
	raise Exception("Hiker not found")

def simulate_algorithm(placement: list[Circle], n: float) -> SimulationResult:
	return find_hiker(placement, Circle(0, 0, n), get_random_hiker_position(n))

def simulate_specific_algorithm(algorithm: int, n: float, num_simulations: int, batch_size: int = 2 ** 13):
	placement = ALGORITHMS[algorithm]
	PRECISION.set_precision(10 if algorithm < 7 else 2)
	
	total_batches = num_simulations // batch_size
	
	for i in range(total_batches):
		batch_results = [simulate_algorithm(placement, n) for _ in range(batch_size)]
		save_simulation_results(algorithm, n, batch_results)
		
		# Report progress periodically (every ~1% or every 10 batches, whichever is less frequent)
		report_interval = max(1, total_batches // 100)
		if i % report_interval == 0 or i == total_batches - 1:
			percent_complete = (i + 1) / total_batches * 100
			print(f"Algorithm {algorithm}: {percent_complete:.1f}% complete ({i+1}/{total_batches} batches)")
	
	# run the remaining simulations
	remaining_simulations = num_simulations % batch_size
	if remaining_simulations > 0:
		remaining_results = [simulate_algorithm(placement, n) for _ in range(remaining_simulations)]
		save_simulation_results(algorithm, n, remaining_results)

def run_algorithm_process(algorithm: int, n: float, num_simulations: int, batch_size: int) -> tuple[int, float]:
	"""Worker function for multiprocessing that runs a single algorithm's simulations"""
	# Seed the random number generator with process-specific seed for independence
	process_seed = int(time.time() * 1000) % 100000 + os.getpid() + algorithm * 1000
	random.seed(process_seed)
	
	start_time = time.time()
	print(f"Process for algorithm {algorithm} started (PID: {os.getpid()})")
	
	simulate_specific_algorithm(algorithm, n, num_simulations, batch_size)
	
	elapsed_time = time.time() - start_time
	print(f"Algorithm {algorithm} completed in {elapsed_time:.2f} seconds")
	return algorithm, elapsed_time

if __name__ == '__main__':
	# Parameters for the simulations
	n = 2**20
	num_simulations = 2**22

	# # for i, c in enumerate(ALGORITHMS[6]):
	# # 	print(f"Circle {i}: {c.r} vs. {ALGORITHMS[6][0].r ** (i + 1)}")


	# hiker = Position(x=492983.0212156907, y=-19415.937748734763)
	# # hiker = Position(x=0, y=0)

	# result = find_hiker(ALGORITHMS[6], Circle(0, 0, n), hiker)

	# print(result)

	# # while True:
	# # 	hiker = get_random_hiker_position(n)
	# # 	result = find_hiker(ALGORITHMS[6], Circle(0, 0, n), hiker)

	# # 	if result.P == 78:
	# # 		print(hiker)
	# # 		break

	# exit(0)
	
	# Setting a reasonable batch size - too large and it will use too much memory
	# Too small and there will be too much overhead from file operations
	batch_size = 2**16  # Adjust this based on available memory
	
	start_time_total = time.time()
	
	# Use spawn method for better cross-platform compatibility
	multiprocessing.set_start_method('spawn', force=True)
	
	# Create a pool of worker processes
	with multiprocessing.Pool() as pool:
		# Map each algorithm to a separate process
		jobs = []
		for _ in range(num_simulations // batch_size):
			for algorithm in range(1, 9):
				jobs.append(pool.apply_async(run_algorithm_process, (algorithm, n, batch_size, batch_size))) # type: ignore
		
		# Wait for all jobs to complete and collect results
		results: list[tuple[int, float]] = []
		for job in jobs: # type: ignore
			algorithm, elapsed_time = job.get() # type: ignore
			results.append((algorithm, elapsed_time)) # type: ignore
	
	# Sort and display results
	results.sort(key=lambda x: x[0])  # Sort by algorithm number
	print("\nFinal Results:")
	for algorithm, elapsed_time in results:
		print(f"Algorithm {algorithm}: {elapsed_time:.2f} seconds")
	
	total_time = time.time() - start_time_total
	print(f"\nAll simulations completed in {total_time:.2f} seconds")