import random, csv, os, time
from pathlib import Path
import multiprocessing
from collections import defaultdict
from typing import Any
from tqdm.auto import tqdm

from square_utils import CoordinateType, Point, Hypercube

from square_algorithms import SimulationResult, get_algorithms

RAW_DATA_DIRECTORY = Path('raw-data')
RAW_DATA_DIRECTORY.mkdir(exist_ok=True)

DATA_FILE_INDEX = 0

DATA_FILE = RAW_DATA_DIRECTORY / f'simulations_{DATA_FILE_INDEX}.csv'

def save_simulation_results(algorithm: str, dims: int, n: CoordinateType, results: list[tuple[CoordinateType, SimulationResult]], file_path: Path | None = None):
	"""
	Save simulation results to a CSV file.

	Args:
		algorithm: Name of the algorithm used
		dims: Number of dimensions
		n: Size of the search area
		results: list of (hiker_distance, SimulationResult) tuples
		file_path: Optional custom file path to save results to
	"""
	output_file = file_path if file_path is not None else DATA_FILE

	# Create header if file doesn't exist
	if not output_file.exists():
		with output_file.open('w') as f:
			writer = csv.writer(f)
			writer.writerow(['algorithm', 'dims', 'n', 'hiker_distance', 'P', 'D', 'num_responses'])

	# Append results to file
	with output_file.open('a') as f:
		writer = csv.writer(f)
		writer.writerows([(algorithm, dims, n, hiker_distance, result.P, result.D, result.num_responses)
						  for hiker_distance, result in results])

# set random seed
# random.seed(0)

def verify_algorithms(min_dim: int = 1, max_dim: int = 5, num_iterations: int = 1000, n: CoordinateType = 2 ** 10):
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

def run_simulation_batch(n: CoordinateType, dims: int, num_iterations: int = 2 ** 6, file_path: Path | None = None):
	"""
	Run a batch of simulations for all algorithms in the given dimensions.

	Args:
		n: Size of the search area
		dims: Number of dimensions
		num_iterations: Number of simulations to run
		file_path: Optional custom file path to save results to
	"""
	search_area = Hypercube(Point(tuple(0 for _ in range(dims))), side_length=n / 2)
	drone = search_area.center

	hiker_positions = [get_random_hiker_position(search_area) for _ in range(num_iterations)]
	hiker_distances = [drone.distance_to(hiker) for hiker in hiker_positions]

	for algorithm in get_algorithms(dims):
		results = [algorithm(search_area, hiker, drone) for hiker in hiker_positions]
		save_simulation_results(algorithm.__name__, dims, n, list(zip(hiker_distances, results)), file_path)

def run_worker_process(task: tuple[CoordinateType, int, int, int]) -> tuple[int, int, float, int]:
	"""
	Worker function for multiprocessing that runs a single batch of simulations.

	Args:
		task: A tuple containing (n, dims, batch_size, task_id)

	Returns:
		tuple of (task_id, dims, elapsed_time, pid)
	"""
	n, dims, batch_size, task_id = task
	pid = os.getpid()

	# Set a process-specific random seed for independence
	process_seed = int(time.time() * 1000) % 100000 + pid + task_id * 1000
	random.seed(process_seed)

	start_time = time.time()

	# Create a process-specific output file to avoid race conditions
	# Include both dimension, task ID and PID to ensure uniqueness
	output_file = RAW_DATA_DIRECTORY / f'simulations_{dims}d_task{task_id}_pid{pid}.csv'

	# Run the batch
	run_simulation_batch(n, dims, batch_size, output_file)

	elapsed_time = time.time() - start_time
	return task_id, dims, elapsed_time, pid

def run_simulations(n: CoordinateType = 2**20, min_dim: int = 1, max_dim: int = 5, num_iterations: int = 2**20, batch_size: int = 2**10, num_processes: int | None = None):
	"""
	Run simulations in parallel using multiprocessing across multiple dimensions.
	Uses a task queue approach to dynamically distribute work across all available processors.

	Args:
		n: Size of the search area (default: 2^20)
		min_dim: Minimum dimension to simulate (default: 1)
		max_dim: Maximum dimension to simulate (default: 5)
		num_iterations: Total number of simulations to run per dimension (default: 2^20)
		batch_size: Number of simulations to run in each batch (default: 2^10)
		num_processes: Number of processes to use (defaults to CPU count)
	"""
	if num_processes is None:
		num_processes = max(multiprocessing.cpu_count() - 1, 1)

	# Calculate how many dimensions we're simulating
	num_dims = max_dim - min_dim + 1

	# Create tasks for all dimensions
	tasks_by_dim: defaultdict[int, list[tuple[CoordinateType, int, int, int]]] = defaultdict(list)
	dimension_tasks: list[list[tuple[CoordinateType, int, int, int]]] = []
	task_id = 0

	# First, create tasks for each dimension separately
	for dims in range(min_dim, max_dim + 1):
		dim_tasks: list[tuple[CoordinateType, int, int, int]] = []

		# Calculate how many batches we need for this dimension
		total_batches = num_iterations // batch_size
		remaining_simulations = num_iterations % batch_size

		# Create tasks for full batches
		for _ in range(total_batches):
			task = (n, dims, batch_size, task_id)
			tasks_by_dim[dims].append(task)
			dim_tasks.append(task)
			task_id += 1

		# Create a task for any remaining simulations
		if remaining_simulations > 0:
			task = (n, dims, remaining_simulations, task_id)
			tasks_by_dim[dims].append(task)
			dim_tasks.append(task)
			task_id += 1

		dimension_tasks.append(dim_tasks)

	# Now interleave tasks from different dimensions to ensure all dimensions
	# make progress even if the process is stopped early
	all_tasks: list[tuple[CoordinateType, int, int, int]] = []
	max_tasks_per_dim = max(len(tasks) for tasks in dimension_tasks)

	for i in range(max_tasks_per_dim):
		for dim_tasks in dimension_tasks:
			if i < len(dim_tasks):
				all_tasks.append(dim_tasks[i])

	# Print simulation setup information
	print(f"🚀 Running {num_iterations:,} simulations per dimension from {min_dim}D to {max_dim}D")
	print(f"📊 Created {len(all_tasks)} tasks across {num_dims} dimensions")
	print(f"💻 Using {num_processes} processes with dynamic task distribution")
	print(f"📦 Each task will process up to {batch_size:,} simulations")
	print("\n" + "=" * 80 + "\n")

	start_time_total = time.time()

	try:
		# Use spawn method for better cross-platform compatibility
		multiprocessing.set_start_method('spawn', force=True)
	except RuntimeError:
		# If already set, ignore the error
		pass

	# Create progress bars for each dimension
	progress_bars: dict[int, Any] = {}
	for dims in range(min_dim, max_dim + 1):
		num_tasks = len(tasks_by_dim[dims])
		desc = f"{dims}D Simulations"
		progress_bars[dims] = tqdm(total=num_tasks, desc=desc, position=dims-min_dim, leave=True)

	# Create a pool of worker processes
	with multiprocessing.Pool(processes=num_processes) as pool:
		# Use imap_unordered for better progress tracking
		results: list[tuple[int, int, float, int]] = []
		for result in pool.imap_unordered(run_worker_process, all_tasks):
			task_id, dims, elapsed_time, pid = result
			results.append(result)

			# Update the progress bar for this dimension
			progress_bars[dims].update(1)
			progress_bars[dims].set_postfix({"Last": f"{elapsed_time:.2f}s"})

	# Close all progress bars
	for bar in progress_bars.values():
		bar.close()

	# Group results by dimension
	dim_results: defaultdict[int, list[tuple[int, float, int]]] = defaultdict(list)
	for task_id, dims, elapsed_time, pid in results:
		dim_results[dims].append((task_id, elapsed_time, pid))

	# Print a summary of results
	print("\n" + "=" * 80)
	print("\n📊 Simulation Summary:")

	total_tasks_completed = 0
	for dims in sorted(dim_results.keys()):
		# Calculate statistics for this dimension
		tasks_completed = len(dim_results[dims])
		total_tasks_completed += tasks_completed
		total_time = sum(elapsed for _, elapsed, _ in dim_results[dims])
		avg_time = total_time / tasks_completed
		max_time = max(elapsed for _, elapsed, _ in dim_results[dims])
		min_time = min(elapsed for _, elapsed, _ in dim_results[dims])

		# Count unique PIDs that processed tasks for this dimension
		unique_pids = len(set(pid for _, _, pid in dim_results[dims]))

		print(f"\n{dims}D: {tasks_completed} tasks completed across {unique_pids} processes")
		print(f"  ⏱️  Task time: avg {avg_time:.2f}s (min: {min_time:.2f}s, max: {max_time:.2f}s)")

	total_elapsed_time = time.time() - start_time_total
	print(f"\n⏱️  Total execution time: {total_elapsed_time:.2f} seconds")
	print(f"✅ Completed {total_tasks_completed} tasks across {num_dims} dimensions")
	print(f"💾 Results saved to {RAW_DATA_DIRECTORY}/simulations_*d_task*_pid*.csv files")


if __name__ == '__main__':
	# Uncomment one of these examples to run:
	# verify_algorithms()
	run_simulations()

	# Run a single batch of simulations for a specific dimension
	# run_simulation_batch(n=2**20, dims=3)

	# Run simulations in parallel for dimensions 1-5 (using all available CPU cores)
	# run_simulations(n=2**20, min_dim=1, max_dim=5, num_iterations=2**20, batch_size=2**10)

	# Run simulations for dimensions 2-3 with a specific number of processes and smaller batches
	# run_simulations(n=2**20, min_dim=2, max_dim=3, num_iterations=2**20, batch_size=2**8, num_processes=4)

	# Run simulations with all default parameters (n=2^20, dims=1-5, iterations=2^20, batch_size=2^10)
	# run_simulations()

	pass  # Remove this line when uncommenting one of the examples above

