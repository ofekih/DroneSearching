from square_plot import SquarePlotter
from square_utils import Hypercube, Point


def plot_orthants(dimensions: int = 3):
	search_area = Hypercube(Point.origin(dimensions), 1)

	plotter = SquarePlotter()

	for probe in search_area.orthants:
		plotter.plot_search_state(search_area, probe=probe)
		plotter.show(block=False)

	plotter.plot_search_state(search_area)
	plotter.show(block=True)

def plot_domino_2d_search(plotter: SquarePlotter, search_area: Hypercube, hiker: Point, drone: Point):
	if search_area.dimension != 2:
		raise ValueError("Domino 2D search only works in 2 dimensions.")

	if search_area.side_length <= 1:
		return

	def plot_domino_2d_reduction(area: Hypercube, empty_adjacent: Hypercube):
		if area.side_length <= 1:
			return

		nonlocal drone

		candidates = list(area.orthants)

		# move drone to halfway between the two areas
		center = area.center.interpolate(empty_adjacent.center, 0.5)
		drone = center

		# probe the center
		probe = Hypercube(center, area.side_length)
		plotter.plot_search_state(search_area, hiker, drone, empty_regions=[empty_adjacent], candidates=candidates, probe=probe)
		plotter.show(block=False)
		if hiker in probe:
			empty_adjacent_candidates = [o for o in empty_adjacent.orthants if o in probe]
			candidates = [c for c in candidates if c in probe]
		else:
			empty_adjacent_candidates = [c for c in candidates if c in probe]
			candidates = [c for c in candidates if c not in probe]

		probe = candidates[0]
		drone = probe.center
		plotter.plot_search_state(search_area, hiker, drone, empty_adjacent_candidates, candidates, probe=probe)
		plotter.show(block=False)
		if hiker in probe:
			candidate = candidates[0]
		else:
			candidate = candidates[1]

		# correct empty adjacent candidate should have one dimension in common
		empty_adjacent_candidate = empty_adjacent_candidates[0] if empty_adjacent_candidates[0].center.shares_any_coordinate(candidate.center) else empty_adjacent_candidates[1]

		plot_domino_2d_reduction(candidate, empty_adjacent_candidate)


	orthant_iter = search_area.orthants
	correct_orthant = next(orthant_iter)
	empty_adjacent_orthant = None
	for probe in orthant_iter:
		drone = probe.center

		plotter.plot_search_state(search_area, hiker, drone, [empty_adjacent_orthant] if empty_adjacent_orthant else [], candidates=[probe], probe=probe)
		plotter.show(block=False)
		if hiker in probe:
			correct_orthant = probe
			break

		empty_adjacent_orthant = probe

	if empty_adjacent_orthant:
		plot_domino_2d_reduction(correct_orthant, empty_adjacent_orthant)
	else:
		plot_domino_2d_search(plotter, correct_orthant, hiker, drone)

def plot_domino_3d_search(plotter: SquarePlotter, search_area: Hypercube, hiker: Point, drone: Point):
	if search_area.dimension != 3:
		raise ValueError("Domino 3D search only works in 3 dimensions.")

	if search_area.side_length <= 1:
		return

	def plot_domino_3d_reduction(area: Hypercube, empty_adjacent_3: tuple[Hypercube, Hypercube, Hypercube]):
		if area.side_length <= 1:
			return

		nonlocal drone

		candidates = list(area.orthants)
		known_empty = list(empty_adjacent_3)

		directly_adjacent = [o for o in empty_adjacent_3 if o.center.shares_any_coordinate(area.center)]
		centers = [area.center.interpolate(o.center, 0.5) for o in directly_adjacent]
		closest_center = min(centers, key=lambda c: c.distance_to(drone))

		probe = Hypercube(closest_center, area.side_length)
		drone = probe.center

		# Visualize first probe
		plotter.plot_search_state(search_area, hiker, drone, empty_regions=known_empty, candidates=candidates, probe=probe)
		plotter.show(block=False)

		if hiker in probe:
			candidates = [c for c in candidates if c in probe]
		else:
			candidates = [c for c in candidates if c not in probe]
			known_empty.append(probe)

		other_center = max(centers, key=lambda c: c.distance_to(drone))
		probe = Hypercube(other_center, area.side_length)
		drone = probe.center

		# Visualize second probe
		plotter.plot_search_state(search_area, hiker, drone, empty_regions=known_empty, candidates=candidates, probe=probe)
		plotter.show(block=False)

		if hiker in probe:
			candidates = [c for c in candidates if c in probe]
		else:
			candidates = [c for c in candidates if c not in probe]
			known_empty.append(probe)

		# only two candidates left
		probe = candidates[0]
		drone = probe.center

		# Visualize third probe
		plotter.plot_search_state(search_area, hiker, drone, empty_regions=known_empty, candidates=candidates, probe=probe)
		plotter.show(block=False)

		if hiker in probe:
			candidate = candidates[0]
		else:
			candidate = candidates[1]

		empty_adjacent_candidates = [n for n in candidate.neighbors if any(n in e for e in known_empty)]
		# should have two
		final_empty_candidate = [n1 for n1 in empty_adjacent_candidates[0].neighbors if any(n1 in e for e in known_empty) and any(n1 == n2 for n2 in empty_adjacent_candidates[1].neighbors)][0]

		plot_domino_3d_reduction(candidate, (final_empty_candidate, empty_adjacent_candidates[0], empty_adjacent_candidates[1]))

	def plot_domino_2d_reduction(area: Hypercube, empty_adjacent: Hypercube):
		if area.side_length <= 1:
			return

		nonlocal drone

		candidates = list(area.orthants)

		# move drone to halfway between the two areas
		center = area.center.interpolate(empty_adjacent.center, 0.5)
		drone = center

		# Visualize first probe
		probe = Hypercube(center, area.side_length)
		plotter.plot_search_state(search_area, hiker, drone, empty_regions=[empty_adjacent], candidates=candidates, probe=probe)
		plotter.show(block=False)

		if hiker in probe:
			empty_adjacent_candidates = [o for o in empty_adjacent.orthants if o in probe]
			candidates = [c for c in candidates if c in probe]
		else:
			empty_adjacent_candidates = [c for c in candidates if c in probe]
			candidates = [c for c in candidates if c not in probe]

		probe = candidates[0]
		drone = probe.center

		# Visualize second probe
		plotter.plot_search_state(search_area, hiker, drone, empty_regions=empty_adjacent_candidates, candidates=candidates, probe=probe)
		plotter.show(block=False)

		if hiker in probe:
			# recurse to 2d reduction
			empty_adjacent_candidate = [e for e in empty_adjacent_candidates if e in probe.neighbors][0]
			plot_domino_2d_reduction(probe, empty_adjacent_candidate)
			return

		new_empty_candidates = [probe]

		plotter.plot_search_state(search_area, hiker, drone, empty_adjacent_candidates, candidates=candidates, probe=probe)
		plotter.show(block=False)

		# find candidate which does not share a coordinate
		correct_candidate = [c for c in candidates if not c in probe.neighbors][0]

		for probe in candidates[1:]:
			if probe == correct_candidate:
				continue

			drone = probe.center

			# Visualize additional probes
			plotter.plot_search_state(search_area, hiker, drone, empty_regions=empty_adjacent_candidates, candidates=[c for c in candidates if c != probe], probe=probe)
			plotter.show(block=False)

			if hiker in probe:
				correct_candidate = probe
				break

			new_empty_candidates.append(probe)

		# find a new empty candidate which shares a coordinate with the correct candidate
		new_empty_candidate = [e for e in new_empty_candidates if e in correct_candidate.neighbors][0]
		
		# find the two old candidates which share a coordinate with these two guys
		old_empty_1 = [e for e in empty_adjacent_candidates if e in correct_candidate.neighbors][0]
		old_empty_2 = [e for e in empty_adjacent_candidates if e in new_empty_candidate.neighbors][0]

		plotter.plot_search_state(search_area, hiker, drone, [old_empty_1, old_empty_2, new_empty_candidate], candidates=[correct_candidate], probe=correct_candidate)
		plotter.show(block=False)

		plot_domino_3d_reduction(correct_candidate, (old_empty_1, old_empty_2, new_empty_candidate))

	orthant_iter = search_area.orthants
	correct_orthant = next(orthant_iter)
	empty_orthants: list[Hypercube] = []

	# Visualize initial state
	plotter.plot_search_state(search_area, hiker, drone)
	plotter.show(block=False)

	for probe in orthant_iter:
		drone = probe.center

		# Visualize each orthant probe
		plotter.plot_search_state(search_area, hiker, drone, empty_orthants, probe=probe)
		plotter.show(block=False)

		if hiker in probe:
			correct_orthant = probe
			break

		empty_orthants.append(probe)

	if len(empty_orthants) == 0:
		plot_domino_3d_search(plotter, correct_orthant, hiker, drone)
	else:
		adj_empty_orthants = [e for e in empty_orthants if e in correct_orthant.neighbors]
		
		plotter.plot_search_state(search_area, hiker, drone, adj_empty_orthants, probe=correct_orthant)
		plotter.show(block=False)

		
		final_empty_orthant = [] if len(adj_empty_orthants) == 1 else [e for e in adj_empty_orthants[0].neighbors if e in adj_empty_orthants[1].neighbors and e in empty_orthants]

		if len(final_empty_orthant) > 0:
			plot_domino_3d_reduction(correct_orthant, (final_empty_orthant[0], adj_empty_orthants[0], adj_empty_orthants[1]))
		else:
			plot_domino_2d_reduction(correct_orthant, adj_empty_orthants[0])

if __name__ == "__main__":
	plotter = SquarePlotter()

	# Uncomment one of the examples below to run it

	# 2D domino search example
	# plot_domino_2d_search(plotter, Hypercube(Point.origin(2), 32), Point((13, 4)), Point.origin(2))

	# 3D domino search example
	plot_domino_3d_search(plotter, Hypercube(Point.origin(3), 8), Point((2.853313899411406, 3.678879071104282, -0.7464353935619741)), Point.origin(3))

	# Orthant visualization examples
	# plot_orthants(1)
	# plot_orthants(2)
	# plot_orthants(3)
