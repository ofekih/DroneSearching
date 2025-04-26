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

		plotter.plot_search_state(search_area, hiker, drone, probe=probe)
		plotter.show(block=False)
		if hiker in probe:
			correct_orthant = probe
			break

		empty_adjacent_orthant = probe

	if empty_adjacent_orthant:
		plot_domino_2d_reduction(correct_orthant, empty_adjacent_orthant)
	else:
		plot_domino_2d_search(plotter, correct_orthant, hiker, drone)

if __name__ == "__main__":
	plotter = SquarePlotter()
	plot_domino_2d_search(plotter, Hypercube(Point.origin(2), 32), Point((13, 4)), Point.origin(2))
	# plot_orthants(1)
	# plot_orthants(2)
	# plot_orthants(3)
