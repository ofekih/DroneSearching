import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from square_utils import Point, Hypercube

class SquarePlotter:
    """
    A class for visualizing hypercubes and points in 3D space.
    Useful for debugging search algorithms.
    """

    def __init__(self, figsize=(10, 8)):
        """Initialize the plotter with a figure of the given size."""
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.colors = {
            'empty': 'lightgray',
            'candidate': 'lightblue',
            'probe': 'green',
            'hiker': 'red',
            'drone': 'blue',
            'search_area': 'black'
        }
        self.alpha = {
            'empty': 0.1,
            'candidate': 0.3,
            'probe': 0.5,
            'search_area': 0.1
        }

    def _get_cube_vertices(self, cube: Hypercube) -> np.ndarray:
        """Get the vertices of a 3D cube from its center and side length."""
        if cube.dimension > 3:
            raise ValueError("Can only visualize up to 3D cubes")

        # Get min and max corners
        min_corner = cube.min_corner
        max_corner = cube.max_corner

        # For dimensions less than 3, pad with zeros
        min_coords = list(min_corner.coordinates)
        max_coords = list(max_corner.coordinates)

        while len(min_coords) < 3:
            min_coords.append(0)
            max_coords.append(0)

        # Create all 8 vertices of the cube
        x_min, y_min, z_min = min_coords
        x_max, y_max, z_max = max_coords

        vertices = np.array([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max]
        ])

        return vertices

    def _get_cube_faces(self, vertices: np.ndarray) -> List[List[int]]:
        """Get the faces of a cube from its vertices."""
        faces = [
            [0, 1, 2, 3],  # bottom
            [4, 5, 6, 7],  # top
            [0, 1, 5, 4],  # front
            [2, 3, 7, 6],  # back
            [0, 3, 7, 4],  # left
            [1, 2, 6, 5]   # right
        ]
        return faces

    def plot_hypercube(self, cube: Hypercube, cube_type: str = 'search_area', label: Optional[str] = None):
        """
        Plot a hypercube with the specified type (determines color and transparency).

        Args:
            cube: The Hypercube to plot
            cube_type: One of 'empty', 'candidate', 'probe', or 'search_area'
            label: Optional label for the cube
        """
        if cube.dimension > 3:
            raise ValueError("Can only visualize up to 3D cubes")

        vertices = self._get_cube_vertices(cube)
        faces = self._get_cube_faces(vertices)

        # Create the 3D polygons
        face_vertices = [[vertices[idx] for idx in face] for face in faces]
        poly = Poly3DCollection(face_vertices, alpha=self.alpha.get(cube_type, 0.3))

        # Set the color
        color = self.colors.get(cube_type, 'gray')
        poly.set_facecolor(color)
        poly.set_edgecolor('black')

        # Add to the plot
        self.ax.add_collection3d(poly)

        # Add label if provided
        if label:
            center = cube.center.coordinates
            # Pad with zeros if dimension < 3
            center_coords = list(center)
            while len(center_coords) < 3:
                center_coords.append(0)
            self.ax.text(center_coords[0], center_coords[1], center_coords[2], label)

    def plot_point(self, point: Point, point_type: str = 'hiker', label: Optional[str] = None, size: int = 100):
        """
        Plot a point with the specified type (determines color).

        Args:
            point: The Point to plot
            point_type: One of 'hiker' or 'drone'
            label: Optional label for the point
            size: Size of the point marker
        """
        coords = list(point.coordinates)
        # Pad with zeros if dimension < 3
        while len(coords) < 3:
            coords.append(0)

        x, y, z = coords
        color = self.colors.get(point_type, 'black')
        self.ax.scatter(x, y, z, c=color, s=size, label=label or point_type)

        # Add label if provided
        if label:
            self.ax.text(x, y, z, label)

    def plot_search_state(self,
                          search_area: Hypercube,
                          hiker: Optional[Point] = None,
                          drone: Optional[Point] = None,
                          empty_regions: List[Hypercube] = [],
                          candidates: List[Hypercube] = [],
                          probe: Optional[Hypercube] = None,
                          title: str = "Search Visualization"):
        """
        Plot the current state of a search algorithm.

        Args:
            search_area: The overall search area
            hiker: The hiker's position
            drone: The drone's position
            empty_regions: List of hypercubes known to be empty
            candidates: List of candidate hypercubes that might contain the hiker
            probe: The current probe hypercube
            title: Title for the plot
        """
        # Clear previous plot
        self.ax.clear()

        # Plot the search area
        self.plot_hypercube(search_area, 'search_area', 'Search Area')

        # Plot empty regions
        if empty_regions:
            for i, region in enumerate(empty_regions):
                self.plot_hypercube(region, 'empty', f'Empty {i+1}')

        # Plot candidate regions
        if candidates:
            for i, candidate in enumerate(candidates):
                self.plot_hypercube(candidate, 'candidate', f'Candidate {i+1}')

        # Plot the probe
        if probe:
            self.plot_hypercube(probe, 'probe', 'Probe')

        # Plot the hiker and drone
        if hiker:
            self.plot_point(hiker, 'hiker', 'Hiker')
        if drone:
            self.plot_point(drone, 'drone', 'Drone')

        # Set title and labels
        self.ax.set_title(title)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Add a legend
        self.ax.legend()

        # Set equal aspect ratio
        self.ax.set_box_aspect([1, 1, 1])

        # Set limits based on search area
        min_corner = search_area.min_corner.coordinates
        max_corner = search_area.max_corner.coordinates

        # Pad with zeros if dimension < 3
        min_coords = list(min_corner)
        max_coords = list(max_corner)
        while len(min_coords) < 3:
            min_coords.append(0)
        while len(max_coords) < 3:
            max_coords.append(0)

        # Add some padding
        padding = search_area.side_length * 0.1
        self.ax.set_xlim(min_coords[0] - padding, max_coords[0] + padding)
        self.ax.set_ylim(min_coords[1] - padding, max_coords[1] + padding)
        self.ax.set_zlim(min_coords[2] - padding, max_coords[2] + padding)

    def show(self, block: bool=False, pause_time: float=5):
        """
        Display the plot.

        Args:
            block: If True, blocks execution until the window is closed.
                  If False, continues execution with the window still open.
            pause_time: Time to pause between plot updates
        """
        plt.tight_layout()
        plt.draw()
        plt.pause(pause_time)  # Small pause to update the figure
        if block:
            plt.show(block=True)

    def save(self, filename: str):
        """Save the plot to a file."""
        plt.tight_layout()
        plt.savefig(filename)
