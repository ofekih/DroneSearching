import random
import math
import matplotlib.pyplot as plt
from shapely.geometry import Point, MultiPolygon
from shapely.ops import unary_union
import time

# Create the unit circle
unit_circle = Point(0, 0).buffer(1.0)

def create_circle(center_x, center_y, radius):
    """Create a circle as a Shapely geometry"""
    return Point(center_x, center_y).buffer(radius)

def calculate_coverage(unit_circle, covering_circles):
    """Calculate the percentage of the unit circle covered by the circles"""
    if not covering_circles:
        return 0.0, None
    
    coverage = unary_union(covering_circles)
    intersection = coverage.intersection(unit_circle)
    coverage_percentage = intersection.area / unit_circle.area * 100
    return coverage_percentage, intersection

def create_circles_from_positions(positions, radii):
    """Create circle geometries from positions and radii"""
    return [create_circle(x, y, r) for (x, y), r in zip(positions, radii)]

def initial_solution(circle_radii):
    """Generate a smart initial solution with strategic placement"""
    positions = []
    
    # Place largest circle at center
    positions.append((0, 0))
    
    # Place remaining circles in a spiral pattern
    if len(circle_radii) > 1:
        # Calculate how many circles we can fit in the first ring
        first_circle_radius = circle_radii[0]
        second_circle_radius = circle_radii[1]
        # Distance from center for the second circle
        distance = first_circle_radius + second_circle_radius * 0.7  # Slight overlap
        
        remaining_circles = len(circle_radii) - 1
        circles_placed = 1
        
        # Place circles in concentric rings
        ring = 1
        while circles_placed < len(circle_radii):
            # Estimate number of circles that can fit in this ring
            if ring == 1:
                # First ring - place circles evenly around the center
                circles_in_ring = min(remaining_circles, 6)
                angle_step = 2.0 * math.pi / circles_in_ring
                
                for i in range(circles_in_ring):
                    angle = i * angle_step
                    x = distance * math.cos(angle)
                    y = distance * math.sin(angle)
                    positions.append((x, y))
                    circles_placed += 1
                    
            else:
                # Outer rings - place more circles with larger distance
                circles_in_ring = min(remaining_circles, 6 * ring)
                angle_step = 2.0 * math.pi / circles_in_ring
                ring_distance = distance * ring * 0.8  # Each ring is farther out
                
                for i in range(circles_in_ring):
                    angle = i * angle_step
                    x = ring_distance * math.cos(angle)
                    y = ring_distance * math.sin(angle)
                    positions.append((x, y))
                    circles_placed += 1
                    if circles_placed >= len(circle_radii):
                        break
            
            remaining_circles -= circles_in_ring
            ring += 1
    
    return positions

def neighbor_solution(positions, temperature, radii):
    """Generate a neighboring solution with controlled perturbation"""
    new_positions = []
    
    # Scale the perturbation by temperature (larger moves at higher temperatures)
    max_perturbation = 0.15 * temperature  # Adjusted perturbation scale
    
    for i, (x, y) in enumerate(positions):
        # Add random perturbation to position
        # Use smaller perturbations for larger circles
        scale_factor = min(0.5, radii[i])  # Adjust based on circle size
        perturbation = max_perturbation * scale_factor
        
        # Simple random perturbation
        dx = random.uniform(-perturbation, perturbation)
        dy = random.uniform(-perturbation, perturbation)
        
        # Keep circles reasonably close to the unit circle
        new_x = x + dx
        new_y = y + dy
        
        # Ensure the center of the circle stays within a reasonable distance of the origin
        # For optimal coverage, centers should generally be within the unit circle
        distance_from_center = math.sqrt(new_x**2 + new_y**2)
        max_allowed_distance = 1.0 - radii[i] * 0.5  # Centers should be inside or slightly outside
        
        if distance_from_center > max_allowed_distance:
            # Scale back to keep centers near or within the unit circle
            scale = max_allowed_distance / distance_from_center
            new_x *= scale
            new_y *= scale
            
        new_positions.append((new_x, new_y))
    
    return new_positions

def acceptance_probability(old_coverage, new_coverage, temperature):
    """Calculate probability of accepting a worse solution"""
    if new_coverage >= old_coverage:
        return 1.0  # Always accept better or equal solutions
    
    # Prevent division by zero or negative temperatures
    if temperature <= 0.001:
        return 0.0
        
    # Calculate acceptance probability based on how much worse the new solution is
    delta = new_coverage - old_coverage
    
    # Adjust the scale to make the probability more reasonable
    return math.exp(delta / (temperature * 3))

def simulated_annealing(circle_radii, initial_temp=5.0, cooling_rate=0.98, 
                        min_temp=0.01, iterations_per_temp=100, max_restarts=5, 
                        max_iterations=20000, no_improvement_limit=2000):
    """
    Simulated annealing algorithm to optimize circle placement
    
    Parameters:
    - circle_radii: List of radii for the circles
    - initial_temp: Starting temperature
    - cooling_rate: Rate at which temperature decreases
    - min_temp: Minimum temperature to stop the algorithm
    - iterations_per_temp: Number of iterations at each temperature level
    - max_restarts: Maximum number of restarts allowed
    - max_iterations: Maximum total iterations allowed
    - no_improvement_limit: Stop if no improvement after this many iterations
    """
    # Start with an initial solution
    current_positions = initial_solution(circle_radii)
    current_circles = create_circles_from_positions(current_positions, circle_radii)
    current_coverage, current_intersection = calculate_coverage(unit_circle, current_circles)
    
    # Keep track of the best solution found
    best_positions = current_positions.copy()
    best_coverage = current_coverage
    best_intersection = current_intersection
    best_circles = current_circles.copy()
    
    # Initialize temperature
    temperature = initial_temp
    original_temp = initial_temp
    
    # For tracking progress
    progress = []
    
    print(f"Initial coverage: {current_coverage:.2f}%")
    
    # Main simulated annealing loop
    iteration = 0
    stagnation_count = 0
    restart_count = 0
    last_improvement = 0
    
    while temperature > min_temp and iteration < max_iterations and best_coverage < 100.0 - 1e-6:
        improved_in_this_temp = False
        
        for _ in range(iterations_per_temp):
            iteration += 1
            if iteration >= max_iterations:
                break
                
            # Generate a neighboring solution
            new_positions = neighbor_solution(current_positions, temperature, circle_radii)
            new_circles = create_circles_from_positions(new_positions, circle_radii)
            new_coverage, new_intersection = calculate_coverage(unit_circle, new_circles)
            
            # Safety check - if coverage calculation failed
            if new_coverage is None or math.isnan(new_coverage):
                continue
                
            # Decide whether to accept the new solution
            p = acceptance_probability(current_coverage, new_coverage, temperature)
            if random.random() < p:
                current_positions = new_positions
                current_coverage = new_coverage
                current_intersection = new_intersection
                current_circles = new_circles
                
                # Update best solution if current is better
                if current_coverage > best_coverage + 0.01:  # Small threshold to avoid rounding issues
                    best_positions = current_positions.copy()
                    best_coverage = current_coverage
                    best_intersection = current_intersection
                    best_circles = current_circles.copy()
                    improved_in_this_temp = True
                    last_improvement = iteration
                    print(f"Iteration {iteration}, Temp: {temperature:.2f}, New best coverage: {best_coverage:.2f}%")
            
            # Track progress
            if iteration % 100 == 0:
                progress.append((iteration, current_coverage))
                
            # Check for long-term stagnation
            if iteration - last_improvement > no_improvement_limit:
                print(f"No improvement for {no_improvement_limit} iterations. Restarting.")
                break
        
        # Check for stagnation
        if not improved_in_this_temp:
            stagnation_count += 1
        else:
            stagnation_count = 0
        
        # If stagnating for too long, do a restart from the best solution
        if stagnation_count >= 5 or iteration - last_improvement > no_improvement_limit:
            restart_count += 1
            if restart_count >= max_restarts:
                print(f"Maximum restarts ({max_restarts}) reached. Stopping.")
                break
                
            print(f"Restarting from best solution (coverage: {best_coverage:.2f}%, restart {restart_count}/{max_restarts})")
            current_positions = best_positions.copy()
            current_circles = best_circles.copy()
            current_coverage = best_coverage
            current_intersection = best_intersection
            stagnation_count = 0
            last_improvement = iteration
            
            # Reset temperature to a higher value but not as high as initial
            temperature = original_temp * (0.7 ** restart_count)
        else:
            # Cool down
            temperature *= cooling_rate
            
        print(f"Temperature: {temperature:.2f}, Current coverage: {current_coverage:.2f}%, Best coverage: {best_coverage:.2f}%")
    
    # Final optimization - local search to fine-tune positions
    if best_coverage < 100.0 - 1e-6:
        print("Performing final local optimization...")
        best_positions, best_coverage, best_intersection = local_optimization(best_positions, circle_radii)
        best_circles = create_circles_from_positions(best_positions, circle_radii)
    
    print(f"Final best coverage: {best_coverage:.2f}%")
    print(f"Total iterations: {iteration}")
    
    return best_positions, best_circles, best_coverage, best_intersection, progress

def local_optimization(positions, radii, iterations=500):
    """Fine-tune positions with a local search"""
    best_positions = positions.copy()
    best_circles = create_circles_from_positions(best_positions, radii)
    best_coverage, best_intersection = calculate_coverage(unit_circle, best_circles)
    
    current_positions = best_positions.copy()
    current_coverage = best_coverage
    
    # Very small perturbations for fine-tuning
    perturbation = 0.01
    
    for i in range(iterations):
        # Try to improve each circle's position
        for j in range(len(radii)):
            # Original position
            orig_x, orig_y = current_positions[j]
            
            # Try small movements in different directions
            for dx, dy in [(perturbation, 0), (-perturbation, 0), (0, perturbation), (0, -perturbation)]:
                # Move one circle
                current_positions[j] = (orig_x + dx, orig_y + dy)
                
                # Check if it improves coverage
                circles = create_circles_from_positions(current_positions, radii)
                coverage, intersection = calculate_coverage(unit_circle, circles)
                
                if coverage > current_coverage:
                    current_coverage = coverage
                    # Keep this move
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_positions = current_positions.copy()
                        best_intersection = intersection
                        print(f"Local optimization: improved to {best_coverage:.2f}%")
                else:
                    # Revert the move
                    current_positions[j] = (orig_x, orig_y)
        
        # Reduce perturbation size over time
        if i % 100 == 0 and i > 0:
            perturbation *= 0.8
    
    return best_positions, best_coverage, best_intersection

def plot_coverage(unit_circle, covering_circles, intersection, coverage, title="Circle Coverage"):
    """Plot the unit circle, covering circles, and their intersection"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot unit circle
    x, y = unit_circle.exterior.xy
    ax.plot(x, y, 'k-', linewidth=2)
    
    # Plot covering circles
    for i, circle in enumerate(covering_circles):
        x, y = circle.exterior.xy
        ax.plot(x, y, 'b-', linewidth=1, alpha=0.5)
        
        # Plot circle centers
        center_x, center_y = circle.centroid.x, circle.centroid.y
        ax.plot(center_x, center_y, 'ro', markersize=3)
        ax.text(center_x, center_y, str(i+1), fontsize=8)
    
    # Plot intersection (covered area)
    if hasattr(intersection, 'geoms'):
        for geom in intersection.geoms:
            if hasattr(geom, 'exterior'):
                x, y = geom.exterior.xy
                ax.fill(x, y, 'g', alpha=0.3)
    else:
        if hasattr(intersection, 'exterior'):
            x, y = intersection.exterior.xy
            ax.fill(x, y, 'g', alpha=0.3)
    
    # Set plot properties
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    plt.title(f"{title}: {coverage:.2f}%")
    plt.show()

def plot_progress(progress):
    """Plot the progress of the simulated annealing algorithm"""
    if not progress:
        print("No progress data to plot.")
        return
        
    iterations, coverages = zip(*progress)
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, coverages)
    plt.xlabel('Iteration')
    plt.ylabel('Coverage (%)')
    plt.title('Simulated Annealing Progress')
    plt.grid(True)
    plt.show()

def main():
    # Example set of circle radii
    # circle_radii = [0.5, 0.4, 0.3, 0.3, 0.25, 0.25, 0.2, 0.2, 0.15]
    p = 0.8
    circle_radii = [p ** k for k in range(1, 100)]

    # Sort radii in descending order (often helps with placement)
    circle_radii.sort(reverse=True)
    
    print(f"Optimizing placement for {len(circle_radii)} circles with radii: {circle_radii}")
    
    # Run simulated annealing
    start_time = time.time()
    best_positions, best_circles, best_coverage, best_intersection, progress = simulated_annealing(
        circle_radii,
        initial_temp=5.0,
        cooling_rate=0.98,  # Slower cooling
        min_temp=0.01,
        iterations_per_temp=100,
        max_restarts=50,
        max_iterations=200000,  # More iterations
        no_improvement_limit=2000
    )
    end_time = time.time()
    
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    print(f"Best coverage achieved: {best_coverage:.2f}%")
    
    # Print the optimized positions
    print("\nOptimized circle positions:")
    for i, ((x, y), r) in enumerate(zip(best_positions, circle_radii)):
        print(f"Circle {i+1}: center=({x:.4f}, {y:.4f}), radius={r:.4f}")
    
    # Plot the results
    plot_coverage(unit_circle, best_circles, best_intersection, best_coverage, 
                 title="Optimized Circle Coverage")
    
    # Plot the progress
    if progress:
        plot_progress(progress)

if __name__ == "__main__":
    main()