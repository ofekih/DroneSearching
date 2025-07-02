# Marco Polo Problem: Geometric Localization Algorithms

This repository contains the supplemental code for "The Marco Polo Problem: A Combinatorial Approach to Geometric Localization" implementing various probe-based search algorithms for geometric localization. The algorithms are designed to efficiently locate points of interest (POIs) using circular probes with binary responses.

## Overview

This repository implements the algorithms described in "The Marco Polo Problem: A Combinatorial Approach to Geometric Localization." The Marco Polo problem is inspired by the children's game and addresses geometric localization using probe-based searching.

In this problem, a mobile search point (∆) starts at the origin and must locate one or more points of interest (POIs) within distance n using circular probes of specified radius d. The search algorithm learns only whether there is a POI within the probed area (binary response), without directional or distance information.

The code implements 8 different algorithms ranging from simple hexagonal tilings to sophisticated optimization-based approaches. Each algorithm minimizes different metrics:
- **$P(n)$**: Number of probes issued
- **$D(n)$**: Total distance traveled by the search point
- **$R_{\max}$**: Maximum number of POI responses

The algorithms demonstrate various trade-offs between probe efficiency and travel distance, with applications to search-and-rescue operations, wildlife tracking, and sensor network localization.

## Setup

### Creating a Virtual Environment

1. Create a Python virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
   - On Linux/macOS:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Algorithms

### Basic Usage

To run any algorithm (1-6), use:
```bash
python algorithms.py --algorithm <1-6>
```

### Algorithm Descriptions

1. **Algorithm 1: Hexagonal Algorithm** *(Hexagonal)*
   - Uses a tiling of the search area with 7 hexagons of radius $n/2$
   - Probes 6 of the 7 hexagons with radius-$n/2$ probes (POI must be in last if others fail)
   - Worst-case: $P(n) \leq 6\lceil\log n\rceil$ probes, $D(n) \leq 10.39n$ distance, $R_{\max} \leq \lceil\log n\rceil$ responses

2. **Algorithm 2: Modified Hexagonal Algorithm** *(Hexagonal)*
   - First probes upper two quadrants with radius $n/\sqrt{2}$ probes (eliminating 3 hexagons)
   - Then probes 3 of the remaining 4 hexagons as in Algorithm 1
   - Better trade-off: $P(n) \leq 5\lceil\log n\rceil$ probes, $D(n) \leq 8.81n$ distance, $R_{\max} \leq 2\lceil\log n\rceil$ responses
3. **Algorithm 3: Progressive Chord-Based Shrinking** *(Chord-Based)*
   - Places probe diameters as chords of the search circle in monotonic counterclockwise order
   - Uses progressively shrinking probes with $\rho_1 \approx 0.844$ to avoid uncovered areas
   - Performance: $P(n) < 4.08\log n$ probes, $D(n) \leq 6.95n$ distance

4. **Algorithm 4: Reordered Chord Placement** *(Chord-Based)*
   - Non-monotonic version of Algorithm 3 with optimized probe placement
   - Places two largest probes side by side, alternates remaining probes to minimize overlap
   - Improved performance: $P(n) < 3.54\log n$ probes, $D(n) \leq 9.31n$ distance

5. **Algorithm 5: Central + Chords** *(Higher-Count Monotonic-Path)*
   - Begins with one large central probe, then places remaining probes along perimeter
   - Uses chord-based placement for up to 8 probes per recursive level
   - Performance: $P(n) < 3.83\log n$ probes, $D(n) \leq 6.72n$ distance

6. **Algorithm 6: Central + Optimized Chords** *(Higher-Count Monotonic-Path)*
   - Advanced version of Algorithm 5 with geometric optimization for probe positioning
   - Balances coverage rate of inner and outer circumferences for optimal probe placement
   - Best distance performance: $P(n) < 3.34\log n$ probes, $D(n) \leq 6.02n$ distance

7. **Algorithm 7: Darting Non-Monotonic (Modified Algorithm 4)** *(Darting)*
   - Starts with Algorithm 4 (minus final probe), then greedily fills gaps
   - Uses computer-assisted probe placement to efficiently cover search area
   - Performance: $P(n) < 2.93\log n$ probes, $D(n) \leq 25.8n$ distance

8. **Algorithm 8: Differential Evolution Optimization** *(Darting)*
   - Uses differential evolution algorithm to optimize placement of initial 6 probes
   - Applies greedy gap-filling method for remaining probes
   - Best probe performance: $P(n) < 2.53\log n$ probes, $D(n) \leq 45.4n$ distance

### Performance Recommendations

For **Algorithms 7-8**, it is recommended to use lower precision settings (1 or 2) to avoid excessive computation time:

```bash
python algorithms.py --algorithm 7 --precision 2
python algorithms.py --algorithm 8 --precision 1
```

### Precision Settings

The `--precision` parameter controls the decimal precision for calculations. The actual computational precision used internally is $2x + 4$ where $x$ is the input precision value.

- Default precision: 5
- Minimum precision: 1
- For Algorithms 7-8: Use precision 1-2 for reasonable execution time

### Additional Options

- `--find-all`: Use alternative radius calculation formula $p^{(k+1)/2}$ instead of $p^k$
- `--debug`: Enable debug output to see intermediate steps
- `--precision <n>`: Set calculation precision (minimum 1)

### Examples

```bash
# Run Algorithm 3 with default settings
python algorithms.py --algorithm 3

# Run Algorithm 7 with low precision for faster execution
python algorithms.py --algorithm 7 --precision 1

# Run Algorithm 4 with alternative radius calculation
python algorithms.py --algorithm 4 --find-all

# Run Algorithm 6 with debug output
python algorithms.py --algorithm 6 --debug --precision 3
```

## Output

Each algorithm outputs:
- `p`: The optimal parameter value found ($\rho_1$ for progressive shrinking algorithms)
- `c`: The efficiency coefficient (probes per $\log n$)
- `ct`: The total distance traveled by the search point
- CPU Time: Execution time in seconds
- A list of probe positions and radii for each recursive level
- A visualization plot showing the probe placement and search pattern

The algorithms demonstrate the theoretical trade-offs between:
- **Probe efficiency**: Algorithms 7-8 achieve the best probe counts ($2.53$-$2.93\log n$)
- **Distance efficiency**: Algorithms 5-6 minimize travel distance ($6.02$-$6.72n$)
- **Response efficiency**: Algorithm 1 minimizes POI responses ($\leq\lceil\log n\rceil$)

## File Structure

```
├── algorithms.py           # Main algorithm implementations and CLI
├── requirements.txt        # Python dependencies
├── src/
│   ├── geometry_types.py     # Core geometric data structures
│   ├── geometry_algorithms.py # Geometric utility functions
│   ├── algorithm_utils.py    # Binary search and optimization utilities
│   ├── algorithm_plot.py     # Visualization functions
│   └── __pycache__/         # Compiled Python files
└── data/                   # Data storage directory
```

## Dependencies

The main dependencies include:
- `numpy`: Numerical computations
- `scipy`: Optimization algorithms
- `matplotlib`: Visualization
- `shapely`: Geometric operations
- `pandas`: Data handling
- `tqdm`: Progress bars

See `requirements.txt` for complete dependency list with versions.

## Research Context

This code implements the algorithms described in "The Marco Polo Problem: A Combinatorial Approach to Geometric Localization" (CCCG 2025). The research introduces the Marco Polo problem as a combinatorial approach to geometric localization, motivated by search-and-rescue scenarios where a searcher must locate POIs using only binary probe responses.

The algorithms provide theoretical bounds for the number of probes required:
- **Lower bound**: $2.4\log n$ probes for progressive shrinking algorithms
- **Best upper bound**: $2.53\log n$ probes (Algorithm 8)
- **Best distance performance**: $6.02n$ total distance (Algorithm 6)

Key theoretical contributions include computer-assisted proofs for probe placement optimization and analysis of trade-offs between probe count, distance traveled, and POI response limits. The work extends to multi-POI scenarios with $O(\log k)$-competitive traveling salesperson solutions.

## Notes

- Algorithms 1-2 use fixed, predetermined solutions
- Algorithms 3-6 use binary search to find optimal parameters
- Algorithms 7-8 are computationally intensive and may require significant time
- The precision setting affects both accuracy and computation time
- Higher precision values result in more accurate but slower computations
