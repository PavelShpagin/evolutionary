# Genetic Algorithm for TSP

A clean implementation of a Genetic Algorithm (GA) to solve the Traveling Salesman Problem (TSP).

## What is TSP?

The Traveling Salesman Problem: Given a list of cities, find the shortest route that visits each city exactly once and returns to the start.

## What is a Genetic Algorithm?

A Genetic Algorithm is inspired by evolution in nature. It works like this:

1. **Population**: Start with a random population of solutions (tours)
2. **Selection**: Select the best solutions to become parents
3. **Crossover**: Combine two parent tours to create children
4. **Mutation**: Randomly modify children to add diversity
5. **Elitism**: Keep the best solutions from each generation
6. **Repeat**: Do this for many generations until convergence

## GA Pseudoalgorithm

```
FUNCTION GeneticAlgorithm(tsp_instance, config):
    population = create_random_tours(population_size)
    best_tour = None
    best_distance = infinity
    
    FOR generation = 1 TO max_generations:
        # Evaluate all tours
        FOR EACH tour IN population:
            distance = calculate_tour_distance(tour)
        
        # Update best solution
        IF min(distances) < best_distance:
            best_tour = tour_with_min_distance
            best_distance = min(distances)
        
        # Keep elite (best solutions)
        elite = select_top_k(population, elitism_count)
        
        # Create new population
        new_population = elite
        
        WHILE size(new_population) < population_size:
            # Select parents
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            
            # Crossover
            IF random() < crossover_rate:
                child1, child2 = ox_crossover(parent1, parent2)
            ELSE:
                child1, child2 = parent1, parent2
            
            # Mutation
            IF random() < mutation_rate:
                child1 = mutate(child1)
            IF random() < mutation_rate:
                child2 = mutate(child2)
            
            new_population.add(child1, child2)
        
        population = new_population
        
        # Early stopping
        IF no_improvement_for(patience) generations:
            BREAK
    
    RETURN best_tour, best_distance
```

## How OX Crossover Works

Order Crossover (OX) preserves the relative order of cities:

```
FUNCTION ox_crossover(parent1, parent2):
    # Select random segment
    start, end = random_range(0, length)
    
    # Copy segment from parent1 to child
    child[start:end] = parent1[start:end]
    
    # Use set for O(1) lookup of cities already in child
    in_child = set(child[start:end])
    
    # Fill remaining positions from parent2 (in order)
    child_idx = 0
    FOR city IN parent2:
        IF city NOT IN in_child:
            # Find next empty position
            WHILE child[child_idx] is not empty:
                child_idx++
            child[child_idx] = city
    
    RETURN child
```

**Example:**
```
Parent1: [0, 1, 2, 3, 4, 5, 6, 7]
Parent2: [7, 6, 5, 4, 3, 2, 1, 0]
Segment: positions 2-5

Step 1: Copy segment from Parent1
Child:   [-, -, 2, 3, 4, -, -, -]

Step 2: Fill from Parent2 (skip 2,3,4)
Child:   [7, 6, 2, 3, 4, 5, 1, 0]
```

## Mutation Types

- **Swap**: Exchange two random cities
- **Inversion**: Reverse a random segment of the tour
- **Insert**: Remove a city and insert it at a different position

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install
pip install -e .
```

## Three Magic Commands

### 1. Generate Test Data
```bash
python -m ga_tsp.cli make-data
```
Creates `data/small_12.csv` (12 cities) and `data/medium_50.csv` (50 cities)

### 2. Run Brute Force (Optimal Solution)
```bash
python -m ga_tsp.cli bruteforce data/small_12.csv
```
Finds the optimal solution by trying all permutations (only works for ≤12 cities)

### 3. Compare GA vs Optimal
```bash
python -m ga_tsp.cli compare data/small_12.csv
```
Runs both methods and creates side-by-side comparison visualization

## All Commands

### `make-data`
Generate test datasets
```bash
python -m ga_tsp.cli make-data
```

### `bruteforce`
Find optimal solution (≤12 cities only)
```bash
python -m ga_tsp.cli bruteforce data/small_12.csv
```

### `solve`
Solve with GA
```bash
python -m ga_tsp.cli solve data/small_12.csv
python -m ga_tsp.cli solve data/medium_50.csv --generations 1500
```

Options:
- `--population-size N` - Population size (default: 200)
- `--generations N` - Max generations (default: 1000)
- `--mutation {swap,inversion,insert}` - Mutation type (default: inversion)
- `--mutation-rate RATE` - Mutation probability (default: 0.2)
- `--selection {tournament,roulette}` - Selection method (default: tournament)
- `--elitism N` - Number of elite individuals (default: 5)
- `--seed N` - Random seed (default: 42)

### `compare`
Compare GA vs optimal (≤12 cities only)
```bash
python -m ga_tsp.cli compare data/small_12.csv
```
Creates side-by-side visualization and convergence plot with optimal line

### `grid-search`
Parameter tuning
```bash
python -m ga_tsp.cli grid-search data/medium_50.csv --pop 100,200,300 --mutation inversion,swap --trials 3
```

## Project Structure

```
ga_tsp/
├── src/ga_tsp/
│   ├── cli.py          # Command-line interface
│   ├── ga.py           # GA engine
│   ├── operators.py    # OX crossover and mutations
│   ├── tsp.py          # TSP data structures
│   ├── brute_force.py  # Optimal solver
│   ├── baseline.py     # Nearest neighbor heuristic
│   ├── viz.py          # Visualization functions
│   └── utils.py        # Utilities
├── tests/              # Test suite
├── data/               # Datasets
└── outputs/            # Results (plots, CSVs)
```

## File Explanations

### Core Source Files

1. **`tsp.py`** - TSP data structures
   - `TSPInstance`: Holds city coordinates and distance matrix
   - `compute_distance_matrix()`: Euclidean distances
   - `validate_tour()`: Check if tour is valid
   - CSV load/save functions

2. **`operators.py`** - Genetic operators
   - `ox_crossover()`: Order Crossover (uses sets for O(1) lookup)
   - `swap_mutation()`: Swap two cities
   - `inversion_mutation()`: Reverse a segment
   - `insert_mutation()`: Move a city

3. **`ga.py`** - Genetic Algorithm engine
   - `GAConfig`: Configuration (population size, generations, etc.)
   - `GAResult`: Results (best tour, convergence history)
   - `GeneticAlgorithm`: Main GA with selection, crossover, mutation

4. **`brute_force.py`** - Optimal solver for small instances

5. **`baseline.py`** - Heuristic algorithms
   - `nearest_neighbor()`: Greedy nearest city
   - `two_opt_improvement()`: Local search optimization

6. **`viz.py`** - Visualization
   - `plot_tour()`: Single tour visualization
   - `plot_tour_comparison()`: Side-by-side optimal vs GA
   - `plot_convergence()`: Fitness over generations

7. **`cli.py`** - Command-line interface with all commands

8. **`utils.py`** - Helper functions
   - `generate_random_points()`: Create random cities
   - `save_tour_csv()`: Save tour to file

### Test Files

All test files are needed and verify different aspects:

1. **`test_operators.py`** - Tests genetic operators
   - Verifies OX crossover creates valid permutations
   - Checks all mutations preserve city sets
   - Tests edge cases and stress testing

2. **`test_tsp_core.py`** - Tests TSP fundamentals
   - Distance matrix symmetry
   - Tour validation (correct/incorrect)
   - Distance calculations
   - CSV I/O

3. **`test_integration_small.py`** - Integration tests
   - GA finds near-optimal solutions (≤1% deviation)
   - Convergence behavior
   - Different mutation operators
   - Stagnation and early stopping
   - Reproducibility with seeds

4. **`test_baseline_vs_ga.py`** - Comparative tests
   - GA vs nearest neighbor baseline
   - Consistency across multiple runs
   - 2-opt improvement verification

## Testing

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test file
pytest tests/test_operators.py -v
```

## Output Files

After running commands, check `outputs/` directory:
- `comparison_*.png` - Side-by-side optimal vs GA tours
- `convergence_*.png` - Fitness over generations (with optimal line if available)
- `best_tour_*.csv` - Tour order and distance
- `optimal_tour_*.csv` - Optimal tour from brute force

## Examples

```bash
# Generate data
python -m ga_tsp.cli make-data

# Compare on small instance
python -m ga_tsp.cli compare data/small_12.csv

# Solve medium instance with custom parameters
python -m ga_tsp.cli solve data/medium_50.csv --population-size 250 --generations 2000 --seed 123

# Parameter tuning
python -m ga_tsp.cli grid-search data/medium_50.csv --pop 150,200,250 --mutation swap,inversion,insert --trials 5
```

## Tips

- Use `--seed` for reproducible results
- Larger `--population-size` explores more solutions but is slower
- More `--generations` allows longer search
- `inversion` mutation works well for TSP
- `compare` command is great for understanding GA behavior on small instances
- All datasets are reusable - generate once, experiment many times

## License

See LICENSE file.
