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

## Genetic Algorithm - Detailed Explanation

### Core Concept

The Genetic Algorithm mimics natural evolution to find good solutions:

1. **Population** = Many different tours (like species in nature)
2. **Fitness** = Tour distance (shorter is better, like survival fitness)
3. **Selection** = Better tours get to reproduce (survival of the fittest)
4. **Crossover** = Combine good tours to make potentially better ones (like sexual reproduction)
5. **Mutation** = Random changes to avoid getting stuck (like genetic mutations)
6. **Evolution** = Over generations, tours get progressively better

### Step-by-Step Process

#### 1. Initialization (Generation 0)
```
Create 200 random tours:
Tour 1: [5, 2, 8, 1, 4, ...] → Distance: 890
Tour 2: [3, 7, 1, 9, 2, ...] → Distance: 920
Tour 3: [1, 4, 5, 2, 8, ...] → Distance: 875
...
Tour 200: [9, 0, 3, 6, 1, ...] → Distance: 910

Best initial: 875 (Tour 3)
Average: 900
```

#### 2. Evaluation
Calculate distance for every tour:
```python
distance = sum of all edges + edge back to start
```

#### 3. Selection (Tournament)
Pick parents for reproduction:
```
1. Randomly select 4 tours
2. Choose the best one as parent
3. Repeat to get second parent

Example:
Random 4: [Tour 5, Tour 12, Tour 87, Tour 34]
Distances: [920, 880, 860, 900]
Winner: Tour 87 (distance 860) → becomes parent1

This ensures: good tours have more children, bad tours gradually die out
```

#### 4. Crossover (OX - Order Crossover)
Combine two good tours to make children:

```
Parent1: [0, 1, 2, 3, 4, 5, 6, 7, 8]  (distance: 850)
Parent2: [8, 7, 6, 5, 4, 3, 2, 1, 0]  (distance: 860)

Step 1: Select random segment (e.g., positions 3-6)
Segment from Parent1: [3, 4, 5, 6]

Step 2: Copy segment to child
Child: [-, -, -, 3, 4, 5, 6, -, -]

Step 3: Fill remaining from Parent2 (in order), skip 3,4,5,6
Parent2 order: [8, 7, 6, 5, 4, 3, 2, 1, 0]
After skipping 3,4,5,6: [8, 7, 2, 1, 0]

Step 4: Place in empty positions (left to right)
Child: [8, 7, 2, 3, 4, 5, 6, 1, 0]  (potentially better!)

Why this works: Preserves good subsequences from both parents
- [3,4,5,6] from Parent1 might be a good local route
- Relative order from Parent2 fills the gaps
```

#### 5. Mutation (Inversion)
Add random changes to escape local optima:

```
Before: [0, 1, 2, 3, 4, 5, 6, 7, 8]

Step 1: Pick random segment (positions 2-5)
Segment: [2, 3, 4, 5]

Step 2: Reverse it
Reversed: [5, 4, 3, 2]

After: [0, 1, 5, 4, 3, 2, 6, 7, 8]

Why this works: 
- Untangles crossed paths
- Explores nearby solutions
- Prevents premature convergence
```

#### 6. Elitism
Always keep the best solutions:
```
Before creating new generation:
1. Find top 5 tours
2. Copy them unchanged to next generation
3. Fill rest with crossover + mutation

Why: Never lose the best solution found so far
```

#### 7. Repeat
```
Generation 1:    Best = 875, Avg = 900
Generation 10:   Best = 780, Avg = 820  (improving fast!)
Generation 50:   Best = 690, Avg = 720
Generation 100:  Best = 660, Avg = 680
Generation 200:  Best = 652, Avg = 665
Generation 346:  Best = 651.6 (no improvement for 150 gens → stop)
```

### Why It Works

**Exploration vs Exploitation Balance:**
- **Crossover** exploits good solutions (combines what works)
- **Mutation** explores new solutions (tries random changes)
- **Selection** focuses on promising areas (better tours reproduce more)
- **Elitism** preserves discoveries (never lose the best)

**Population Diversity:**
- 200 tours explore 200 different parts of solution space
- Crossover mixes solutions (like genetic recombination)
- Mutation introduces novelty (prevents getting stuck)
- Over time: bad solutions die out, good solutions dominate

**Convergence:**
```
Early generations: Wild diversity, big improvements
  Population: [890, 875, 920, 865, ...]  (varied)
  
Mid generations: Less diversity, steady improvements  
  Population: [690, 695, 688, 692, ...]  (converging)
  
Late generations: High similarity, small improvements
  Population: [652, 651.6, 652.3, 651.8, ...]  (converged)
  
Stop: No improvement for 150 generations (locally optimal)
```

### Key Parameters Explained

**Population Size (200):**
- Larger = More exploration, slower per generation
- Smaller = Less exploration, faster per generation
- 200 is good balance for TSP

**Crossover Rate (0.9 = 90%):**
- High rate = More exploitation of good solutions
- Low rate = More random search
- 0.9 means 90% of children come from combining parents

**Mutation Rate (0.2 = 20%):**
- Too high = Random search (loses good solutions)
- Too low = Premature convergence (gets stuck)
- 0.2 provides good exploration without destroying good tours

**Elitism (5):**
- Always keep 5 best tours unchanged
- Ensures monotonic improvement (best never gets worse)
- Small number to preserve diversity

**Stagnation Patience (150):**
- Stop if no improvement for 150 generations
- Saves time when locally optimal is reached
- Typically stops at 30-50% of max generations

### Real Example: 12 Cities

```
Generation 0 (Random initialization):
Best: [5,2,8,1,4,0,7,3,9,6,11,10] = 420.5
Avg: 480.2

Generation 1-10 (Rapid improvement):
- Tournament selection picks good tours
- OX crossover combines them
- Lots of improvement from combining random pieces
Best: 340.2  (-19%)

Generation 11-50 (Steady improvement):
- Population converging on good regions
- Crossover refining solutions
- Mutation finding local improvements
Best: 310.8  (-9%)

Generation 51-150 (Fine-tuning):
- Most tours very similar
- Mutation doing most work
- Small incremental improvements
Best: 298.06 (OPTIMAL!)

Generation 151-163 (Stagnation):
- No improvement
- All tours near optimal
- Early stop triggered
Final: 298.06 (0% deviation from optimal)
```

### Why GA Beats Brute Force

**Brute Force:**
```
12 cities = 479,001,600 permutations
15 cities = 1,307,674,368,000 permutations
20 cities = 2,432,902,008,176,640,000 permutations
50 cities = 3×10^64 permutations (more than atoms in universe!)
```

**Genetic Algorithm:**
```
12 cities: Evaluates ~30,000 tours (0.006% of all) → finds optimal
50 cities: Evaluates ~70,000 tours (0.000...% of all) → finds near-optimal
```

The secret: GA learns from evaluations, brute force doesn't!

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
