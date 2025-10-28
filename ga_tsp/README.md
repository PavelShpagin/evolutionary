# Genetic Algorithm for Traveling Salesman Problem

A comprehensive implementation of a Genetic Algorithm (GA) for solving the Euclidean Traveling Salesman Problem (TSP). This project demonstrates how evolutionary algorithms can find near-optimal solutions for combinatorial optimization problems.

## Overview

The Traveling Salesman Problem asks: Given a list of cities and the distances between them, what is the shortest possible route that visits each city exactly once and returns to the starting city?

This implementation uses a Genetic Algorithm, which:
- Represents solutions as permutations of cities
- Uses selection, crossover, and mutation operators to evolve populations
- Applies elitism to preserve the best solutions
- Supports multiple crossover strategies (PMX, OX) and mutation types (swap, inversion, insert)

## Features

- Multiple genetic operators (PMX/OX crossover, swap/inversion/insert mutations)
- Tournament and roulette wheel selection
- Elitism and adaptive mutation
- Baseline algorithms (nearest neighbor, 2-opt improvement)
- Visualization of tours and convergence
- Comprehensive test suite with brute-force validation
- Command-line interface for easy experimentation
- Grid search for parameter optimization

## Setup

### Requirements

- Python 3.10 or higher
- pip

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development (includes pytest):

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Generate Datasets

```bash
python -m ga_tsp.cli make-data
```

This creates:
- `data/small_12.csv` - 12 cities (for validation against brute-force optimal)
- `data/medium_50.csv` - 50 cities (for realistic demonstrations)

### 2. Solve a TSP Instance

```bash
python -m ga_tsp.cli solve data/small_12.csv \
    --population-size 200 \
    --generations 1000 \
    --crossover pmx \
    --mutation inversion \
    --seed 42
```

This will:
- Run the genetic algorithm
- Compare against nearest neighbor baseline
- Save the best tour visualization to `outputs/best_tour_small_12.png`
- Save convergence plot to `outputs/convergence_small_12.png`
- Save tour data to `outputs/best_tour_small_12.csv`

### 3. Run Tests

```bash
pytest -v
```

Or for a quick summary:

```bash
pytest -q
```

## Command-Line Interface

### Generate Data

```bash
python -m ga_tsp.cli make-data [--output-dir DIR]
```

### Solve TSP

```bash
python -m ga_tsp.cli solve INPUT_FILE [OPTIONS]
```

Options:
- `--population-size N` - Population size (default: 200)
- `--generations N` - Maximum generations (default: 1000)
- `--selection {tournament,roulette}` - Selection method (default: tournament)
- `--tournament-k N` - Tournament size (default: 4)
- `--crossover {pmx,ox}` - Crossover operator (default: pmx)
- `--crossover-rate RATE` - Crossover probability (default: 0.9)
- `--mutation {swap,inversion,insert}` - Mutation operator (default: inversion)
- `--mutation-rate RATE` - Mutation probability (default: 0.2)
- `--elitism N` - Number of elite individuals (default: 5)
- `--stagnation-patience N` - Stop after N generations without improvement (default: 150)
- `--seed N` - Random seed for reproducibility (default: 42)
- `--adaptive-mutation` - Enable adaptive mutation rate
- `--output-dir DIR` - Output directory (default: outputs)

### Grid Search

```bash
python -m ga_tsp.cli grid-search INPUT_FILE \
    --pop 100,200,300 \
    --mutation inversion,swap \
    --crossover pmx,ox \
    --trials 5
```

## Usage Examples

### Small Instance (Optimal Validation)

```bash
python -m ga_tsp.cli solve data/small_12.csv \
    --population-size 200 \
    --generations 1200 \
    --crossover pmx \
    --mutation inversion \
    --seed 42
```

For small instances (≤10 cities), the program computes the optimal solution via brute force and reports the deviation.

### Medium Instance

```bash
python -m ga_tsp.cli solve data/medium_50.csv \
    --population-size 250 \
    --generations 1500 \
    --crossover pmx \
    --mutation inversion \
    --seed 42
```

### With Adaptive Mutation

```bash
python -m ga_tsp.cli solve data/medium_50.csv \
    --population-size 200 \
    --generations 2000 \
    --adaptive-mutation \
    --seed 42
```

### Grid Search Example

```bash
python -m ga_tsp.cli grid-search data/medium_50.csv \
    --pop 150,200,250 \
    --mutation inversion,swap,insert \
    --crossover pmx,ox \
    --generations 800 \
    --trials 3 \
    --seed 42
```

## Architecture

### Module Overview

- `tsp.py` - Core TSP data structures and distance calculations
- `operators.py` - Genetic operators (PMX/OX crossover, swap/inversion/insert mutations)
- `ga.py` - Main genetic algorithm engine with selection and evolution loop
- `baseline.py` - Baseline algorithms (nearest neighbor, 2-opt)
- `brute_force.py` - Brute-force optimal solver for small instances
- `viz.py` - Visualization functions for tours and convergence
- `utils.py` - Utility functions for data generation and file I/O
- `cli.py` - Command-line interface

### Genetic Operators

**Crossover:**
- **PMX (Partially Mapped Crossover)**: Preserves relative ordering by mapping positions between parents
- **OX (Order Crossover)**: Preserves absolute ordering by copying a segment and filling remaining positions

**Mutation:**
- **Swap**: Exchanges two random cities
- **Inversion**: Reverses a random subsequence
- **Insert**: Removes a city and inserts it at a different position

**Selection:**
- **Tournament**: Selects best from k random individuals
- **Roulette**: Probabilistic selection based on fitness

## Testing

The test suite validates:

1. **Operator correctness** (`test_operators.py`)
   - PMX and OX produce valid permutations
   - Mutations preserve the city set

2. **TSP core functionality** (`test_tsp_core.py`)
   - Distance matrix symmetry
   - Tour validation
   - CSV I/O

3. **Small instance optimality** (`test_integration_small.py`)
   - GA finds optimal or near-optimal solutions (≤1% deviation)
   - Convergence behavior
   - Reproducibility with fixed seeds

4. **GA vs baseline comparison** (`test_baseline_vs_ga.py`)
   - GA outperforms or matches nearest neighbor
   - Consistent quality across multiple runs

Run all tests:

```bash
pytest -v
```

Run specific test file:

```bash
pytest tests/test_operators.py -v
```

## Reproducing Results

All experiments are reproducible using fixed random seeds:

```bash
python -m ga_tsp.cli solve data/small_12.csv --seed 42
```

The same seed will produce identical results across runs.

## Output Files

After running the solver, you'll find in the `outputs/` directory:

- `best_tour_<name>.png` - Visualization of the best tour found
- `convergence_<name>.png` - Plot showing fitness over generations
- `best_tour_<name>.csv` - Tour order and total distance

## Performance

Typical results on the provided datasets:

- **small_12.csv** (12 cities): GA finds optimal or near-optimal (≤1% deviation) in ~500 generations
- **medium_50.csv** (50 cities): GA improves 5-15% over nearest neighbor baseline in ~1000 generations

## Extending the Project

### Add New Mutation Operator

Edit `src/ga_tsp/operators.py`:

```python
def my_mutation(tour, rng):
    # Your implementation
    return mutated_tour

MUTATION_OPS['my_mutation'] = my_mutation
```

### Add Local Search

Implement 2-opt or 3-opt in `baseline.py` and apply it to GA solutions:

```python
from ga_tsp.baseline import two_opt_improvement

# After GA evolution
improved_tour, improved_dist = two_opt_improvement(tsp, result.best_tour)
```

### Parallel Evaluation

Modify `ga.py` to use multiprocessing for fitness evaluation:

```python
from multiprocessing import Pool

def evaluate_population_parallel(self, population):
    with Pool() as pool:
        fitness = pool.map(self.tsp.evaluate, population)
    return np.array(fitness)
```

## References

- Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning
- Larrañaga, P. et al. (1999). Genetic Algorithms for the Travelling Salesman Problem: A Review of Representations and Operators

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please ensure:
- All tests pass (`pytest -q`)
- Code is formatted and documented
- New features include tests
