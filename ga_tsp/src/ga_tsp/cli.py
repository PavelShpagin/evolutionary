import argparse
import sys
from pathlib import Path
import numpy as np
from typing import List
import itertools

from .tsp import TSPInstance
from .ga import GeneticAlgorithm, GAConfig
from .baseline import nearest_neighbor
from .brute_force import brute_force_tsp
from .viz import plot_tour, plot_convergence
from .utils import generate_random_points, save_tour_csv


def make_data_command(args):
    print("Generating datasets...")
    
    data_dir = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    small_points = generate_random_points(12, seed=42, bounds=(0, 100))
    small_tsp = TSPInstance(small_points)
    small_path = data_dir / "small_12.csv"
    small_tsp.to_csv(str(small_path))
    print(f"Created {small_path} with 12 cities")
    
    medium_points = generate_random_points(50, seed=7, bounds=(0, 100))
    medium_tsp = TSPInstance(medium_points)
    medium_path = data_dir / "medium_50.csv"
    medium_tsp.to_csv(str(medium_path))
    print(f"Created {medium_path} with 50 cities")
    
    print("Dataset generation complete.")


def solve_command(args):
    print(f"Loading TSP instance from {args.input}...")
    tsp = TSPInstance.from_csv(args.input)
    print(f"Loaded {tsp.n_cities} cities")
    
    print("\nRunning baseline (Nearest Neighbor)...")
    baseline_tour, baseline_distance = nearest_neighbor(tsp, start=0)
    print(f"Baseline distance: {baseline_distance:.4f}")
    
    config = GAConfig(
        population_size=args.population_size,
        generations=args.generations,
        selection=args.selection,
        tournament_k=args.tournament_k,
        crossover_rate=args.crossover_rate,
        mutation=args.mutation,
        mutation_rate=args.mutation_rate,
        elitism=args.elitism,
        stagnation_patience=args.stagnation_patience,
        seed=args.seed,
        adaptive_mutation=args.adaptive_mutation,
    )
    
    print(f"\nRunning Genetic Algorithm...")
    print(f"  Population size: {config.population_size}")
    print(f"  Generations: {config.generations}")
    print(f"  Crossover: OX (rate={config.crossover_rate})")
    print(f"  Mutation: {config.mutation} (rate={config.mutation_rate})")
    print(f"  Selection: {config.selection}")
    print(f"  Elitism: {config.elitism}")
    print(f"  Seed: {config.seed}")
    
    ga = GeneticAlgorithm(tsp, config)
    result = ga.evolve()
    
    improvement = ((baseline_distance - result.best_distance) / baseline_distance) * 100
    
    print(f"\nResults:")
    print(f"  Generations run: {result.generation}")
    print(f"  Best distance found: {result.best_distance:.4f}")
    print(f"  Improvement over baseline: {improvement:.2f}%")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_name = Path(args.input).stem
    
    tour_plot_path = output_dir / f"best_tour_{input_name}.png"
    plot_tour(tsp, result.best_tour, 
              title=f"Best TSP Tour - Distance: {result.best_distance:.2f}",
              output_path=str(tour_plot_path))
    print(f"\nSaved tour plot to {tour_plot_path}")
    
    convergence_plot_path = output_dir / f"convergence_{input_name}.png"
    plot_convergence(result.convergence_best, result.convergence_avg,
                    title=f"GA Convergence - {input_name}",
                    output_path=str(convergence_plot_path))
    print(f"Saved convergence plot to {convergence_plot_path}")
    
    tour_csv_path = output_dir / f"best_tour_{input_name}.csv"
    save_tour_csv(result.best_tour, result.best_distance, str(tour_csv_path))
    print(f"Saved tour data to {tour_csv_path}")
    
    if tsp.n_cities <= 10:
        print("\nSmall instance detected. Computing optimal solution via brute force...")
        optimal_tour, optimal_distance = brute_force_tsp(tsp)
        print(f"  Optimal distance: {optimal_distance:.4f}")
        print(f"  GA distance: {result.best_distance:.4f}")
        deviation = ((result.best_distance - optimal_distance) / optimal_distance) * 100
        print(f"  Deviation from optimal: {deviation:.2f}%")


def bruteforce_command(args):
    print(f"Loading TSP instance from {args.input}...")
    tsp = TSPInstance.from_csv(args.input)
    print(f"Loaded {tsp.n_cities} cities")
    
    if tsp.n_cities > 12:
        print(f"Error: Brute force is only feasible for small instances (n <= 12)")
        print(f"Your instance has {tsp.n_cities} cities.")
        sys.exit(1)
    
    print("\nComputing optimal solution via brute force...")
    optimal_tour, optimal_distance = brute_force_tsp(tsp)
    
    print(f"\nOptimal solution found!")
    print(f"  Distance: {optimal_distance:.4f}")
    print(f"  Tour: {optimal_tour}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_name = Path(args.input).stem
    
    tour_plot_path = output_dir / f"optimal_tour_{input_name}.png"
    plot_tour(tsp, optimal_tour, 
              title=f"Optimal TSP Tour - Distance: {optimal_distance:.2f}",
              output_path=str(tour_plot_path))
    print(f"\nSaved tour plot to {tour_plot_path}")
    
    tour_csv_path = output_dir / f"optimal_tour_{input_name}.csv"
    save_tour_csv(optimal_tour, optimal_distance, str(tour_csv_path))
    print(f"Saved tour data to {tour_csv_path}")


def compare_command(args):
    print(f"Loading TSP instance from {args.input}...")
    tsp = TSPInstance.from_csv(args.input)
    print(f"Loaded {tsp.n_cities} cities")
    
    if tsp.n_cities > 12:
        print(f"Error: Brute force comparison is only feasible for small instances (n <= 12)")
        print(f"Your instance has {tsp.n_cities} cities.")
        sys.exit(1)
    
    print("\nComputing optimal solution via brute force...")
    optimal_tour, optimal_distance = brute_force_tsp(tsp)
    print(f"  Optimal distance: {optimal_distance:.4f}")
    
    config = GAConfig(
        population_size=args.population_size,
        generations=args.generations,
        selection=args.selection,
        mutation=args.mutation,
        mutation_rate=args.mutation_rate,
        elitism=args.elitism,
        seed=args.seed,
    )
    
    print(f"\nRunning Genetic Algorithm...")
    print(f"  Population size: {config.population_size}")
    print(f"  Generations: {config.generations}")
    print(f"  Crossover: OX")
    print(f"  Mutation: {config.mutation}")
    print(f"  Seed: {config.seed}")
    
    ga = GeneticAlgorithm(tsp, config)
    result = ga.evolve()
    
    print(f"\nResults:")
    print(f"  Optimal distance: {optimal_distance:.4f}")
    print(f"  GA distance: {result.best_distance:.4f}")
    
    deviation = ((result.best_distance - optimal_distance) / optimal_distance) * 100
    if result.best_distance <= optimal_distance * 1.001:
        print(f"  Deviation: {deviation:.2f}% (OPTIMAL FOUND!)")
    else:
        print(f"  Deviation: {deviation:.2f}%")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_name = Path(args.input).stem
    
    from .viz import plot_tour_comparison, plot_convergence
    
    comparison_plot_path = output_dir / f"comparison_{input_name}.png"
    plot_tour_comparison(tsp, optimal_tour, result.best_tour,
                        optimal_distance, result.best_distance,
                        output_path=str(comparison_plot_path))
    print(f"\nSaved comparison plot to {comparison_plot_path}")
    
    convergence_plot_path = output_dir / f"convergence_{input_name}.png"
    plot_convergence(result.convergence_best, result.convergence_avg,
                    title=f"GA Convergence - {input_name}",
                    output_path=str(convergence_plot_path),
                    optimal_distance=optimal_distance)
    print(f"Saved convergence plot to {convergence_plot_path}")


def grid_search_command(args):
    print(f"Loading TSP instance from {args.input}...")
    tsp = TSPInstance.from_csv(args.input)
    print(f"Loaded {tsp.n_cities} cities")
    
    pop_sizes = [int(x) for x in args.pop.split(',')]
    mutations = args.mutation.split(',')
    
    print(f"\nGrid search configuration:")
    print(f"  Population sizes: {pop_sizes}")
    print(f"  Mutations: {mutations}")
    print(f"  Crossover: OX")
    print(f"  Trials per configuration: {args.trials}")
    
    results = []
    
    for pop_size in pop_sizes:
        for mutation in mutations:
            print(f"\nTesting: pop={pop_size}, mutation={mutation}")
            
            trial_distances = []
            for trial in range(args.trials):
                config = GAConfig(
                    population_size=pop_size,
                    generations=args.generations,
                    mutation=mutation,
                    seed=args.seed + trial,
                )
                
                ga = GeneticAlgorithm(tsp, config)
                result = ga.evolve()
                trial_distances.append(result.best_distance)
            
            avg_distance = np.mean(trial_distances)
            std_distance = np.std(trial_distances)
            min_distance = np.min(trial_distances)
            
            print(f"  Average: {avg_distance:.4f} +/- {std_distance:.4f}")
            print(f"  Best: {min_distance:.4f}")
            
            results.append({
                'pop_size': pop_size,
                'mutation': mutation,
                'avg_distance': avg_distance,
                'std_distance': std_distance,
                'min_distance': min_distance,
            })
    
    print("\n" + "="*80)
    print("GRID SEARCH SUMMARY")
    print("="*80)
    results.sort(key=lambda x: x['avg_distance'])
    
    for i, r in enumerate(results[:5], 1):
        print(f"\n{i}. pop_size={r['pop_size']}, mutation={r['mutation']}")
        print(f"   Avg: {r['avg_distance']:.4f} +/- {r['std_distance']:.4f}, Best: {r['min_distance']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Genetic Algorithm for TSP")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    make_data_parser = subparsers.add_parser('make-data', help='Generate datasets')
    make_data_parser.add_argument('--output-dir', default='data', help='Output directory')
    
    bruteforce_parser = subparsers.add_parser('bruteforce', help='Solve small TSP instance optimally (<=12 cities)')
    bruteforce_parser.add_argument('input', help='Input CSV file')
    bruteforce_parser.add_argument('--output-dir', default='outputs')
    
    solve_parser = subparsers.add_parser('solve', help='Solve TSP instance with GA')
    solve_parser.add_argument('input', help='Input CSV file')
    solve_parser.add_argument('--population-size', type=int, default=200)
    solve_parser.add_argument('--generations', type=int, default=1000)
    solve_parser.add_argument('--selection', choices=['tournament', 'roulette'], default='tournament')
    solve_parser.add_argument('--tournament-k', type=int, default=4)
    solve_parser.add_argument('--crossover-rate', type=float, default=0.9)
    solve_parser.add_argument('--mutation', choices=['swap', 'inversion', 'insert'], default='inversion')
    solve_parser.add_argument('--mutation-rate', type=float, default=0.2)
    solve_parser.add_argument('--elitism', type=int, default=5)
    solve_parser.add_argument('--stagnation-patience', type=int, default=150)
    solve_parser.add_argument('--seed', type=int, default=42)
    solve_parser.add_argument('--adaptive-mutation', action='store_true')
    solve_parser.add_argument('--output-dir', default='outputs')
    
    compare_parser = subparsers.add_parser('compare', help='Compare GA vs brute force optimal (<=12 cities)')
    compare_parser.add_argument('input', help='Input CSV file')
    compare_parser.add_argument('--population-size', type=int, default=200)
    compare_parser.add_argument('--generations', type=int, default=1000)
    compare_parser.add_argument('--selection', choices=['tournament', 'roulette'], default='tournament')
    compare_parser.add_argument('--mutation', choices=['swap', 'inversion', 'insert'], default='inversion')
    compare_parser.add_argument('--mutation-rate', type=float, default=0.2)
    compare_parser.add_argument('--elitism', type=int, default=5)
    compare_parser.add_argument('--seed', type=int, default=42)
    compare_parser.add_argument('--output-dir', default='outputs')
    
    grid_parser = subparsers.add_parser('grid-search', help='Grid search over parameters')
    grid_parser.add_argument('input', help='Input CSV file')
    grid_parser.add_argument('--pop', default='100,200', help='Population sizes (comma-separated)')
    grid_parser.add_argument('--mutation', default='inversion,swap', help='Mutations (comma-separated)')
    grid_parser.add_argument('--generations', type=int, default=500)
    grid_parser.add_argument('--trials', type=int, default=3)
    grid_parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if args.command == 'make-data':
        make_data_command(args)
    elif args.command == 'bruteforce':
        bruteforce_command(args)
    elif args.command == 'solve':
        solve_command(args)
    elif args.command == 'compare':
        compare_command(args)
    elif args.command == 'grid-search':
        grid_search_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
