import pytest
import numpy as np
from ga_tsp.tsp import TSPInstance
from ga_tsp.ga import GeneticAlgorithm, GAConfig
from ga_tsp.baseline import nearest_neighbor, two_opt_improvement
from ga_tsp.utils import generate_random_points


def test_ga_vs_baseline_improvement():
    points = generate_random_points(30, seed=555)
    tsp = TSPInstance(points)
    
    baseline_tour, baseline_distance = nearest_neighbor(tsp, start=0)
    
    config = GAConfig(
        population_size=150,
        generations=500,
        mutation='inversion',
        seed=555,
    )
    
    ga = GeneticAlgorithm(tsp, config)
    result = ga.evolve()
    
    assert result.best_distance <= baseline_distance, \
        f"GA ({result.best_distance:.2f}) should be <= baseline ({baseline_distance:.2f})"
    
    improvement = ((baseline_distance - result.best_distance) / baseline_distance) * 100
    
    print(f"\nBaseline: {baseline_distance:.4f}")
    print(f"GA: {result.best_distance:.4f}")
    print(f"Improvement: {improvement:.2f}%")


def test_ga_consistent_quality():
    points = generate_random_points(25, seed=777)
    tsp = TSPInstance(points)
    
    baseline_tour, baseline_distance = nearest_neighbor(tsp, start=0)
    
    ga_distances = []
    for seed in range(3):
        config = GAConfig(
            population_size=100,
            generations=300,
            seed=seed,
        )
        ga = GeneticAlgorithm(tsp, config)
        result = ga.evolve()
        ga_distances.append(result.best_distance)
    
    avg_ga_distance = np.mean(ga_distances)
    
    assert avg_ga_distance <= baseline_distance, \
        f"Average GA distance ({avg_ga_distance:.2f}) should be <= baseline ({baseline_distance:.2f})"


def test_baseline_nearest_neighbor():
    points = generate_random_points(20, seed=333)
    tsp = TSPInstance(points)
    
    tour, distance = nearest_neighbor(tsp, start=0)
    
    assert len(tour) == tsp.n_cities
    assert set(tour) == set(range(tsp.n_cities))
    assert tour[0] == 0
    assert distance == tsp.evaluate(tour)


def test_two_opt_improvement():
    points = generate_random_points(15, seed=444)
    tsp = TSPInstance(points)
    
    initial_tour, initial_distance = nearest_neighbor(tsp, start=0)
    
    improved_tour, improved_distance = two_opt_improvement(tsp, initial_tour, max_iterations=100)
    
    assert improved_distance <= initial_distance
    assert len(improved_tour) == tsp.n_cities
    assert set(improved_tour) == set(range(tsp.n_cities))


def test_multiple_baselines_comparison():
    points = generate_random_points(20, seed=888)
    tsp = TSPInstance(points)
    
    nn_distances = []
    for start in range(min(5, tsp.n_cities)):
        tour, distance = nearest_neighbor(tsp, start=start)
        nn_distances.append(distance)
    
    best_nn_distance = min(nn_distances)
    
    config = GAConfig(
        population_size=100,
        generations=300,
        seed=888,
    )
    ga = GeneticAlgorithm(tsp, config)
    result = ga.evolve()
    
    assert result.best_distance <= max(nn_distances), \
        "GA should beat worst nearest neighbor"
