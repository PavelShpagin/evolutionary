import pytest
import numpy as np
from ga_tsp.tsp import TSPInstance
from ga_tsp.ga import GeneticAlgorithm, GAConfig
from ga_tsp.brute_force import brute_force_tsp
from ga_tsp.utils import generate_random_points


def test_small_instance_optimal():
    points = generate_random_points(8, seed=42)
    tsp = TSPInstance(points)
    
    optimal_tour, optimal_distance = brute_force_tsp(tsp)
    
    config = GAConfig(
        population_size=100,
        generations=500,
        mutation='inversion',
        seed=42,
        stagnation_patience=100,
    )
    
    ga = GeneticAlgorithm(tsp, config)
    result = ga.evolve()
    
    deviation = ((result.best_distance - optimal_distance) / optimal_distance) * 100
    
    assert result.best_distance >= optimal_distance
    assert deviation <= 1.0, f"GA deviation {deviation:.2f}% exceeds 1% threshold"


def test_ga_convergence():
    points = generate_random_points(15, seed=123)
    tsp = TSPInstance(points)
    
    config = GAConfig(
        population_size=50,
        generations=200,
        seed=123,
    )
    
    ga = GeneticAlgorithm(tsp, config)
    result = ga.evolve()
    
    assert len(result.convergence_best) > 0
    assert len(result.convergence_avg) > 0
    
    assert result.convergence_best[0] >= result.convergence_best[-1]
    
    assert result.best_tour is not None
    assert len(result.best_tour) == tsp.n_cities
    assert set(result.best_tour) == set(range(tsp.n_cities))


def test_ga_different_operators():
    points = generate_random_points(10, seed=456)
    tsp = TSPInstance(points)
    
    configs = [
        GAConfig(population_size=50, generations=100, mutation='swap', seed=1),
        GAConfig(population_size=50, generations=100, mutation='inversion', seed=1),
        GAConfig(population_size=50, generations=100, mutation='insert', seed=1),
    ]
    
    for config in configs:
        ga = GeneticAlgorithm(tsp, config)
        result = ga.evolve()
        
        assert result.best_tour is not None
        assert set(result.best_tour) == set(range(tsp.n_cities))
        assert result.best_distance > 0


def test_ga_stagnation_stop():
    points = generate_random_points(10, seed=789)
    tsp = TSPInstance(points)
    
    config = GAConfig(
        population_size=30,
        generations=1000,
        stagnation_patience=20,
        seed=789,
    )
    
    ga = GeneticAlgorithm(tsp, config)
    result = ga.evolve()
    
    assert result.generation < 1000


def test_ga_reproducibility():
    points = generate_random_points(12, seed=999)
    tsp = TSPInstance(points)
    
    config1 = GAConfig(population_size=50, generations=100, seed=999)
    ga1 = GeneticAlgorithm(tsp, config1)
    result1 = ga1.evolve()
    
    config2 = GAConfig(population_size=50, generations=100, seed=999)
    ga2 = GeneticAlgorithm(tsp, config2)
    result2 = ga2.evolve()
    
    assert result1.best_distance == result2.best_distance
    assert result1.best_tour == result2.best_tour
