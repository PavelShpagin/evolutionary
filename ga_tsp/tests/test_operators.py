import pytest
import numpy as np
from ga_tsp.operators import (
    pmx_crossover, ox_crossover, 
    swap_mutation, inversion_mutation, insert_mutation
)


def test_pmx_crossover_valid_permutation():
    rng = np.random.default_rng(42)
    parent1 = [0, 1, 2, 3, 4, 5]
    parent2 = [5, 4, 3, 2, 1, 0]
    
    child1, child2 = pmx_crossover(parent1, parent2, rng)
    
    assert len(child1) == len(parent1)
    assert len(child2) == len(parent2)
    assert set(child1) == set(parent1)
    assert set(child2) == set(parent2)
    assert len(set(child1)) == len(child1)
    assert len(set(child2)) == len(child2)


def test_ox_crossover_valid_permutation():
    rng = np.random.default_rng(42)
    parent1 = [0, 1, 2, 3, 4, 5, 6, 7]
    parent2 = [7, 6, 5, 4, 3, 2, 1, 0]
    
    child1, child2 = ox_crossover(parent1, parent2, rng)
    
    assert len(child1) == len(parent1)
    assert len(child2) == len(parent2)
    assert set(child1) == set(parent1)
    assert set(child2) == set(parent2)
    assert len(set(child1)) == len(child1)
    assert len(set(child2)) == len(child2)


def test_swap_mutation_preserves_cities():
    rng = np.random.default_rng(42)
    tour = [0, 1, 2, 3, 4, 5]
    
    mutated = swap_mutation(tour, rng)
    
    assert len(mutated) == len(tour)
    assert set(mutated) == set(tour)
    assert mutated != tour


def test_inversion_mutation_preserves_cities():
    rng = np.random.default_rng(42)
    tour = [0, 1, 2, 3, 4, 5, 6, 7]
    
    mutated = inversion_mutation(tour, rng)
    
    assert len(mutated) == len(tour)
    assert set(mutated) == set(tour)


def test_insert_mutation_preserves_cities():
    rng = np.random.default_rng(42)
    tour = [0, 1, 2, 3, 4, 5]
    
    mutated = insert_mutation(tour, rng)
    
    assert len(mutated) == len(tour)
    assert set(mutated) == set(tour)


def test_crossover_same_parents():
    rng = np.random.default_rng(42)
    parent = [0, 1, 2, 3, 4]
    
    child1, child2 = pmx_crossover(parent, parent, rng)
    assert set(child1) == set(parent)
    assert set(child2) == set(parent)
    
    child1, child2 = ox_crossover(parent, parent, rng)
    assert set(child1) == set(parent)
    assert set(child2) == set(parent)


def test_mutations_multiple_runs():
    rng = np.random.default_rng(123)
    tour = list(range(20))
    
    for _ in range(100):
        mutated = swap_mutation(tour, rng)
        assert set(mutated) == set(tour)
        
        mutated = inversion_mutation(tour, rng)
        assert set(mutated) == set(tour)
        
        mutated = insert_mutation(tour, rng)
        assert set(mutated) == set(tour)
