from typing import List, Tuple
import numpy as np


def pmx_crossover(parent1: List[int], parent2: List[int], rng: np.random.Generator) -> Tuple[List[int], List[int]]:
    n = len(parent1)
    cx_point1, cx_point2 = sorted(rng.integers(0, n, size=2))
    
    if cx_point1 == cx_point2:
        return parent1.copy(), parent2.copy()
    
    def create_child(p1, p2):
        child = [-1] * n
        child[cx_point1:cx_point2] = p1[cx_point1:cx_point2]
        
        for i in range(cx_point1, cx_point2):
            if p2[i] not in child[cx_point1:cx_point2]:
                pos = i
                while cx_point1 <= pos < cx_point2:
                    pos = p2.index(p1[pos])
                child[pos] = p2[i]
        
        for i in range(n):
            if child[i] == -1:
                child[i] = p2[i]
        
        return child
    
    child1 = create_child(parent1, parent2)
    child2 = create_child(parent2, parent1)
    
    return child1, child2


def ox_crossover(parent1: List[int], parent2: List[int], rng: np.random.Generator) -> Tuple[List[int], List[int]]:
    n = len(parent1)
    cx_point1, cx_point2 = sorted(rng.integers(0, n, size=2))
    
    if cx_point1 == cx_point2:
        return parent1.copy(), parent2.copy()
    
    def create_child(p1, p2):
        child = [-1] * n
        child[cx_point1:cx_point2] = p1[cx_point1:cx_point2]
        
        p2_filtered = [x for x in p2 if x not in child[cx_point1:cx_point2]]
        
        idx = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = p2_filtered[idx]
                idx += 1
        
        return child
    
    child1 = create_child(parent1, parent2)
    child2 = create_child(parent2, parent1)
    
    return child1, child2


def swap_mutation(tour: List[int], rng: np.random.Generator) -> List[int]:
    mutated = tour.copy()
    i, j = rng.choice(len(tour), size=2, replace=False)
    mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def inversion_mutation(tour: List[int], rng: np.random.Generator) -> List[int]:
    mutated = tour.copy()
    i, j = sorted(rng.choice(len(tour), size=2, replace=False))
    mutated[i:j+1] = reversed(mutated[i:j+1])
    return mutated


def insert_mutation(tour: List[int], rng: np.random.Generator) -> List[int]:
    mutated = tour.copy()
    i = rng.integers(0, len(tour))
    j = rng.integers(0, len(tour))
    element = mutated.pop(i)
    mutated.insert(j, element)
    return mutated


CROSSOVER_OPS = {
    'pmx': pmx_crossover,
    'ox': ox_crossover,
}

MUTATION_OPS = {
    'swap': swap_mutation,
    'inversion': inversion_mutation,
    'insert': insert_mutation,
}
