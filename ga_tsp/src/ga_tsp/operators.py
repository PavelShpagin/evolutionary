from typing import List, Tuple
import numpy as np


def ox_crossover(parent1: List[int], parent2: List[int], rng: np.random.Generator) -> Tuple[List[int], List[int]]:
    """
    Order Crossover (OX): Copies a segment from p1 to child, then fills 
    remaining positions from p2 in order, inserting only if not present.
    """
    n = len(parent1)
    cx_point1, cx_point2 = sorted(rng.integers(0, n, size=2))
    
    if cx_point1 == cx_point2:
        return parent1.copy(), parent2.copy()
    
    def create_child(p1, p2):
        child = [-1] * n
        # Copy segment from p1 to child
        child[cx_point1:cx_point2] = p1[cx_point1:cx_point2]
        
        # Use set for O(1) lookup
        in_child = set(child[cx_point1:cx_point2])
        
        # Fill remaining positions from p2, inserting only if not present
        child_idx = 0
        for city in p2:
            if city not in in_child:
                # Find next empty position
                while child_idx < n and child[child_idx] != -1:
                    child_idx += 1
                child[child_idx] = city
        
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
    'ox': ox_crossover,
}

MUTATION_OPS = {
    'swap': swap_mutation,
    'inversion': inversion_mutation,
    'insert': insert_mutation,
}
