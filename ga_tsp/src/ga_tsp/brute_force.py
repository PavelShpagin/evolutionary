import itertools
from typing import List, Tuple
import numpy as np

from .tsp import TSPInstance


def brute_force_tsp(tsp: TSPInstance) -> Tuple[List[int], float]:
    if tsp.n_cities > 12:
        raise ValueError("Brute force is only feasible for small instances (n <= 12)")
    
    cities = list(range(tsp.n_cities))
    best_tour = None
    best_distance = float('inf')
    
    for perm in itertools.permutations(cities[1:]):
        tour = [0] + list(perm)
        distance = tsp.evaluate(tour)
        
        if distance < best_distance:
            best_distance = distance
            best_tour = tour
    
    return best_tour, best_distance
