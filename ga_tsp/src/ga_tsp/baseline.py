from typing import List, Tuple
import numpy as np

from .tsp import TSPInstance


def nearest_neighbor(tsp: TSPInstance, start: int = 0) -> Tuple[List[int], float]:
    n = tsp.n_cities
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    
    current = start
    while unvisited:
        nearest = min(unvisited, key=lambda city: tsp.dist_matrix[current, city])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    distance = tsp.evaluate(tour)
    return tour, distance


def two_opt_improvement(tsp: TSPInstance, tour: List[int], max_iterations: int = 1000) -> Tuple[List[int], float]:
    improved = True
    best_tour = tour.copy()
    best_distance = tsp.evaluate(best_tour)
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        for i in range(1, len(best_tour) - 1):
            for j in range(i + 1, len(best_tour)):
                new_tour = best_tour.copy()
                new_tour[i:j] = reversed(new_tour[i:j])
                
                new_distance = tsp.evaluate(new_tour)
                
                if new_distance < best_distance:
                    best_tour = new_tour
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break
        iteration += 1
    
    return best_tour, best_distance
