from typing import List, Tuple
import numpy as np


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.sqrt(np.sum((p1 - p2) ** 2)))


def compute_distance_matrix(points: np.ndarray) -> np.ndarray:
    n = len(points)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean_distance(points[i], points[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix


def tour_length(tour: List[int], dist_matrix: np.ndarray) -> float:
    total = 0.0
    n = len(tour)
    for i in range(n):
        total += dist_matrix[tour[i], tour[(i + 1) % n]]
    return total


def validate_tour(tour: List[int], n_cities: int) -> bool:
    if len(tour) != n_cities:
        return False
    if len(set(tour)) != n_cities:
        return False
    if min(tour) != 0 or max(tour) != n_cities - 1:
        return False
    return True


class TSPInstance:
    def __init__(self, points: np.ndarray):
        self.points = points
        self.n_cities = len(points)
        self.dist_matrix = compute_distance_matrix(points)
    
    def evaluate(self, tour: List[int]) -> float:
        if not validate_tour(tour, self.n_cities):
            raise ValueError("Invalid tour")
        return tour_length(tour, self.dist_matrix)
    
    @classmethod
    def from_csv(cls, filepath: str) -> 'TSPInstance':
        data = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=(1, 2))
        return cls(data)
    
    def to_csv(self, filepath: str):
        with open(filepath, 'w') as f:
            f.write('id,x,y\n')
            for i, (x, y) in enumerate(self.points):
                f.write(f'{i},{x},{y}\n')
