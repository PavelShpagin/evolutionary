from typing import List, Callable, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from .tsp import TSPInstance
from .operators import CROSSOVER_OPS, MUTATION_OPS


@dataclass
class GAConfig:
    population_size: int = 200
    generations: int = 1000
    selection: str = 'tournament'
    tournament_k: int = 4
    crossover: str = 'pmx'
    crossover_rate: float = 0.9
    mutation: str = 'inversion'
    mutation_rate: float = 0.2
    elitism: int = 5
    stagnation_patience: Optional[int] = 150
    seed: int = 42
    adaptive_mutation: bool = False
    stagnation_threshold: int = 50


@dataclass
class GAResult:
    best_tour: List[int]
    best_distance: float
    convergence_best: List[float]
    convergence_avg: List[float]
    generation: int


class GeneticAlgorithm:
    def __init__(self, tsp: TSPInstance, config: GAConfig):
        self.tsp = tsp
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        
        self.crossover_fn = CROSSOVER_OPS[config.crossover]
        self.mutation_fn = MUTATION_OPS[config.mutation]
        self.mutation_rate = config.mutation_rate
    
    def initialize_population(self) -> List[List[int]]:
        population = []
        base = list(range(self.tsp.n_cities))
        for _ in range(self.config.population_size):
            tour = base.copy()
            self.rng.shuffle(tour)
            population.append(tour)
        return population
    
    def evaluate_population(self, population: List[List[int]]) -> np.ndarray:
        fitness = np.array([self.tsp.evaluate(tour) for tour in population])
        return fitness
    
    def tournament_selection(self, population: List[List[int]], fitness: np.ndarray) -> List[int]:
        indices = self.rng.choice(len(population), size=self.config.tournament_k, replace=False)
        best_idx = indices[np.argmin(fitness[indices])]
        return population[best_idx].copy()
    
    def roulette_selection(self, population: List[List[int]], fitness: np.ndarray) -> List[int]:
        inv_fitness = 1.0 / (fitness + 1e-10)
        probs = inv_fitness / np.sum(inv_fitness)
        idx = self.rng.choice(len(population), p=probs)
        return population[idx].copy()
    
    def select_parent(self, population: List[List[int]], fitness: np.ndarray) -> List[int]:
        if self.config.selection == 'tournament':
            return self.tournament_selection(population, fitness)
        else:
            return self.roulette_selection(population, fitness)
    
    def evolve(self) -> GAResult:
        population = self.initialize_population()
        
        best_tour = None
        best_distance = float('inf')
        convergence_best = []
        convergence_avg = []
        
        stagnation_counter = 0
        
        for gen in range(self.config.generations):
            fitness = self.evaluate_population(population)
            
            gen_best_idx = np.argmin(fitness)
            gen_best_distance = fitness[gen_best_idx]
            
            if gen_best_distance < best_distance:
                best_distance = gen_best_distance
                best_tour = population[gen_best_idx].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            convergence_best.append(best_distance)
            convergence_avg.append(np.mean(fitness))
            
            if self.config.stagnation_patience and stagnation_counter >= self.config.stagnation_patience:
                break
            
            if self.config.adaptive_mutation and stagnation_counter > self.config.stagnation_threshold:
                self.mutation_rate = min(0.5, self.mutation_rate * 1.1)
            
            elite_indices = np.argsort(fitness)[:self.config.elitism]
            elite = [population[i].copy() for i in elite_indices]
            
            new_population = elite.copy()
            
            while len(new_population) < self.config.population_size:
                parent1 = self.select_parent(population, fitness)
                parent2 = self.select_parent(population, fitness)
                
                if self.rng.random() < self.config.crossover_rate:
                    child1, child2 = self.crossover_fn(parent1, parent2, self.rng)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                if self.rng.random() < self.mutation_rate:
                    child1 = self.mutation_fn(child1, self.rng)
                
                if self.rng.random() < self.mutation_rate:
                    child2 = self.mutation_fn(child2, self.rng)
                
                new_population.append(child1)
                if len(new_population) < self.config.population_size:
                    new_population.append(child2)
            
            population = new_population[:self.config.population_size]
        
        return GAResult(
            best_tour=best_tour,
            best_distance=best_distance,
            convergence_best=convergence_best,
            convergence_avg=convergence_avg,
            generation=gen + 1
        )
