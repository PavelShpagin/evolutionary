from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .tsp import TSPInstance


def plot_tour(tsp: TSPInstance, tour: List[int], title: str = "TSP Tour", 
              output_path: Optional[str] = None, show: bool = False):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    points = tsp.points
    tour_points = points[tour]
    
    ax.plot(tour_points[:, 0], tour_points[:, 1], 'b-', linewidth=2, alpha=0.7, label='Tour')
    ax.plot([tour_points[-1, 0], tour_points[0, 0]], 
            [tour_points[-1, 1], tour_points[0, 1]], 'b-', linewidth=2, alpha=0.7)
    
    ax.scatter(points[:, 0], points[:, 1], c='red', s=100, zorder=5, edgecolors='black', linewidths=1)
    
    ax.scatter(points[tour[0], 0], points[tour[0], 1], c='green', s=200, 
               marker='*', zorder=6, edgecolors='black', linewidths=1, label='Start')
    
    for i, (x, y) in enumerate(points):
        ax.annotate(str(i), (x, y), fontsize=8, ha='center', va='center', color='white', weight='bold')
    
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()


def plot_tour_comparison(tsp: TSPInstance, optimal_tour: List[int], ga_tour: List[int],
                        optimal_distance: float, ga_distance: float,
                        output_path: Optional[str] = None, show: bool = False):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    points = tsp.points
    
    for ax, tour, distance, title in zip(
        axes,
        [optimal_tour, ga_tour],
        [optimal_distance, ga_distance],
        ['Optimal (Brute Force)', 'GA Solution']
    ):
        tour_points = points[tour]
        
        ax.plot(tour_points[:, 0], tour_points[:, 1], 'b-', linewidth=2, alpha=0.7, label='Tour')
        ax.plot([tour_points[-1, 0], tour_points[0, 0]], 
                [tour_points[-1, 1], tour_points[0, 1]], 'b-', linewidth=2, alpha=0.7)
        
        ax.scatter(points[:, 0], points[:, 1], c='red', s=100, zorder=5, edgecolors='black', linewidths=1)
        
        ax.scatter(points[tour[0], 0], points[tour[0], 1], c='green', s=200, 
                   marker='*', zorder=6, edgecolors='black', linewidths=1, label='Start')
        
        for i, (x, y) in enumerate(points):
            ax.annotate(str(i), (x, y), fontsize=8, ha='center', va='center', color='white', weight='bold')
        
        ax.set_title(f'{title}\nDistance: {distance:.2f}', fontsize=13, weight='bold')
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    deviation = ((ga_distance - optimal_distance) / optimal_distance) * 100
    fig.suptitle(f'TSP Comparison - GA Deviation: {deviation:.2f}%', 
                 fontsize=15, weight='bold')
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()


def plot_convergence(best_fitness: List[float], avg_fitness: List[float],
                    title: str = "GA Convergence", output_path: Optional[str] = None,
                    show: bool = False, optimal_distance: Optional[float] = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    generations = list(range(1, len(best_fitness) + 1))
    
    ax.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
    ax.plot(generations, avg_fitness, 'r--', linewidth=1.5, alpha=0.7, label='Average Fitness')
    
    if optimal_distance is not None:
        ax.axhline(y=optimal_distance, color='g', linestyle=':', linewidth=2, 
                   label=f'Optimal: {optimal_distance:.2f}')
    
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Tour Distance', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()
