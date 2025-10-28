import numpy as np
from pathlib import Path


def generate_random_points(n: int, seed: int = 42, bounds: tuple = (0, 100)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.uniform(bounds[0], bounds[1], size=(n, 2))
    return points


def save_tour_csv(tour: list, distance: float, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('position,city_id\n')
        for pos, city_id in enumerate(tour):
            f.write(f'{pos},{city_id}\n')
        f.write(f'\n# Total distance: {distance:.4f}\n')
