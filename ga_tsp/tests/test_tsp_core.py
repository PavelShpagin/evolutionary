import pytest
import numpy as np
from ga_tsp.tsp import TSPInstance, validate_tour, compute_distance_matrix, tour_length


def test_distance_matrix_symmetry():
    points = np.array([[0, 0], [1, 1], [2, 0], [1, 2]])
    tsp = TSPInstance(points)
    
    assert tsp.dist_matrix.shape == (4, 4)
    
    for i in range(4):
        for j in range(4):
            assert abs(tsp.dist_matrix[i, j] - tsp.dist_matrix[j, i]) < 1e-10
    
    for i in range(4):
        assert tsp.dist_matrix[i, i] == 0


def test_validate_tour_correct():
    assert validate_tour([0, 1, 2, 3], 4) == True
    assert validate_tour([3, 0, 2, 1], 4) == True
    assert validate_tour([0], 1) == True


def test_validate_tour_incorrect():
    assert validate_tour([0, 1, 2], 4) == False
    assert validate_tour([0, 1, 1, 2], 4) == False
    assert validate_tour([1, 2, 3, 4], 4) == False
    assert validate_tour([0, 1, 2, 3, 4], 4) == False


def test_tour_length_calculation():
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    tsp = TSPInstance(points)
    
    tour = [0, 1, 2, 3]
    length = tsp.evaluate(tour)
    expected = 4.0
    assert abs(length - expected) < 1e-10


def test_tour_length_different_order():
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    tsp = TSPInstance(points)
    
    tour1 = [0, 1, 2, 3]
    tour2 = [0, 2, 1, 3]
    
    length1 = tsp.evaluate(tour1)
    length2 = tsp.evaluate(tour2)
    
    assert abs(length1 - 4.0) < 1e-10
    assert length2 > length1


def test_tsp_from_csv(tmp_path):
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("id,x,y\n0,0.0,0.0\n1,1.0,0.0\n2,1.0,1.0\n3,0.0,1.0\n")
    
    tsp = TSPInstance.from_csv(str(csv_path))
    
    assert tsp.n_cities == 4
    assert tsp.points.shape == (4, 2)


def test_tsp_to_csv(tmp_path):
    points = np.array([[0, 0], [1, 1], [2, 2]])
    tsp = TSPInstance(points)
    
    csv_path = tmp_path / "output.csv"
    tsp.to_csv(str(csv_path))
    
    assert csv_path.exists()
    
    loaded_tsp = TSPInstance.from_csv(str(csv_path))
    assert loaded_tsp.n_cities == 3
    np.testing.assert_array_almost_equal(loaded_tsp.points, points)
