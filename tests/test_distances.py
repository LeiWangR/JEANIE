import pytest
import torch

from jeanie.distances import euclidean_cost


def test_euclidean_cost_exact_match_has_finite_gradient():
    query = torch.zeros(
        1, 2, 3,
        dtype=torch.double,
        requires_grad=True,
    )
    support = torch.zeros(
        2, 3,
        dtype=torch.double,
    )

    distance = euclidean_cost(query, support).sum()
    distance.backward()

    assert torch.isfinite(distance)
    assert torch.isfinite(query.grad).all()
    assert torch.allclose(query.grad, torch.zeros_like(query.grad))
    
def test_euclidean_cost_matches_definition():
    query = torch.tensor(
        [[[0.0, 0.0, 0.0],
          [3.0, 4.0, 0.0]]],
        dtype=torch.double,
    )
    support = torch.tensor(
        [[0.0, 0.0, 0.0]],
        dtype=torch.double,
    )

    distance = euclidean_cost(query, support)

    expected = torch.tensor(
        [[[0.0],
          [5.0]]],
        dtype=torch.double,
    )

    assert torch.allclose(distance, expected)