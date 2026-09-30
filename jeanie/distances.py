"""Base-distance helpers for JEANIE/FVM."""

import torch


def euclidean_cost(
    query: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Euclidean base distance.

    Args:
        query: [K, T, D].
        support: [U, D].

    Returns:
        [K, T, U], where the final two axes contain pairwise distances.
    """
    if query.ndim != 3 or support.ndim != 2:
        raise ValueError("query must be [K,T,D] and support must be [U,D]")
    if query.shape[-1] != support.shape[-1]:
        raise ValueError("query/support feature dimensions do not match")

    difference = query[:, :, None, :] - support[None, None, :, :]
    return torch.sqrt(difference.pow(2).sum(dim=-1))


def squared_euclidean_cost(
    query: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Squared-Euclidean base distance; returns [K, T, U]."""
    if query.ndim != 3 or support.ndim != 2:
        raise ValueError("query must be [K,T,D] and support must be [U,D]")
    if query.shape[-1] != support.shape[-1]:
        raise ValueError("query/support feature dimensions do not match")

    difference = query[:, :, None, :] - support[None, None, :, :]
    return difference.pow(2).sum(dim=-1)


def rbf_cost(
    query: torch.Tensor,
    support: torch.Tensor,
    sigma: float = 0.5,
) -> torch.Tensor:
    """RBF-style base cost.

    d(x,y) = sum_d [2 - 2 exp(-sigma (x_d-y_d)^2)].

    Returns [K, T, U].
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if query.ndim != 3 or support.ndim != 2:
        raise ValueError("query must be [K,T,D] and support must be [U,D]")
    if query.shape[-1] != support.shape[-1]:
        raise ValueError("query/support feature dimensions do not match")

    difference = query[:, :, None, :] - support[None, None, :, :]
    return (2.0 - 2.0 * torch.exp(-sigma * difference.pow(2))).sum(dim=-1)
