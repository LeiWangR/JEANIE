"""Free Viewpoint Matching (FVM).

FVM follows Eq. (13) of the JEANIE paper:
perform a viewpoint SoftMin independently for every temporal pair, then
apply soft-DTW to the resulting temporal distance matrix.
"""

import torch

from .alignment import soft_dtw, softmin


def _viewpoint_softmin(cost: torch.Tensor, gamma: float) -> torch.Tensor:
    """Reduce every viewpoint axis, leaving [T, U]."""
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    if cost.ndim == 3:
        # [K, T, U] -> [T, U]
        return softmin(cost, gamma, dim=0)

    if cost.ndim == 4:
        # [Kq, Ks, T, U] -> [T, U]
        return softmin(
            cost.reshape(
                cost.shape[0] * cost.shape[1],
                cost.shape[2],
                cost.shape[3],
            ),
            gamma,
            dim=0,
        )

    if cost.ndim == 6:
        # [Kq1,Kq2,Ks1,Ks2,T,U] -> [T,U]
        return softmin(
            cost.reshape(
                cost.shape[0] * cost.shape[1] * cost.shape[2] * cost.shape[3],
                cost.shape[4],
                cost.shape[5],
            ),
            gamma,
            dim=0,
        )

    raise ValueError(
        "cost must have shape [K,T,U], [Kq,Ks,T,U], "
        "or [Kq1,Kq2,Ks1,Ks2,T,U]"
    )


def fvm_from_cost(cost: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """General FVM implementation.

    Accepted layouts:

        [K, T, U]
            Query-only one-axis special case.

        [K1, K2, T, U]
            Query-only two-axis special case.

        [Kq, Ks, T, U]
            Full one-axis query/support viewpoint matching.

        [Kq1, Kq2, Ks1, Ks2, T, U]
            Full two-axis query/support viewpoint matching.

    In each case, all viewpoint indices are SoftMin-reduced independently
    at every temporal pair before soft-DTW is applied.
    """
    if not cost.is_floating_point():
        raise TypeError("cost must be a floating-point tensor")

    temporal_cost = _viewpoint_softmin(cost, gamma)
    return soft_dtw(temporal_cost, gamma)


def fvm_query_only_1d(cost: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Query-only 1-D FVM for cost [K, T, U]."""
    if cost.ndim != 3:
        raise ValueError("cost must have shape [K, T, U]")
    return fvm_from_cost(cost, gamma)


def fvm_query_only_2d(cost: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Query-only 2-D FVM for cost [K1, K2, T, U]."""
    if cost.ndim != 4:
        raise ValueError("cost must have shape [K1, K2, T, U]")
    return fvm_from_cost(cost, gamma)


def fvm_1d_from_cost(cost: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Full 1-D FVM for cost [K_query, K_support, T, U]."""
    if cost.ndim != 4:
        raise ValueError("cost must have shape [K_query, K_support, T, U]")
    return fvm_from_cost(cost, gamma)


def fvm_2d_from_cost(cost: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Full 2-D FVM for cost [Kq1,Kq2,Ks1,Ks2,T,U]."""
    if cost.ndim != 6:
        raise ValueError(
            "cost must have shape [Kq1,Kq2,Ks1,Ks2,T,U]"
        )
    return fvm_from_cost(cost, gamma)
