"""JEANIE temporal-viewpoint alignment and soft-DTW.

The 1-D JEANIE recurrence below is a direct implementation of Algorithm 1
in the paper. The 2-D version is the direct extension to a K1 x K2 viewpoint
grid described in the paper.
"""

from typing import Optional, Tuple

import torch


def softmin(x: torch.Tensor, gamma: float, dim: Optional[int] = None) -> torch.Tensor:
    """Differentiable soft minimum.

    Args:
        x: Input tensor.
        gamma: Positive soft-min temperature.
        dim: Dimension over which to compute the soft minimum. If None,
            all elements are reduced.

    Returns:
        Soft minimum of `x`.
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    if dim is None:
        x = x.reshape(-1)
        return -gamma * torch.logsumexp(-x / gamma, dim=0)

    return -gamma * torch.logsumexp(-x / gamma, dim=dim)


def soft_dtw(cost: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Compute scalar soft-DTW for a [T, U] cost matrix.

    The implementation is intentionally out-of-place with respect to the
    differentiable DP values, so it is safe for PyTorch autograd/gradcheck.
    """
    if cost.ndim != 2:
        raise ValueError("cost must have shape [T, U]")
    if not cost.is_floating_point():
        raise TypeError("cost must be a floating-point tensor")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    T, U = cost.shape
    if T == 0 or U == 0:
        raise ValueError("cost must be non-empty")

    # Python containers hold the graph nodes. No Tensor accumulator is
    # mutated in-place.
    R = [[None for _ in range(U)] for _ in range(T)]
    R[0][0] = cost[0, 0]

    for t in range(T):
        for u in range(U):
            if t == 0 and u == 0:
                continue

            predecessors = []
            if t > 0:
                predecessors.append(R[t - 1][u])
            if u > 0:
                predecessors.append(R[t][u - 1])
            if t > 0 and u > 0:
                predecessors.append(R[t - 1][u - 1])

            R[t][u] = cost[t, u] + softmin(
                torch.stack(predecessors), gamma
            )

    return R[-1][-1]


def jeanie_1d_from_cost(
    cost: torch.Tensor,
    gamma: float = 0.1,
    max_shift: int = 1,
    return_accumulator: bool = False,
):
    """JEANIE for one viewpoint axis, matching Algorithm 1.

    Args:
        cost: Base-distance tensor with shape [K, T, U].
              cost[n, t, u] is the base distance between query viewpoint
              n at query temporal block t and support block u.
        gamma: Positive soft-min temperature.
        max_shift: Viewpoint smoothness parameter. A value of 1 allows
            predecessor viewpoints n-1, n, n+1.
        return_accumulator: Also return the accumulated DP tensor.

    Returns:
        Scalar JEANIE distance, and optionally the accumulator [K, T, U].
    """
    if cost.ndim != 3:
        raise ValueError("cost must have shape [K, T, U]")
    if not cost.is_floating_point():
        raise TypeError("cost must be a floating-point tensor")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if max_shift < 0:
        raise ValueError("max_shift must be >= 0")

    K, T, U = cost.shape
    if min(K, T, U) == 0:
        raise ValueError("cost dimensions must be non-empty")

    # Algorithm 1 initializes every possible starting viewpoint.
    R = [[[None for _ in range(U)] for _ in range(T)] for _ in range(K)]

    for n in range(K):
        R[n][0][0] = cost[n, 0, 0]

    temporal_moves = ((0, 1), (1, 0), (1, 1))

    for t in range(T):
        for u in range(U):
            if t == 0 and u == 0:
                continue

            for n in range(K):
                predecessors = []

                for i in range(-max_shift, max_shift + 1):
                    previous_n = n - i
                    if previous_n < 0 or previous_n >= K:
                        continue

                    for j, k in temporal_moves:
                        previous_t = t - j
                        previous_u = u - k

                        if previous_t < 0 or previous_u < 0:
                            continue

                        predecessors.append(
                            R[previous_n][previous_t][previous_u]
                        )

                R[n][t][u] = cost[n, t, u] + softmin(
                    torch.stack(predecessors), gamma
                )

    final_states = torch.stack(
        [R[n][T - 1][U - 1] for n in range(K)]
    )
    distance = softmin(final_states, gamma)

    if not return_accumulator:
        return distance

    accumulator = torch.stack(
        [
            torch.stack(
                [torch.stack(R[n][t]) for t in range(T)]
            )
            for n in range(K)
        ]
    )
    return distance, accumulator


def jeanie_2d_from_cost(
    cost: torch.Tensor,
    gamma: float = 0.1,
    max_shift_az: int = 1,
    max_shift_alt: int = 1,
    return_accumulator: bool = False,
):
    """JEANIE on a two-axis viewpoint grid.

    Args:
        cost: Base-distance tensor [K1, K2, T, U].
        gamma: Positive soft-min temperature.
        max_shift_az: Maximum predecessor shift on viewpoint axis 1.
        max_shift_alt: Maximum predecessor shift on viewpoint axis 2.
        return_accumulator: Also return the accumulated DP tensor.

    Returns:
        Scalar JEANIE distance, and optionally [K1, K2, T, U].
    """
    if cost.ndim != 4:
        raise ValueError("cost must have shape [K1, K2, T, U]")
    if not cost.is_floating_point():
        raise TypeError("cost must be a floating-point tensor")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if max_shift_az < 0 or max_shift_alt < 0:
        raise ValueError("max shifts must be >= 0")

    K1, K2, T, U = cost.shape
    if min(K1, K2, T, U) == 0:
        raise ValueError("cost dimensions must be non-empty")

    R = [
        [
            [[None for _ in range(U)] for _ in range(T)]
            for _ in range(K2)
        ]
        for _ in range(K1)
    ]

    for k1 in range(K1):
        for k2 in range(K2):
            R[k1][k2][0][0] = cost[k1, k2, 0, 0]

    temporal_moves = ((0, 1), (1, 0), (1, 1))

    for t in range(T):
        for u in range(U):
            if t == 0 and u == 0:
                continue

            for k1 in range(K1):
                for k2 in range(K2):
                    predecessors = []

                    for shift_1 in range(-max_shift_az, max_shift_az + 1):
                        previous_k1 = k1 - shift_1
                        if previous_k1 < 0 or previous_k1 >= K1:
                            continue

                        for shift_2 in range(-max_shift_alt, max_shift_alt + 1):
                            previous_k2 = k2 - shift_2
                            if previous_k2 < 0 or previous_k2 >= K2:
                                continue

                            for j, k in temporal_moves:
                                previous_t = t - j
                                previous_u = u - k

                                if previous_t < 0 or previous_u < 0:
                                    continue

                                predecessors.append(
                                    R[previous_k1][previous_k2][previous_t][previous_u]
                                )

                    R[k1][k2][t][u] = cost[k1, k2, t, u] + softmin(
                        torch.stack(predecessors), gamma
                    )

    final_states = torch.stack(
        [
            R[k1][k2][T - 1][U - 1]
            for k1 in range(K1)
            for k2 in range(K2)
        ]
    )
    distance = softmin(final_states, gamma)

    if not return_accumulator:
        return distance

    accumulator = torch.stack(
        [
            torch.stack(
                [
                    torch.stack([torch.stack(R[k1][k2][t]) for t in range(T)])
                    for k2 in range(K2)
                ]
            )
            for k1 in range(K1)
        ]
    )
    return distance, accumulator
