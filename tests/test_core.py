
import pytest
import torch

from jeanie import (
    fvm_1d_from_cost,
    fvm_2d_from_cost,
    fvm_from_cost,
    fvm_query_only_1d,
    fvm_query_only_2d,
    jeanie_1d_from_cost,
    jeanie_2d_from_cost,
    soft_dtw,
    softmin,
)


DTYPE = torch.float64


def _all_temporal_paths(T, U):
    paths = []

    def visit(t, u, path):
        if t == T - 1 and u == U - 1:
            paths.append(tuple(path))
            return

        for dt, du in ((0, 1), (1, 0), (1, 1)):
            nt, nu = t + dt, u + du
            if nt < T and nu < U:
                visit(nt, nu, path + [(nt, nu)])

    visit(0, 0, [(0, 0)])
    return paths


def _all_jeanie_paths_1d(K, T, U, max_shift):
    paths = []

    def visit(n, t, u, path):
        if t == T - 1 and u == U - 1:
            paths.append(tuple(path))
            return

        for dt, du in ((0, 1), (1, 0), (1, 1)):
            nt, nu = t + dt, u + du
            if nt >= T or nu >= U:
                continue

            for dn in range(-max_shift, max_shift + 1):
                nn = n + dn
                if 0 <= nn < K:
                    visit(nn, nt, nu, path + [(nn, nt, nu)])

    for n in range(K):
        visit(n, 0, 0, [(n, 0, 0)])

    return paths


def _all_jeanie_paths_2d(K1, K2, T, U, shift1, shift2):
    paths = []

    def visit(k1, k2, t, u, path):
        if t == T - 1 and u == U - 1:
            paths.append(tuple(path))
            return

        for dt, du in ((0, 1), (1, 0), (1, 1)):
            nt, nu = t + dt, u + du
            if nt >= T or nu >= U:
                continue

            for dk1 in range(-shift1, shift1 + 1):
                nk1 = k1 + dk1
                if not (0 <= nk1 < K1):
                    continue

                for dk2 in range(-shift2, shift2 + 1):
                    nk2 = k2 + dk2
                    if not (0 <= nk2 < K2):
                        continue

                    visit(
                        nk1,
                        nk2,
                        nt,
                        nu,
                        path + [(nk1, nk2, nt, nu)],
                    )

    for k1 in range(K1):
        for k2 in range(K2):
            visit(k1, k2, 0, 0, [(k1, k2, 0, 0)])

    return paths


def _soft_path_value(values, gamma):
    return -gamma * torch.logsumexp(
        -torch.stack([v if torch.is_tensor(v) else torch.as_tensor(v, dtype=DTYPE)
                      for v in values]) / gamma,
        dim=0,
    )


@pytest.mark.parametrize(
    "K,T,U,max_shift,gamma",
    [
        (1, 1, 1, 0, 0.3),
        (2, 2, 2, 0, 0.3),
        (3, 2, 3, 1, 0.2),
        (3, 3, 3, 1, 0.4),
        (4, 3, 3, 2, 0.25),
    ],
)
def test_jeanie_1d_matches_bruteforce(K, T, U, max_shift, gamma):
    torch.manual_seed(10 + K + T + U)
    cost = torch.rand(K, T, U, dtype=DTYPE)

    actual = jeanie_1d_from_cost(cost, gamma, max_shift)

    values = []
    for path in _all_jeanie_paths_1d(K, T, U, max_shift):
        values.append(sum(cost[n, t, u] for n, t, u in path))

    expected = _soft_path_value(values, gamma)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    "T,U,gamma",
    [(1, 1, 0.2), (2, 3, 0.25), (3, 3, 0.4)],
)
def test_soft_dtw_matches_bruteforce(T, U, gamma):
    torch.manual_seed(T + U)
    cost = torch.rand(T, U, dtype=DTYPE)

    actual = soft_dtw(cost, gamma)

    values = [
        sum(cost[t, u] for t, u in path)
        for path in _all_temporal_paths(T, U)
    ]
    expected = _soft_path_value(values, gamma)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    "K,T,U,gamma",
    [(2, 2, 2, 0.3), (3, 2, 3, 0.2), (2, 3, 3, 0.4)],
)
def test_fvm_1d_matches_eq13_plus_bruteforce(K, T, U, gamma):
    torch.manual_seed(K + T + U)
    cost = torch.rand(K, T, U, dtype=DTYPE)

    temporal_cost = softmin(cost, gamma, dim=0)
    expected = _soft_path_value(
        [
            sum(temporal_cost[t, u] for t, u in path)
            for path in _all_temporal_paths(T, U)
        ],
        gamma,
    )

    actual = fvm_query_only_1d(cost, gamma)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)



def test_fvm_full_1d_matches_eq13_plus_bruteforce():
    """Full 1-D query/support viewpoint matching."""
    torch.manual_seed(13)
    cost = torch.rand(2, 2, 2, 3, dtype=DTYPE)
    gamma = 0.25

    # Eq. (13): SoftMin over both query/support viewpoint indices.
    temporal_cost = softmin(
        cost.reshape(4, 2, 3),
        gamma,
        dim=0,
    )
    expected = _soft_path_value(
        [
            sum(temporal_cost[t, u] for t, u in path)
            for path in _all_temporal_paths(2, 3)
        ],
        gamma,
    )

    actual = fvm_1d_from_cost(cost, gamma)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_fvm_full_2d_matches_eq13_plus_bruteforce():
    """Full 2-D query/support viewpoint matching."""
    torch.manual_seed(14)
    cost = torch.rand(2, 2, 2, 2, 2, 2, dtype=DTYPE)
    gamma = 0.2

    temporal_cost = softmin(
        cost.reshape(16, 2, 2),
        gamma,
        dim=0,
    )
    expected = _soft_path_value(
        [
            sum(temporal_cost[t, u] for t, u in path)
            for path in _all_temporal_paths(2, 2)
        ],
        gamma,
    )

    actual = fvm_2d_from_cost(cost, gamma)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_jeanie_2d_matches_bruteforce():
    torch.manual_seed(4)
    K1, K2, T, U = 2, 2, 2, 3
    cost = torch.rand(K1, K2, T, U, dtype=DTYPE)

    gamma = 0.3
    shift1 = 1
    shift2 = 0

    actual = jeanie_2d_from_cost(
        cost, gamma, shift1, shift2
    )

    values = []
    for path in _all_jeanie_paths_2d(
        K1, K2, T, U, shift1, shift2
    ):
        values.append(sum(cost[k1, k2, t, u] for k1, k2, t, u in path))

    expected = _soft_path_value(values, gamma)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_fvm_2d_query_only_matches_viewpoint_softmin_then_dtw():
    torch.manual_seed(5)
    cost = torch.rand(2, 2, 2, 3, dtype=DTYPE)
    gamma = 0.2

    temporal_cost = softmin(
        cost.reshape(4, 2, 3),
        gamma,
        dim=0,
    )
    expected = soft_dtw(temporal_cost, gamma)

    actual = fvm_query_only_2d(cost, gamma)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_fvm_full_1d_wrapper():
    torch.manual_seed(6)
    cost = torch.rand(2, 3, 2, 3, dtype=DTYPE)
    assert torch.allclose(
        fvm_1d_from_cost(cost, 0.3),
        fvm_from_cost(cost, 0.3),
        atol=1e-12,
        rtol=1e-12,
    )


def test_fvm_full_2d_wrapper():
    torch.manual_seed(7)
    cost = torch.rand(2, 2, 2, 2, 2, 3, dtype=DTYPE)
    assert torch.allclose(
        fvm_2d_from_cost(cost, 0.3),
        fvm_from_cost(cost, 0.3),
        atol=1e-12,
        rtol=1e-12,
    )


def test_jeanie_gradcheck():
    torch.manual_seed(8)
    cost = torch.rand(2, 2, 2, dtype=DTYPE, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda x: jeanie_1d_from_cost(x, gamma=0.3, max_shift=1),
        (cost,),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-5,
    )


def test_fvm_gradcheck():
    torch.manual_seed(9)
    cost = torch.rand(2, 2, 2, dtype=DTYPE, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda x: fvm_query_only_1d(x, gamma=0.3),
        (cost,),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-5,
    )


def test_jeanie_2d_gradcheck():
    torch.manual_seed(10)
    cost = torch.rand(2, 2, 2, 2, dtype=DTYPE, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda x: jeanie_2d_from_cost(
            x,
            gamma=0.3,
            max_shift_az=1,
            max_shift_alt=1,
        ),
        (cost,),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-5,
    )


def test_backward_produces_finite_gradients():
    torch.manual_seed(11)
    cost = torch.rand(3, 3, 4, dtype=DTYPE, requires_grad=True)

    distance = jeanie_1d_from_cost(cost, gamma=0.2, max_shift=1)
    distance.backward()

    assert cost.grad is not None
    assert torch.isfinite(cost.grad).all()


def test_edge_case_singleton():
    cost = torch.tensor([[[2.0]]], dtype=DTYPE)

    assert torch.allclose(
        jeanie_1d_from_cost(cost, gamma=0.1, max_shift=0),
        torch.tensor(2.0, dtype=DTYPE),
    )
    assert torch.allclose(
        fvm_query_only_1d(cost, gamma=0.1),
        torch.tensor(2.0, dtype=DTYPE),
    )


def test_zero_view_shift_reduces_to_softmin_of_fixed_view_soft_dtw():
    torch.manual_seed(12)
    cost = torch.rand(3, 3, 3, dtype=DTYPE)
    gamma = 0.3

    jeanie = jeanie_1d_from_cost(cost, gamma=gamma, max_shift=0)

    fixed_view_distances = torch.stack([
        soft_dtw(cost[k], gamma)
        for k in range(cost.shape[0])
    ])
    expected = softmin(fixed_view_distances, gamma)

    assert torch.allclose(jeanie, expected, atol=1e-10, rtol=1e-10)
