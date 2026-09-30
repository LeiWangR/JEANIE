"""A small JEANIE/FVM demo.

Run from the repository root:

    python examples/mini_demo.py
"""

import math
import sys
from pathlib import Path

import torch

# Allow direct execution from the repository root:
#     python examples/mini_demo.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jeanie import euclidean_cost, jeanie_1d_from_cost, fvm_query_only_1d


def make_tiny_motion(t):
    """Return a tiny 3-joint 3D motion: [T, J, 3]."""
    x = torch.zeros(t.numel(), 3, 3)

    # Root
    x[:, 0] = torch.stack(
        [0.05 * torch.sin(2 * math.pi * t), 1.0 + 0.02 * t, 0.0 * t],
        dim=-1,
    )

    # Left/right joint
    x[:, 1] = torch.stack(
        [-0.25 + 0.05 * torch.sin(2 * math.pi * t), 0.6 + 0.1 * t, 0.0 * t],
        dim=-1,
    )
    x[:, 2] = torch.stack(
        [0.25 - 0.05 * torch.sin(2 * math.pi * t), 0.6 + 0.1 * t, 0.0 * t],
        dim=-1,
    )
    return x


def rotate_y(x, degrees):
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    rotation = x.new_tensor([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])
    return x @ rotation.T


def main():
    torch.manual_seed(0)

    # Support: 6 temporal blocks, one observed viewpoint.
    support = make_tiny_motion(torch.linspace(0, 1, 6))

    # Query: 5 temporal blocks, viewed from a different angle.
    query = rotate_y(make_tiny_motion(torch.linspace(0, 1, 5)), 20.0)
    query = query + 0.005 * torch.randn_like(query)

    # Simulate three query viewpoints.
    angles = (-20.0, 0.0, 20.0)
    query_views = torch.stack(
        [rotate_y(query, a) for a in angles],
        dim=0,
    )  # [K, T, J, 3]

    # Flatten each skeleton block into a feature vector.
    query_features = query_views.reshape(len(angles), 5, -1)
    support_features = support.reshape(6, -1)

    cost = euclidean_cost(query_features, support_features)

    d_jeanie = jeanie_1d_from_cost(
        cost,
        gamma=0.1,
        max_shift=1,
    )
    d_fvm = fvm_query_only_1d(
        cost,
        gamma=0.1,
    )

    print("cost shape :", tuple(cost.shape))
    print("JEANIE     :", float(d_jeanie.detach()))
    print("FVM        :", float(d_fvm.detach()))


if __name__ == "__main__":
    main()
