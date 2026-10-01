# JEANIE

```text
JEANIE/
├── jeanie/
│   ├── __init__.py
│   ├── alignment.py       # JEANIE + soft-DTW
│   ├── fvm.py             # FVM
│   └── distances.py       # base-distance helpers
├── examples/
│   └── mini_demo.py       # very small runnable example
├── tests/
│   └── test_core.py       # correctness + gradient tests
├── README.md
└── requirements.txt
```

There are four main public entry points:

```python
jeanie_1d_from_cost(...)
jeanie_2d_from_cost(...)
fvm_1d_from_cost(...)
fvm_2d_from_cost(...)
```

---

## JEANIE

We first define the complete viewpoint-aware distance tensor in
Eq. (12), then give the concrete dynamic program in Algorithm 1.

For the single-viewpoint-axis form used to present Algorithm 1, let

```text
D[n, t, u] = d_base(query_view[n, t], support[u])
```

with shape

```text
[K, T, U]
```

where:

- `K` = number of query viewpoint states
- `T` = number of query temporal blocks
- `U` = number of support temporal blocks

JEANIE initializes every possible viewpoint:

```text
R[n, 0, 0] = D[n, 0, 0]
```

and recursively uses the predecessor set

```text
i ∈ {-eta, ..., +eta}
(j, k) ∈ {(0,1), (1,0), (1,1)}
```

so that

```text
R[n,t,u] =
    D[n,t,u]
    + SoftMin_gamma(
        R[n-i, t-j, u-k]
      ).
```

Out-of-range states are treated as `+inf`.

The final distance is

```text
SoftMin_gamma(R[:, T-1, U-1]).
```

This is the core of Algorithm 1.

### Two viewpoint axes

We describe the complete viewpoint representation as a `K × K'`
viewpoint grid. The repository therefore also provides:

```text
D[k1, k2, t, u]    # shape [K1, K2, T, U]
```

and the corresponding direct two-axis extension:

```python
jeanie_2d_from_cost(...)
```

with independent viewpoint shifts along both axes.

---

## FVM

FVM is the baseline in Eq. (13).

For every temporal pair `(t,u)`, FVM first performs a SoftMin over the
viewpoint indices independently:

```text
C[t,u] = SoftMin_gamma(viewpoint_costs[t,u])
```

and then runs soft-DTW on `C`.

This allows viewpoint selection to change freely from one
temporal alignment step to another.

The repository provides explicit wrappers for:

```python
fvm_1d_from_cost(...)
fvm_2d_from_cost(...)
```

It also supports query-only viewpoint tensors through:

```python
fvm_query_only_1d(...)
fvm_query_only_2d(...)
```

The full FVM layouts are:

```text
1-D viewpoints:
D.shape = [K_query, K_support, T, U]

2-D viewpoints:
D.shape = [Kq1, Kq2, Ks1, Ks2, T, U]
```

For the common query-only special cases:

```text
1-D: D.shape = [K, T, U]
2-D: D.shape = [K1, K2, T, U]
```

---

## Installation

The code uses Python-3.7-compatible syntax. Install a PyTorch release compatible with your local Python version.

Install PyTorch and the test dependency:

```bash
pip install torch pytest
```

For an editable local install:

```bash
pip install -e .
```

No CUDA, Numba, compiled extensions, or external repository code is required.

---

## Very small example

From the repository root:

```bash
python examples/mini_demo.py
```

You can also run:

```bash
python -m examples.mini_demo
```

The demo uses a tiny synthetic sequence. It is lightweight so
that the core API is easy to inspect before plugging in a real skeleton
encoder.

The core usage is:

```python
from jeanie import (
    euclidean_cost,
    jeanie_1d_from_cost,
    fvm_query_only_1d,
)

# query_features:  [K, T, D]
# support_features:[U, D]

D = euclidean_cost(query_features, support_features)

d_jeanie = jeanie_1d_from_cost(
    D,
    gamma=0.1,
    max_shift=1,
)

d_fvm = fvm_query_only_1d(
    D,
    gamma=0.1,
)
```

For the full one-axis FVM case, where both query and support have viewpoint
states, use `fvm_1d_from_cost` with cost shape `[K_query, K_support, T, U]`.
For two viewpoint axes, use `fvm_2d_from_cost` with cost shape
`[Kq1, Kq2, Ks1, Ks2, T, U]`.

Both outputs remain differentiable with respect to the input features.

---

## Using real 3D skeletons

A simple downstream pipeline is:

```text
3D skeleton sequence
        │
        ├── temporal blocking
        │
        ├── viewpoint simulation
        │
        ├── feature encoder
        │
        ▼
query features  [K, T, D]
support features[U, D]
        │
        ▼
base-distance tensor D
        │
        ├───────────────┐
        ▼               ▼
    JEANIE             FVM
```

The alignment module does **not** require a particular skeleton encoder.

For example, if a skeleton block has `J` joints:

```python
# [K, T, J, 3] -> [K, T, 3J]
query_features = query_blocks.reshape(K, T, 3 * J)

# [U, J, 3] -> [U, 3J]
support_features = support_blocks.reshape(U, 3 * J)
```

and then:

```python
D = euclidean_cost(query_features, support_features)
distance = jeanie_1d_from_cost(D, gamma=0.1, max_shift=1)
```

In a learned model, `query_features` and `support_features` can instead be
outputs of a GNN, MLP, transformer, or any other differentiable encoder.

---

## Computational note

The DP is implemented in plain PyTorch/Python loops. This is appropriate as a
reference implementation and for small experiments, but it is not intended
to replace highly optimized GPU/CUDA kernels for large-scale training.

For large downstream workloads, the reference implementation can serve as a
clear specification against which an optimized implementation can be tested.

---

## Soft-min can be negative

JEANIE and FVM use a soft minimum of the form

```text
SoftMin_gamma(x) =
    -gamma * log(sum(exp(-x / gamma))).
```

As with soft-DTW, the resulting value can be below the minimum individual
cost because of the entropy/smoothing term. A negative raw value is therefore
not, by itself, an implementation error.

---

## Citation

If this implementation is useful in your work, please cite the JEANIE paper:

```bibtex
@article{wang2024meet,
  title={Meet jeanie: a similarity measure for 3d skeleton sequences via temporal-viewpoint alignment},
  author={Wang, Lei and Liu, Jun and Zheng, Liang and Gedeon, Tom and Koniusz, Piotr},
  journal={International Journal of Computer Vision},
  volume={132},
  number={9},
  pages={4091--4122},
  year={2024},
  publisher={Springer}
}
```

```bibtex
@inproceedings{wang2022temporal,
  title={Temporal-viewpoint transportation plan for skeletal few-shot action recognition},
  author={Wang, Lei and Koniusz, Piotr},
  booktitle={Asian Conference on Computer Vision},
  pages={307--326},
  year={2022},
  organization={Springer}
}
```
