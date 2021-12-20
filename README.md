# JEANIE
Implementation of JEANIE

## 1. The use of JEANIE

The following sample codes show how to use our proposed JEANIE for sequence alignment in 6D (similar to the use of softDTW):

Note that our JEANIE implementation based on [soft-DTW](https://github.com/Maghoumi/pytorch-softdtw-cuda) supports the pruning for difference lengths of features (the use of bandwidth setup).

```
from jeanie import SoftDTW
import torch

# each feature is in the shape of
# [batchsize, temp, view1, view2, featdim]
x1 = torch.rand(1, 3, 4, 4, 10)
x2 = torch.rand(1, 5, 1, 1, 10)

# similar to the use of any losses
criterion = SoftDTW(use_cuda=False, gamma=0.01, normalize=True, bandwidth = 1)
print(criterion(x1, x2))
```

## 2. Pre-trained models for smaller datasets

### 2.1 Some descriptions

We provide some sample pre-trained models. 

- MSRAction3D: Nway = 10, topkk = 3, 500 episodes
- 3DActionPairs: Nway = 6, topkk = 2, 200 episodes
- UWA3DActivity: Nway = 15, topkk = 5, 150 episodes

All evaluation uses the same setting: adam optimizer, video block length = 8 and overlap frame per block = 4, degrees for the SSGC is 6, viewing angles = [-pi / 180, 0, pi / 180], without the use of transformer.

For more details, please refer to our JEANIE paper.

### 2.2 Some performance on sample datasets

The experimental results reported here are without the use of hyperopt, and we simply set the viewing angles between -1 and 1 degrees.

|   | MSRAction3D | 3DActionPairs | UWA3DActivity |
| ------------- | :---: | :---: | :---: |
| softDTW (temp. align. only)  |  79.24 |  79.44 |  40.35 |
| Our JEANIE  |   |   |   |

#### Acknowledgement
Thanks to the implementation of [soft-DTW](https://github.com/Maghoumi/pytorch-softdtw-cuda).
