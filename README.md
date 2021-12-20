# JEANIE
Implementation of JEANIE

## The use of JEANIE

The following sample codes show how to use our proposed JEANIE for sequence alignment in 6D:

```
from jeanie import SoftDTW
import torch

x1 = torch.rand(1, 3, 4, 4, 10)
x2 = torch.rand(1, 5, 1, 1, 10)

cc = SoftDTW(use_cuda=False, gamma=0.01, normalize=True, bandwidth = 1)
print(cc(x1, x2))
```

## Pre-trained models for smaller datasets

We provide some sample pre-trained models. 

- MSRAction3D: Nway = 10, topkk = 3, 500 episodes
- 3DActionPairs: Nway = 6, topkk = 2, 200 episodes
- UWA3DActivity: Nway = 15, topkk = 5, 150 episodes

All evaluation uses the same setting: adam optimizer, video block length = 8 and overlap frame per block = 4, degrees for the SSGC is 6, viewing angles = [-pi / 180, 0, pi / 180], without the use of transformer.

\*_jeanie.pt model uses the loss:

l(\vd^{+}\!,\vd^{-}) = (max(0, \mu(\vd^{+})-\detach(\mu(\topminb(\vd^{+}))))
+ max(0, \detach(\mu(\topmaxbb(\vd^{-})))-\mu(\vd^{-})))^2
