# JEANIE
Implementation of JEANIE

## 0. Environment setup

- Create the environment from the *myenv.yml* file:  `conda env create -f myenv.yml`
- Activate the new environment via: `conda activate myenv`

## 1. The use of JEANIE

The following sample codes show how to use our proposed JEANIE for sequence alignment in 6D (similar to the use of soft-DTW):

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

## 2. Datasets and evalution protocols

### 2.1 Smaller datasets in hdf5 format

We provide smaller datasets in the data/ folder which you can use to reproduce the results in the following sections.

### 2.2 Evaluation protocols for smaller datasets

We also provide sample evaluation protocols for one-shot learning on smaller datasets. We provide the *exampler* for evaluation (similar to NTU-120 one-shot protocol, refer to trtesplit/ folder for more details).

## 3. Run the sample codes on smaller datasets (on CPU)

To run the temporal alignment (the use of soft-DTW) only: `python3 main.py --Nway 10 --topkk 3 --dataset MSRAction3D`

To run the JEANIE: `python3 mainJEANIE.py --Nway 10 --topkk 3 --dataset MSRAction3D`


## 4. Pre-trained models for smaller datasets (on CPU)

### 4.1 Some descriptions

We provide some sample pre-trained models. 

- MSRAction3D: Nway = 10, topkk = 3
- 3DActionPairs: Nway = 6, topkk = 2
- UWA3DActivity: Nway = 15, topkk = 5

All evaluation uses the same setting: adam optimizer, video block length = 8 and overlap frame per block = 4, degrees for the SSGC is 6, alpha for SSGC is set to 0.7, viewing angles = [-pi / 180, 0, pi / 180], without the use of transformer.

For more details, please refer to our JEANIE paper.

### 4.2 One-shot performance on sample datasets

**The experimental results reported here are without the use of hyperopt, and we simply set the viewing angles between -1 and 1. We use Euler angles for the viewpoint augmentation in the following table.**

Note that M and S in the table represents the frame counts per temporal block and stride step, respectively. For more views mentioned in the table, we set the viewing angles between -2 and 2.

The use of soft-DTW is to only align the temporal information, whereas the use of JEANIE is to jointly align the temporal and viewpoint information. More details please refer to our paper.

|   | MSRAction3D | 3DActionPairs | UWA3DActivity | Model provided|
| ------------- | :---: | :---: | :---: | :---: |
| soft-DTW (M = 2, S = 1)  |  79.24 |  79.44 |  40.35 | Yes |
| soft-DTW (M = 8, S = 4)  | 72.66  |  77.78 |  42.40 | Yes |
| Our JEANIE (M = 8, S = 4)  |  73.70 |  82.78 |  42.11 | Yes |
| Our JEANIE (M = 8, S = 2) | 75.78 |  78.33 | 40.94 | Yes |
| Our JEANIE (M = 10, S = 2) |  - |  - |  43.57 | Yes|
| Our JEANIE (M = 12, S = 2) |  - |  81.67 |  - | No |
| Our JEANIE (M = 15, S = 5) |  - |  82.22 | -  | Yes|

#### Acknowledgement
Thanks to the implementation of [soft-DTW](https://github.com/Maghoumi/pytorch-softdtw-cuda).
