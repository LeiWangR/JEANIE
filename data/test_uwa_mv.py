import numpy as np
import h5py
import os
from scipy.io import loadmat

dataset_name = 'UWA3DMultiviewActivityII'
# dataset_path = '/Users/wanglei/Desktop/UWA3DMultiviewActivityII_skeleton'

hdf5_dataset_name = dataset_name + '.hdf5'
# ---------- check if I can read the skeleton data
with h5py.File(hdf5_dataset_name, "r") as ff:
    for a_key in ff.keys():
        frame_num = ff[a_key + '/frame_num'][()]
        num_joints = ff[a_key + '/num_joints'][()]
        skeleton = ff[a_key + '/skeleton'][()]
        print(a_key, ' | frame num.: ', frame_num, ' | joints: ', num_joints, ' | shape: ', skeleton.shape, '\n' )
    ff.close()

print('read success')
