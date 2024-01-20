import pandas as pd
import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import numpy as np
from torch.utils.data.sampler import Sampler
import random
# for the viewpoint augmentation
from math import pi
import math


def rotationtrans(skeleton, flag):
    if flag == 'support':
        skeletons = torch.unsqueeze(torch.unsqueeze(skeleton, 0), 0)
    elif flag == 'query':
        degree = [-pi / 180, 0, pi / 180]
        # degree = [-pi / 90, -pi / 180, 0, pi / 180, pi / 90]
        # degree = [-pi / 60, -pi / 90, -pi / 180, 0, pi / 180, pi / 90, pi / 60]
        # degree = [-pi/30, -pi/36, -pi/45, -pi / 60, -pi / 90, -pi / 180, 0, pi / 180, pi / 90, pi / 60, pi/45, pi/36, pi/30]
        # degree = [-pi / 12, -pi / 15, -pi / 20, -pi / 30, -pi / 60, 0, pi / 60, pi / 30, pi / 20, pi / 15, pi / 12]
        # degree = [-pi/12, 0, pi/12]
        # degree = [-pi / 6, -pi / 12, 0, pi / 12, pi / 6]
        # degree = [-pi/4, -pi/6, -pi/12, 0, pi/12, pi/6, pi/4]
        # degree = [-pi/3, -pi/4, -pi/6, -pi/12, 0, pi/12, pi/6, pi/4, pi/3]
        # degree = [-5*pi/12, -pi/3, -pi/4, -pi/6, -pi/12, 0, pi/12, pi/6, pi/4, pi/3, 5*pi/12]
        # degree = [-pi/2, -5*pi/12, -pi/3, -pi/4, -pi/6, -pi/12, 0, pi/12, pi/6, pi/4, pi/3, 5*pi/12, pi/2]
        # define a matrix to store the rotated skeletons in [#viewpoints, #joints, 2D/3D, temporal]
        skeletons = torch.from_numpy(
            np.zeros((len(degree), len(degree), skeleton.shape[0], skeleton.shape[1], skeleton.shape[2])))
        for idx in range(len(degree)):
            d_idx = degree[idx]
            rotMy = torch.from_numpy(np.array(
                [[math.cos(d_idx), 0, -math.sin(d_idx)], [0, 1, 0], [math.sin(d_idx), 0, math.cos(d_idx)]])).float()
            for jdx in range(len(degree)):
                d_jdx = degree[jdx]
                rotMx = torch.from_numpy(np.array(
                    [[1, 0, 0], [0, math.cos(d_jdx), math.sin(d_jdx)], [0, -math.sin(d_jdx), math.cos(d_jdx)]])).float()
                # rotM = np.dot(np.dot(rotMx, rotMy), rotMz)
                rotM = torch.mm(rotMx, rotMy).float()

                for f in range(skeleton.shape[2]):
                    skeleton[:, :, f] = torch.mm(skeleton[:, :, f], rotM)
                skeletons[idx, jdx, :, :, :] = skeleton
    else:
        print('error in rotation trans.')

    return skeletons

def get_start_frame(total_frame, frame_per_block, overlap_frame):

    '''
    input:
    total number of frames -- total_frame
    the number of frames per video block -- frame_per_block
    the number of overlap frames -- overlap_frame

    output:
    the index of each starting frame of each video block
    in the form of a list -- start_index_list
    '''

    num_block = int((total_frame - frame_per_block) / (frame_per_block - overlap_frame) + 1)
    start_index_list = []

    for block_idx in range(num_block):
        start_frame = block_idx * (frame_per_block - overlap_frame)
        start_index_list.append(start_frame)

    if num_block < ((total_frame - frame_per_block) / (frame_per_block - overlap_frame) + 1):
        start_index_list.append(total_frame - frame_per_block)

    # print(start_index_list)

    return start_index_list

class ActionRecognitionDatasetTE(Dataset):
    def __init__(self, trte_csv_path, hdf5_data_path, dataset_name, transform=None):
        '''
        Args:
            trte_csv_path (string): path to the training or test split files
            hdf5_data_path (string): path to the hdf5 file (contains the key-value pairs of the data)
            dataset_name (string): the name of the dataset, it should be 'MSRAction3D, 3DActionPairs, UWA3DActivity, NTU-RGBD-60, NTU-RGBD-120'
            transform (callable, optional): optional transform to be applied on the data sample
        '''
        self.trte_list = pd.read_csv(trte_csv_path, header=None)
        self.data_path = hdf5_data_path
        self.dataset_name = dataset_name
        self.transform = transform

    def __len__(self):
        return len(self.trte_list)

    def __getitem__(self, idx):
        # print(idx, ' xxxxxxxxxx')
        video_name = self.trte_list.iloc[idx, 0]

        if self.dataset_name == 'NTU-RGBD-60' or self.dataset_name == 'NTU-RGBD-120':
            video_name = video_name[4:20] + video_name[0:4]

        # print(idx, ' | ', video_name)
        # to get the label of the video
        # labels should range from 0 to (num_class-1)
        # different datasets have different labelling schemes for labels
        if self.dataset_name == 'MSRAction3D' or self.dataset_name == '3DActionPairs':
            label = int(video_name[1:3]) - 1
        elif self.dataset_name == 'UWA3DActivity':
            label = int(video_name[1:3]) - 1
        elif self.dataset_name == 'NTU-RGBD-60' or self.dataset_name == 'NTU-RGBD-120':
            label = int(video_name[17:20]) - 1
        else:
            print('This dataset is not recorded!')

        # get the data from hdf5 file
        with h5py.File(self.data_path, "r") as f:
            if self.dataset_name == 'UWA3DActivity':
                video_name = video_name[6:9] + video_name[0:6]
            # print(video_name)
            skeleton = f[video_name + '/skeleton'][()]

            if self.dataset_name == 'NTU-RGBD-60' or self.dataset_name == 'NTU-RGBD-120':
                if f[video_name + '/num_subjects'][()] == 1:
                    skeleton = f[video_name + '/skeleton'][()][0:25, :, :]
            f.close()

        sample = {'skeleton': skeleton, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class NormalizationTE(object):
    def __init__(self, torso, njoints):
        self.torso = torso
        self.njoints = njoints

    def __call__(self, sample):
        skeleton, label = sample['skeleton'], sample['label']
        # if there are nans values in the data
        skeleton[np.isnan(skeleton)] = 0
        skeleton[np.isinf(skeleton)] = 0

        num_subject = int(skeleton.shape[0] / self.njoints)

        # normalize the joint w.r.t. the torso joint
        for sub_id in range(num_subject):
            skeleton[sub_id * self.njoints: sub_id * self.njoints + self.njoints, :, :] \
                = skeleton[sub_id * self.njoints: sub_id * self.njoints + self.njoints,:, :] \
                  - skeleton[self.torso + self.njoints * sub_id,:, :]

        # normalize the joint in each channel (x, y & z) into range (-1, 1)
        for slice in range(skeleton.shape[1]):
            skeleton[:, slice, :] = skeleton[:, slice, :] / (np.max(abs(skeleton[:, slice, :])) + 1e-10)

        sample = {'skeleton': skeleton, 'label': label}
        return sample


class ToTensorTE(object):
    def __call__(self, sample):
        skeleton, label = sample['skeleton'], sample['label']
        skeleton = torch.FloatTensor(skeleton).float()

        sample = {'skeleton': skeleton, 'label': label}
        return sample

def my_collateTE(batch):
    # define frame per block
    frame_per_block = 2
    # define overlap frames
    overlap_frame = 1
    data = []

    label = [item['label'] for item in batch]
    label = torch.LongTensor(label)

    lenbatch = len(label)


    for ii in range(lenbatch):
        if ii <= int(lenbatch/2):
            flag = 'support'
            skeleton = batch[ii]['skeleton']
        else:
            flag = 'query'
            skeleton = batch[ii]['skeleton']

        batch[ii]['skeleton'] = rotationtrans(skeleton, flag)


    for item in batch:
        video = item['skeleton']
        total_frame = video.shape[4]
        start_idx = get_start_frame(total_frame, frame_per_block, overlap_frame)

        num_blocks = len(start_idx)

        video_blocks = torch.zeros((num_blocks, video.shape[0], video.shape[1], video.shape[2], video.shape[3], frame_per_block)).float()
        for idx in range(num_blocks):
            video_block = video[:, :, :, :, start_idx[idx]:start_idx[idx] + frame_per_block]
            video_blocks[idx, :, :, :, :, :] = video_block
        data.append(video_blocks)



    # inside data is a list of [#blocks, #joints, 2D/3D, frame_per_block]
    return [data, label]
