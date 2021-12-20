import pandas as pd
import h5py
import torch
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.sampler import Sampler
import random
# for the viewpoint augmentation
from math import pi
import math

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


class ActionRecognitionDataset(Dataset):
    def __init__(self, trte_csv_path, hdf5_data_path, dataset_name, transform=None):
        '''
        Args:
            trte_csv_path (string): path to the training or test split files
            hdf5_data_path (string): path to the hdf5 file
            (contains the key-value pairs of the data)
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
        video_name = self.trte_list.iloc[idx, 0]

        if self.dataset_name == 'NTU-RGBD-60' or self.dataset_name == 'NTU-RGBD-120':
            video_name = video_name[4:20] + video_name[0:4]
        # the names of video samples are different
        # convert labels into range between 0 and (num_class-1)
        if self.dataset_name == 'MSRAction3D' or self.dataset_name == '3DActionPairs':
            label = int(video_name[1:3]) - 1
        elif self.dataset_name == 'UWA3DActivity':
            label = int(video_name[1:3]) - 1
        elif self.dataset_name == 'NTU-RGBD-60' or self.dataset_name == 'NTU-RGBD-120':
            label = int(video_name[17:20]) - 1
        else:
            print('Please add this dataset into evaluation!')

        # get the data from hdf5 file
        with h5py.File(self.data_path, "r") as f:
            if self.dataset_name == 'UWA3DActivity':
                video_name = video_name[6:9] + video_name[0:6]
            skeleton = f[video_name + '/skeleton'][()]

            if self.dataset_name == 'NTU-RGBD-60' or self.dataset_name == 'NTU-RGBD-120':
                if f[video_name + '/num_subjects'][()] == 1:
                    # if there are only 1 performing subject
                    skeleton = f[video_name + '/skeleton'][()][0:25, :, :]

            f.close()

        sample = {'skeleton': skeleton, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Normalization(object):
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

class RotationTransform(object):
    def __call__(self, sample):
        skeleton, label = sample['skeleton'], sample['label']

        degree = [-pi/12, 0, pi/12]
        # degree = [-pi / 6, -pi / 12, 0, pi / 12, pi / 6]
        # degree = [-pi/4, -pi/6, -pi/12, 0, pi/12, pi/6, pi/4]
        # degree = [-pi/3, -pi/4, -pi/6, -pi/12, 0, pi/12, pi/6, pi/4, pi/3]
        # degree = [-5*pi/12, -pi/3, -pi/4, -pi/6, -pi/12, 0, pi/12, pi/6, pi/4, pi/3, 5*pi/12]
        # degree = [-pi/2, -5*pi/12, -pi/3, -pi/4, -pi/6, -pi/12, 0, pi/12, pi/6, pi/4, pi/3, 5*pi/12, pi/2]

        # define a matrix to store the rotated skeletons in [#viewpoints, #joints, 2D/3D, temporal]
        skeletons = np.zeros((len(degree), skeleton.shape[0], skeleton.shape[1], skeleton.shape[2]))
        for idx in range(len(degree)):
            d_idx = degree[idx]
            rotMy = np.array([[math.cos(d_idx), 0, -math.sin(d_idx)], [0, 1, 0], [math.sin(d_idx), 0, math.cos(d_idx)]])
            rotMx = np.array([[1, 0, 0], [0, math.cos(d_idx), math.sin(d_idx)], [0, -math.sin(d_idx), math.cos(d_idx)]])
            rotMz = np.array([[math.cos(d_idx), math.sin(d_idx), 0], [-math.sin(d_idx), math.cos(d_idx), 0], [0,0,1]])
            rotM = np.dot(np.dot(rotMx, rotMy), rotMz)
            for f in range(skeleton.shape[2]):
                skeleton[:, :, f] = np.dot(skeleton[:, :, f], rotM)
            skeletons[idx, :, :, :] = skeleton

        sample = {'skeleton': skeletons, 'label': label}
        return sample

class ToTensor(object):
    def __call__(self, sample):
        skeleton, label = sample['skeleton'], sample['label']
        skeleton = torch.FloatTensor(skeleton).float()
        sample = {'skeleton': skeleton, 'label': label}
        return sample


def my_collate(batch):
    # define frame per block
    frame_per_block = 2
    # define overlap frames
    overlap_frame = 1
    data = []

    for item in batch:
        video = item['skeleton']
        total_frame = video.shape[2]
        start_idx = get_start_frame(total_frame, frame_per_block, overlap_frame)

        num_blocks = len(start_idx)
        video_blocks = torch.zeros((num_blocks, video.shape[0], video.shape[1], frame_per_block)).float()
        for idx in range(num_blocks):
            video_block = video[:, :, start_idx[idx]:start_idx[idx] + frame_per_block]
            video_blocks[idx, :, :, :] = video_block
        data.append(video_blocks)

    label = [item['label'] for item in batch]
    label = torch.LongTensor(label)
    # inside data is a list of
    # [#blocks, #joints, 2D/3D, frame_per_block]
    return [data, label]

class ClassBalancedSampler(Sampler):
    '''
    Samples 'num_inst' examples each from 'num_cl' pools
    of examples of size 'num_per_class'
    '''

    # num_count is the npy for storing
    # the number of samples per class in the form of integer array
    def __init__(self, s_num_per_class, q_num_per_class, num_cl, numclass, npy_file, shuffle=True):
        self.s_num_per_class = s_num_per_class
        self.q_num_per_class = q_num_per_class
        self.num_cl = num_cl
        self.numclass = numclass
        self.npy_file = npy_file
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        count = np.load(self.npy_file)
        if self.shuffle:
            batch = [[(i + np.sum(count[:j])).item() for i in
                      torch.randperm(count[j].item())[:(self.s_num_per_class + self.q_num_per_class)]] for j in
                     range(self.num_cl)]

        else:
            batch = [
                [i + np.sum(count[:j]) for i in range(count[j].item())[:(self.s_num_per_class + self.q_num_per_class)]]
                for j in range(self.num_cl)]

        batch = random.sample(batch, self.numclass)
        batch = np.array(batch).T.tolist()

        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            batch_s = batch[0:self.s_num_per_class * self.numclass]
            batch_q = batch[self.s_num_per_class * self.numclass:]
            random.shuffle(batch_s)
            random.shuffle(batch_q)
            batch = batch_s + batch_q

        return iter(batch)

    def __len__(self):
        return 1


def get_dataloader(s_num_per_class, q_num_per_class, num_class, numclass, trte_list, data_dir, dataset_name, npy_file,
                   torso, njoints):
    # num_class is the total number of action classes
    # training & test split, each has num_class/2 action classes in total

    data = ActionRecognitionDataset(trte_csv_path=trte_list, hdf5_data_path=data_dir, dataset_name=dataset_name,
                                    transform=transforms.Compose([Normalization(torso, njoints), ToTensor()]))

    # for smaller datasets, half of the samples for training
    # and the rest half for testing
    if dataset_name == 'MSRAction3D' or dataset_name == '3DActionPairs' or dataset_name == 'UWA3DActivity':
        num_class = round(num_class / 2)
    elif dataset_name == 'NTU-RGBD-60':
        num_class = 50
    elif dataset_name == 'NTU-RGBD-120':
        num_class = 100
    else:
        print('No such a dataset')
    sampler = ClassBalancedSampler(s_num_per_class, q_num_per_class, num_class, numclass, npy_file, shuffle=True)
    loader = DataLoader(data, shuffle=False, num_workers=4, collate_fn=my_collate, pin_memory=False,
                        batch_size=(s_num_per_class + q_num_per_class) * numclass, sampler=sampler)
    return loader