import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import scipy.sparse as sp
import networkx as nx

from args import get_args

from data import *
from dataTE import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from train import train
from test import test

from model import get_model

import os





args = get_args()

EPISODE = args.episode
CLASS_NUM = args.Nway
SUPPORT_NUM_PER_CLASS = args.Kshot
QUERY_NUM_PER_CLASS = args.Qnum
TEST_EPISODE = args.test_episode

gamma = args.gamma
kk = args.topkk

ndim = 3  # 3D coordinates, for 2D would be 2 here
nframe = 2  # number of frame per block (refer to dataloader my_collate func. for overlap & frame_per_block)
# nfeat = 50 for small dataset and 100 for big dataset
nfeat = 10

cuda = args.withcuda

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

set_seed(args.seed, cuda)
dataset_name = args.dataset
batch = args.batch
# epochs=args.epochs
weight_decay = args.weight_decay
lr = args.lr
degree = args.degree
test_episode_size = args.test_episode_size


alpha = args.alpha

path = './trtesplit'
npy_file_tr = os.path.join(path, dataset_name + '_tr_count.npy')
npy_file_te = os.path.join(path, dataset_name + '_te_count.npy')

data_dir = './data/' + dataset_name + '.hdf5'
tr_te_split_dir = './trtesplit/'

tr_list = tr_te_split_dir + dataset_name + '_fsl_train.txt'
te_list = tr_te_split_dir + dataset_name + '_fsl_test.txt'

exampler = tr_te_split_dir + 'exampler_' + dataset_name + '.txt'


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def fetch_normalization(type):
    switcher = {'AugNormAdj': aug_normalized_adjacency, }  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    func = switcher.get(type, lambda: "Invalid normalization technique.")
    return func


def adj_compute(graph, normalization):
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # normalized adjacency matrix
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    # convert the numpy array to torch tensor
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    return adj


# weight initialization
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


if dataset_name == 'MSRAction3D':
    num_class = 20
elif dataset_name == '3DActionPairs':
    num_class = 12
elif dataset_name == 'UWA3DActivity':
    num_class = 30
elif dataset_name == 'NTU-RGBD-60':
    num_class = 60
else:
    print('add the number of action classes for this dataset!')

if dataset_name == 'MSRAction3D':
    graph = {0: [2, 7], 1: [2, 8], 2: [0, 1, 3, 19], 3: [2, 6], 4: [6, 13], 5: [6, 14], 6: [3, 4, 5], 7: [0, 9],
             8: [1, 10], 9: [7, 11], 10: [8, 12], 11: [9], 12: [10], 13: [4, 15], 14: [5, 16], 15: [13, 17],
             16: [14, 18], 17: [15], 18: [16], 19: [2]}
elif dataset_name == '3DActionPairs':
    graph = {0: [1, 12, 16], 1: [0, 2], 2: [1, 3, 4, 8], 3: [2], 4: [2, 5], 5: [4, 6], 6: [5, 7], 7: [6], 8: [2, 9],
             9: [8, 10], 10: [9, 11], 11: [10], 12: [0, 13], 13: [12, 14], 14: [13, 15], 15: [14], 16: [0, 17],
             17: [16, 18], 18: [17, 19], 19: [18]}
elif dataset_name == 'UWA3DActivity':
    graph = {0: [1], 1: [0, 2, 5, 8], 2: [1, 3], 3: [2, 4], 4: [3], 5: [1, 6], 6: [5, 7], 7: [6], 8: [1, 9, 12],
             9: [8, 10], 10: [9, 11], 11: [10], 12: [8, 13], 13: [12, 14], 14: [13]}
elif dataset_name == 'NTU-RGBD-60':
    graph = {0: [1, 12, 16], 1: [0, 20], 2: [3, 20], 3: [2], 4: [5, 20], 5: [4, 6], 6: [5, 7], 7: [6, 22], 8: [9, 20],
             9: [8, 10], 10: [9, 11], 11: [10, 24], 12: [0, 13], 13: [12, 14], 14: [13, 15], 15: [14], 16: [0, 17],
             17: [16, 18], 18: [17, 19], 19: [18], 20: [1, 2, 4, 8], 21: [22], 22: [7, 21], 23: [24], 24: [11, 23]}
else:
    print('no graph for ', dataset_name, ' please add graph!')

if dataset_name == 'MSRAction3D':
    torso = 3
    num_joints = 20
elif dataset_name == '3DActionPairs':
    torso = 1
    num_joints = 20
elif dataset_name == 'UWA3DActivity':
    torso = 8
    num_joints = 15
elif dataset_name == 'NTU-RGBD-60':
    torso = 1  # & 26
    num_joints = 25
else:
    print('specify the torso joint and total number of body joints for this dataset!')

adj = adj_compute(graph, normalization='AugNormAdj')

model = get_model(num_joints, ndim, nframe, num_class, nfeat, cuda=cuda)

model.apply(weight_init)
# load pre-trained model
'''
print('load pretrained model')

model.load_state_dict(torch.load('/Users/wanglei/Desktop/2311_fewshot/models/MSRAction3D-nframe-10-degree-5-epochs-2000.pt'))

print('load success')
'''

PATH = 'models/' + dataset_name + '-nf' + str(nframe) + '-d' + str(degree) + '-' + str(CLASS_NUM) + 'way-' + str(
    SUPPORT_NUM_PER_CLASS) + 'shot' + '.pt'

print(PATH)

print('======Train======')

best_acc = 0
best_episode = 0

data_exampler = ActionRecognitionDatasetTE(trte_csv_path = exampler, hdf5_data_path = data_dir, dataset_name = dataset_name, transform = transforms.Compose([NormalizationTE(torso, num_joints), ToTensorTE()]))
dataloader_exampler = DataLoader(data_exampler, batch_size = CLASS_NUM, shuffle = False, num_workers = 4, collate_fn=my_collateTE, pin_memory=False)

data_te = ActionRecognitionDatasetTE(trte_csv_path = te_list, hdf5_data_path = data_dir, dataset_name = dataset_name, transform = transforms.Compose([NormalizationTE(torso, num_joints), ToTensorTE()]))
dataloader_te = DataLoader(data_te, batch_size = batch, shuffle = False, num_workers = 4, collate_fn=my_collateTE, pin_memory=False)

for episode in range(EPISODE):
    # print(episode, '------')

    loss, model = train(model, tr_list, data_dir, dataset_name, npy_file_tr, torso, SUPPORT_NUM_PER_CLASS,
                        QUERY_NUM_PER_CLASS, CLASS_NUM, adj, degree, weight_decay, lr, num_class, PATH, best_acc, kk,
                        gamma, cuda, num_joints, alpha)
    print('episode: {} | loss: {:.6f}'.format(episode + 1, loss.item()))

    if (episode + 1) % test_episode_size == 0:
        print('======Test======')
        acc = test(dataloader_exampler, dataloader_te, model, torso, adj, degree, batch, num_class, TEST_EPISODE, SUPPORT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS, CLASS_NUM, alpha, gamma, cuda)

        acc_val = acc
        print('Total Accuracy: {:.4f}'.format(acc_val))

        if acc_val > best_acc:
            torch.save(model.state_dict(), PATH)
            best_acc = acc_val
        print('current best accuracy is: {:0.4f}'.format(best_acc))

print('--- Overall best accuracy is: {:0.4f}'.format(best_acc))


