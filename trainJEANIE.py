from testJEANIE import test
import torch
import torch.nn.functional as F
import torch.optim as optim
from jeanie import SoftDTW
import math

import numpy as np

from torch.autograd import Variable

from dataJEANIE import *
from modelJEANIE import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, tr_list, data_dir, dataset_name, npy_file_tr, torso, SUPPORT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS,
          CLASS_NUM, adj, degree, weight_decay, lr, num_class, model_save_path, best_acc, kk, gamma, cuda, num_joints, alpha):
    sq_dataloader = get_dataloader(SUPPORT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS, num_class=num_class, numclass=CLASS_NUM,
                                   trte_list=tr_list, data_dir=data_dir, dataset_name=dataset_name,
                                   npy_file=npy_file_tr, torso=torso, njoints=num_joints)

    device = torch.device("cuda" if cuda else "cpu")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    optimizer.zero_grad()

    for idx, sample in enumerate(sq_dataloader):
        train_data_s = sample[0][0:SUPPORT_NUM_PER_CLASS * CLASS_NUM]
        train_data_q = sample[0][SUPPORT_NUM_PER_CLASS * CLASS_NUM:]
        tr_label_s = sample[1][0:SUPPORT_NUM_PER_CLASS * CLASS_NUM]
        tr_label_q = sample[1][SUPPORT_NUM_PER_CLASS * CLASS_NUM:]
        samp_size_s = tr_label_s.shape[0]
        samp_size_q = tr_label_q.shape[0]

    batch_output_s, _ = model(train_data_s, samp_size_s, adj, degree, num_class, alpha, device)
    batch_output_q, _ = model(train_data_q, samp_size_q, adj, degree, num_class, alpha, device)


    if cuda:
        criterion = SoftDTW(use_cuda=True, gamma=gamma, normalize=True, bandwidth=0)
    else:
        criterion = SoftDTW(use_cuda=False, gamma=gamma, normalize=True, bandwidth=0)

    L = 0
    sim_count = 0
    not_sim_count = 0

    sim_loss = 0.0
    not_sim_loss = 0.0

    sim_loss_list = []
    not_sim_loss_list = []

    K1 = kk * CLASS_NUM
    K2 = kk

    for batch_idx in range(samp_size_s):
        label_s = tr_label_s[batch_idx].item()
        # print(one_hot_s_samp.shape, ' xxxxxx')

        for batch_jdx in range(samp_size_q):
            label_q = tr_label_q[batch_jdx].item()
            # [block, num_class]

            # block_pred_s = torch.unsqueeze(F.normalize(batch_output_s[batch_idx], p=2, dim=2), 0).to(device)
            # block_pred_q = torch.unsqueeze(F.normalize(batch_output_q[batch_jdx], p=2, dim=2), 0).to(device)
            # block_pred_s = torch.unsqueeze(batch_output_s[batch_idx], 0).to(device)
            # block_pred_q = torch.unsqueeze(batch_output_q[batch_jdx], 0).to(device)
            block_pred_s = torch.unsqueeze(znormalize(batch_output_s[batch_idx]), 0).to(device)
            block_pred_q = torch.unsqueeze(znormalize(batch_output_q[batch_jdx]), 0).to(device)
            # print(block_pred_s.shape, block_pred_q.shape)
            # print(torch.sum(torch.isnan(block_pred_s)).item(), ' ---')
            # print(torch.sum(torch.isnan(block_pred_q)).item(), ' ===')
            n = block_pred_s.shape[1]
            p = block_pred_q.shape[1]

            loss_sq = criterion(block_pred_s, block_pred_q) / (n * p)

            loss = loss_sq


            if label_s == label_q:
                # suppose only 1-shot!!!!!!!!!!
                sim_count = sim_count + 1
                sim_loss = sim_loss + loss
                sim_loss_list.append(loss.item())


            else:
                not_sim_count = not_sim_count + 1
                not_sim_loss = not_sim_loss + loss
                not_sim_loss_list.append(loss.item())

    not_sim_loss_mean = not_sim_loss / not_sim_count
    sim_loss_mean = sim_loss / sim_count

    a = np.array(not_sim_loss_list)

    xxx = a[np.argpartition(a, -K1)[-K1:]]

    b = np.array(sim_loss_list)

    yyy = b[np.argpartition(b, K2)[:K2]]

    not_sim_max = np.mean(xxx)

    sim_min = np.mean(yyy)

    # print(not_sim_max, sim_min)

    # L = torch.pow((F.relu(sim_loss_mean - sim_min) + F.relu(not_sim_max - not_sim_loss_mean)), 2)

    L = F.relu(sim_loss_mean - sim_min) + F.relu(not_sim_max - not_sim_loss_mean)

    L.backward()
    optimizer.step()

    return L, model
