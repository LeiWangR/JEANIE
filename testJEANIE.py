import torch
from jeanie import SoftDTW
import torch.nn.functional as F

from modelJEANIE import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dictlist(dict):
    def __setitem__(self, key, value):
        try:
            self[key]
        except KeyError:
            super(Dictlist, self).__setitem__(key, [])
        self[key].append(value)

def test(dataloader_exampler, dataloader_te, model, torso, adj, degree, batch, num_class, TEST_EPISODE,
         SUPPORT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS, CLASS_NUM, alpha, gamma, cuda):
    device = torch.device("cuda" if cuda else "cpu")
    with torch.no_grad():
        model.eval()

        if cuda:
            criterion = SoftDTW(use_cuda=True, gamma=gamma, normalize=True, bandwidth=0)
        else:
            criterion = SoftDTW(use_cuda=False, gamma=gamma, normalize=True, bandwidth=0)

        for i_batch_exampler, sample_batched_exampler in enumerate(dataloader_exampler):
            test_data_exampler = sample_batched_exampler[0]
            label_exampler = sample_batched_exampler[1]
            label_exampler_size = label_exampler.shape[0]

            if cuda:
                label_exampler.cuda()

        batch_output_exampler, _ = model(test_data_exampler, label_exampler_size, adj, degree, num_class, alpha, device)
        correct_num = 0
        total_num = 0
        
        d = Dictlist()
        for i_batch, sample_batched in enumerate(dataloader_te):
            test_data = sample_batched[0]
            te_label = sample_batched[1]

            if cuda:
                te_label.cuda()

            batch_size = te_label.shape[0]

            batch_output_q, _ = model(test_data, batch_size, adj, degree, num_class, alpha, device)

            total_num = total_num + batch_size
            for batch_idx in range(batch_size):
                # [block, num_class]
                label_q = te_label[batch_idx].item()

                # block_pred_q = torch.unsqueeze(znormalize(batch_output_q[batch_idx]), 0).to(device)
                block_pred_q = torch.unsqueeze(batch_output_q[batch_idx], 0).to(device)
                # block_pred_q = torch.unsqueeze(F.normalize(batch_output_q[batch_idx], p=2, dim=2), 0).to(device)
                
                d[str(label_q)] = block_pred_q

                M_loss = torch.zeros((label_exampler_size))

                for batch_jdx in range(label_exampler_size):
                    label_ex = label_exampler[batch_jdx].item()
                    # block_pred_exampler = torch.unsqueeze(znormalize(batch_output_exampler[batch_jdx]), 0).to(device)
                    block_pred_exampler = torch.unsqueeze(batch_output_exampler[batch_jdx], 0).to(device)
                    # block_pred_exampler = torch.unsqueeze(F.normalize(batch_output_exampler[batch_jdx], p=2, dim=2), 0).to(device)

                    n = block_pred_exampler.shape[1]
                    p = block_pred_q.shape[1]

                    loss = criterion(block_pred_exampler, block_pred_q) / (n * p)

                    M_loss[batch_jdx] = loss

                for batch_kdx in range(label_exampler_size):
                    label_ex = label_exampler[batch_kdx].item()
                    # print(label_s, label_q, M_loss, torch.argmin(M_loss), batch_jdx)
                    if label_q == label_ex and label_exampler[torch.argmin(M_loss)] == label_q:
                        correct_num = correct_num + 1
            torch.save(d, "key-value-features.pth")
            acc = correct_num / total_num
            print('test batch ID: {} | accuracy: {:.4f}'.format(batch_idx + 1, acc))

    return acc  # correct_num, total_num
