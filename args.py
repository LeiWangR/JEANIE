import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--withcuda', action='store_false', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='MSRAction3D', help='MSRAction3D or 3DActionPairs')
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--Nway', type=int, default=10, help='N-way')
    parser.add_argument('--Kdic', type=int, default=4096, help='dictionary size')
    parser.add_argument('--topkk', type=int, default=3, help='top k min or max')
    parser.add_argument('--gamma', type=float, default=0.01, help='gamma for the soft-DTW')
    parser.add_argument('--Kshot', type=int, default=1, help='K-shot, support number of samples per class')
    parser.add_argument('--Qnum', type=int, default=1, help='query number of samples per class')
    parser.add_argument('--test_episode_size', type=int, default=1, help='test episode size')
    parser.add_argument('--finetune_episode', type=int, default=5, help='finetune episode size')
    parser.add_argument('--episode', type=int, default=5000, help='Number of episode to train.')
    parser.add_argument('--test_episode', type=int, default=1, help='Number of episode to test.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--degree', type=int, default=6, help='degree of the approximation.')
    parser.add_argument('--alpha', type=float, default=0.7, help='alpha for ssgc')

    parser.add_argument('--params', type=str, default='Kdic',
                        choices=['Kdic', 'topkk', 'gamma', 'degree', 'alpha', 'lr', 'weight_decay'],
                        help='parameters for tuning in tuning.py')
    # parser.add_argument('--tuned', action='store_true', help='use tuned hyperparams')

    args, _ = parser.parse_known_args()
    args.cuda = args.withcuda and torch.cuda.is_available()
    return args
