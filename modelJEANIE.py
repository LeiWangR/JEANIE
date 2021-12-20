import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def znormalize(x):
    mu_ = torch.mean(x, 2, keepdim=True)
    std_ = torch.std(x, 2, keepdim=True)

    # print(mu_.shape, '\n', mu_)
    y = (x - mu_) / (std_ + 1e-10)

    print(x.shape, y.shape)
    return y

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = True)#False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, njoints, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        num_patches = njoints
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)

        return self.mlp_head(x)


def sgc_precompute(features, adj, degree, alpha):
    feature_ori = features
    feature_set = torch.zeros_like(features)
    for i in range(degree):
        features = torch.spmm(adj, features)
        feature_set += (1 - alpha) * features + alpha * feature_ori
    feature_set /= degree
    return feature_set

class SGC(nn.Module):
    '''
    an MLP is used to compute the features of each joint
    within each video block
    '''

    def __init__(self, njoint, ndim, nframe, nclass, nfeat):
        super(SGC, self).__init__()
        # nfeat = 50 for smaller dataset and 100 for big dataset
        self.fc1 = nn.Linear(ndim * nframe, ndim * nframe * 2)
        self.ln1 = nn.LayerNorm(ndim * nframe * 2)
        self.fc2 = nn.Linear(ndim * nframe * 2, ndim * nframe * 3)
        self.ln2 = nn.LayerNorm(ndim * nframe * 3)
        self.fc3 = nn.Linear(ndim * nframe * 3, nfeat)
        self.ln3 = nn.LayerNorm(nfeat)
        self.vit = ViT(njoints=njoint, num_classes=nclass, dim=nfeat, depth=6, heads=6, dim_head = 64, mlp_dim=njoint * nfeat,dropout=0.5, emb_dropout=0.5)
        self.dropout = nn.Dropout(p=0.5)
        # self.dropout = nn.Dropout(p=0.1)
        self.W = nn.Linear(njoint * nfeat, nclass)
        self.cls = nn.Softmax(dim=1)

    def forward(self, x, batch_size, adj, degree, nclass, alpha, device):
        # alpha is for SSGC
        adj = adj.to(device)
        njoints = adj.shape[0]
        pred = torch.zeros((batch_size, nclass)).float().to(device)
        batch_block_output = []

        for sample_idx in range(batch_size):
            # [#blocks, #joints, 2d/3d, frame_per_block]
            blocks = x[sample_idx]
            # print(blocks.shape)
            num_block = blocks.shape[0]
            num_views1 = blocks.shape[1]
            num_views2 = blocks.shape[2]
            num_joint = blocks.shape[3]
            num_subject = int(num_joint / njoints)

            block_features = torch.zeros((num_block, num_views1, num_views2, num_subject, nclass)).float().to(device)
            # for each performing subject
            for sub_id in range(num_subject):
                for block_idx in range(num_block):
                    for view_idx in range(num_views1):
                        for view_jdx in range(num_views2):

                            video_block = blocks[block_idx, view_idx, view_jdx, sub_id * njoints: sub_id * njoints + njoints, :, :]
                            # [#joints, 2d/3d, frame_per_block]
                            joint_features = video_block.view(njoints, -1).to(device)
                            joint_features = F.relu(self.ln1(self.fc1(joint_features)))
                            joint_features = F.relu(self.ln2(self.fc2(joint_features)))
                            joint_features = self.dropout(joint_features)
                            joint_features = self.ln3((self.fc3(joint_features)))
                            # [#joints, 100]
                            sgc_joint_feat = sgc_precompute(joint_features, adj, degree, alpha)

                            # sgc_joint_vec = torch.unsqueeze(sgc_joint_feat, 0)
                            # sgc_joint_vec_reg = self.vit(sgc_joint_vec)

                            sgc_joint_vec = sgc_joint_feat.view(1, -1).to(device)
                            sgc_joint_vec_reg = self.W(sgc_joint_vec)

                            # each row is a prediction score for each video block
                            # [#block, #class]
                            # sgc_joint_vec_reg = self.cls(sgc_joint_vec_reg)
                            block_features[block_idx, view_idx, view_jdx, sub_id, :] = sgc_joint_vec_reg
            block_features_mean = torch.mean(block_features, 3)
            batch_block_output.append(block_features_mean)
            video_feat_aggr = torch.mean(torch.mean(torch.mean(block_features_mean, 0), 0), 0)
            pred[sample_idx, :] = video_feat_aggr
        return batch_block_output, pred

def get_model(njoint, ndim, nframe, nclass, nfeat, cuda=True):
    model = SGC(njoint=njoint, ndim=ndim, nframe=nframe, nclass=nclass, nfeat=nfeat)
    if cuda:
        model.cuda()
    return model
