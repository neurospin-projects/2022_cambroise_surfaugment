import numpy as np
import torch
from torch import nn
import torch.nn.functional as func
from sklearn.metrics.pairwise import rbf_kernel

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwins(nn.Module):
    def __init__(self, args, backbone):
        super().__init__()
        self.lambd = args.loss_param
        self.backbone = backbone

        # projector
        sizes = [args.latent_dim] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(z1.shape[0])

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


class yAwareSimCLR(nn.Module):
    """ Class implementing the y-aware simCLR model. Code originally insipired
    from https://github.com/Duplums/yAwareContrastiveLearning/
    """
    def __init__(self, args, backbone, return_logits=False):
        """
        :param kernel: a callable function f: [K, *] x [K, *] -> [K, K]
                                              y1, y2          -> f(y1, y2)
                        where (*) is the dimension of the labels (yi)
        default: an rbf kernel parametrized by 'sigma' which corresponds to gamma=1/(2*sigma**2)
        :param temperature:
        :param return_logits:
        """
        super().__init__()
        self.backbone = backbone
        # projector
        sizes = [args.latent_dim] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # sigma = prior over the label's range
        self.kernel = "rbf"
        self.temperature = args.loss_param
        self.sigma = args.sigma
        if self.kernel == 'rbf' and self.sigma > 0:
            self.kernel = lambda y1, y2: rbf_kernel(
                y1, y2, gamma=1./(2 * self.sigma**2))
        # else:
            # assert hasattr(self.kernel, '__call__'), 'kernel must be a callable'
        self.return_logits = return_logits
        self.INF = 1e8

    def forward(self, y1, y2, labels=None):
        
        z_i = self.projector(self.backbone(y1))
        z_j = self.projector(self.backbone(y2))
        N = len(z_i)
        if labels is not None:
            assert N == len(labels), "Unexpected labels length: %i"%len(labels)
        else:
            assert self.sigma == 0, "Labels must be provided when sigma is > 0"
        z_i = func.normalize(z_i, p=2, dim=-1)
        z_j = func.normalize(z_j, p=2, dim=-1)
        sim_zii= (z_i @ z_i.T) / self.temperature
        sim_zjj = (z_j @ z_j.T) / self.temperature
        sim_zij = (z_i @ z_j.T) / self.temperature
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        if self.sigma > 0:
            all_labels = labels.view(N, -1).repeat(2, 1).detach().cpu().numpy()
            weights = self.kernel(all_labels, all_labels)
        else:
            weights = np.ones((2 * N, 2 * N))
        weights = weights * (1 - np.eye(2 * N)) # puts 0 on the diagonal
        weights /= weights.sum(axis=1)
        sim_Z = torch.cat([torch.cat([sim_zii, sim_zij], dim=1),
                           torch.cat([sim_zij.T, sim_zjj], dim=1)], dim=0)
        log_sim_Z = func.log_softmax(sim_Z, dim=1)

        loss = - (
            (torch.from_numpy(weights).to(z_i.device) * log_sim_Z).sum() / N)

        correct_pairs = torch.arange(N, device=z_i.device).long()

        if self.return_logits:
            return loss, sim_zij, correct_pairs

        return loss
