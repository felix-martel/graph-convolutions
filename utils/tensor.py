import torch
from scipy import sparse
import numpy as np


def sparse_eye(dim: int) -> torch.sparse.FloatTensor:
    """
    Make a sparse identity matrix of dimension `dim`.
    """
    return torch.sparse.FloatTensor(
        torch.arange(dim).repeat(2, 1),
        torch.ones(dim),
        torch.Size([dim, dim])
    )

def normalize(a):
    """
    Normalize an adjacency matrix, as described in Kipf et al. 2017
    :param a: a ndarray representing an adjacency matrix
    :return: a normalized ndarray, with the same shape as `a`
    """
    d = np.array(a.sum(1)).squeeze()
    d = np.divide(1, d, where=d!=0)
    d = sparse.diags(d, format="csr")
    return d * a

def csr_to_torch(csr: sparse.csr_matrix) -> torch.sparse.LongTensor:
    """
    Transform a `scipy.sparse.csr_matrix` to a sparse Torch tensor.

    Based on https://stackoverflow.com/questions/50665141/converting-a-scipy-coo-matrix-to-pytorch-sparse-tensor/50665264#50665264
    """
    coo = csr.tocoo()
    t = torch.sparse.LongTensor(
        torch.LongTensor(np.vstack((coo.row, coo.col))),
        torch.FloatTensor(coo.data),
        torch.Size(coo.shape)
    )
    return t

def make_onehot(n_samples, n_classes):
    y = torch.zeros(n_samples, n_classes)
    y[torch.arange(n_samples),torch.randint(0, n_classes, (n_samples,))] = 1.
    return y