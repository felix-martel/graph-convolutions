import torch
from scipy import sparse
import numpy as np


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