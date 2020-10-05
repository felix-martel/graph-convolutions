import torch
from scipy import sparse
import numpy as np


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