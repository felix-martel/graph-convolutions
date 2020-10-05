import torch.nn as nn
import torch


class RGCNLayer(nn.Module):
    """
    A graph convolution layer.

    Based on "Modeling Relational Data with Graph Convolutional Networks", Schlichtkrull et al 2017. For Wr,
    we use basis functions decomposition instead of block diagonal decomposition, no dropout for now, and
    featureless inputs (that is, entities are represented by one-hot vectors and have no extra features attached).
    """

    def __init__(self, T, B, dim_in, dim_out, init="random"):
        """
        T: adjacency tensor (n_relations * n_entities * n_entities)
        B: number of basis functions
        dim_in: dimension of input feature vectors
        dim_out: dimension of output feature vectors
        """
        super().__init__()
        Nr, (Ne, _) = len(T), T[0].shape
        self.V = self.init(B, dim_in, dim_out, how=init)
        self.A = self.init(Nr, B, how=init)
        self.T = T

    def init(self, *size, how="random", fill_value=1., requires_grad=True):
        if how == "random":
            data = torch.rand(*size)
        elif how == "constant":
            data = torch.full(size, fill_value)
        else:
            raise ValueError(f"Unsupported initialization method '{how}'")
        return nn.Parameter(data=data, requires_grad=requires_grad)

    def forward(self, H):
        # Input: N * d_in
        W = torch.einsum("rb,bio->rio", self.A, self.V) # -> "R * d_in * d_out"
        W.requires_grad_()
        if isinstance(H, torch.sparse.Tensor):
            HxW = []
            for w in W:
                h = torch.sparse.mm(H, w)
                h.requires_grad_()
                HxW.append(h)
            # H = [torch.sparse.mm(H, w) for w in W] #
        else:
            HxW = torch.matmul(H, W)
        H = []
        for a, hw in zip(self.T, HxW):
            h = torch.sparse.mm(a, hw)
            h.requires_grad_()
            H.append(h)
        # H = torch.stack([torch.sparse.mm(a, hw) for a, hw in zip(self.T, H)])
        H = torch.stack(H).sum(axis=0)
        return H

class RGCN(nn.Module):
    """
    A relational-graph convolutional network implementation.

    This is only the encoder part, the decoder is not implemented yet. It can be used
    in a (semi-)supervised setting for entity classification. Based on "Modeling
    Relational Data with Graph Convolutional Networks", Schlichtkrull et al 2017.
    """
    def __init__(self, T, n_classes, hidden_sizes=None, n_basis=10):
        """

        :param T: adjacency tensor of the corresponding knowledge graph, as outputed
        by `utils.graph.build_adjacency_tensor`
        :param n_classes: number of classes (for the entity classification task)
        :param hidden_sizes: (optional) a list of hidden sizes for each convolution layer
        :param n_basis: the number of basis functions (named `B` in Schlichtkrull's paper). Can
        be an int, in which case it is used for each layer, or a list of ints (one per layer).
        """
        super().__init__()
        Nr, (Ne, _) = len(T), T[0].shape
        self.n_relations = Nr
        self.n_entities = Ne
        self.n_classes = n_classes
        self.input_size = self.n_entities
        self.output_size = self.n_classes
        self.T = T

        self.convolutions = self.build_convolutions(hidden_sizes, n_basis)
        self.softmax = nn.Softmax(0)

    def build_convolutions(self, hidden_sizes, n_basis, init="random"):
        if hidden_sizes is None:
            hidden_sizes = [32]
        hidden_sizes = [self.input_size, *hidden_sizes]
        if not isinstance(n_basis, int):
            assert len(n_basis) == len(hidden_sizes), "You must provide a \
              number of basis functions (`n_basis`) for each layer"
        else:
            n_basis = [n_basis] * len(hidden_sizes)
        hidden_sizes.append(self.output_size)
        layers = []
        for input_size, hidden_size, B in zip(hidden_sizes,
                                              hidden_sizes[1:],
                                              n_basis
                                              ):
            conv = RGCNLayer(self.T, B, input_size, hidden_size, init=init)
            layers.append(conv)
        return nn.ModuleList(layers)

    def forward(self, x):
        for conv in self.convolutions:
            x = conv(x)
        x = self.softmax(x)
        return x