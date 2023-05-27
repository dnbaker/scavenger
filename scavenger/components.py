import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def nchoose2(x):
    return (x * (x - 1)) // 2


def default_size(x, in_dim, out_dim):
    if x > 0:
        return x
    return int(np.ceil(in_dim / np.sqrt(in_dim / out_dim)))


def in_outs(data_dim, out_dim, hidden_dims=[-1, -1]):
    hidden_empty = not hidden_dims

    def def_size(x):
        return default_size(x, data_dim, out_dim)
    first_out = def_size(hidden_dims[0] if not hidden_empty else out_dim)
    size_pairs = [(data_dim, first_out)]
    for hidden_index in range(len(hidden_dims) - 1):
        lhsize, rhsize = map(
            def_size, hidden_dims[hidden_index:hidden_index + 2])
        size_pairs.append((lhsize, rhsize))
    size_pairs.append(tuple(
        map(def_size, (hidden_dims[-1] if hidden_dims else data_dim, out_dim))))
    print(
        f"sizepairs: {size_pairs}. in: {data_dim}, out: {out_dim} and hiddens {hidden_dims}", file=sys.stderr)
    return size_pairs


class ULayerSet(nn.Module):
    def __init__(self, data_dim, out_dim, hidden_dims=[-1, -1],
                 batch_norm=False, layer_norm=True, dropout=0.1,
                 momentum=0.01, eps=1e-3, activation=nn.Mish,
                 skip_last_activation=False):
        super().__init__()
        # print(data_dim, out_dim, "data, out")

        self.data_dim = data_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.n_layers = len(hidden_dims) + 1
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.dropout = dropout
        layerlist = []
        hidden_empty = len(self.hidden_dims) == 0
        first_out = default_size(
            self.hidden_dims[0] if not hidden_empty else out_dim, data_dim, out_dim)
        size_pairs = in_outs(data_dim, out_dim, hidden_dims)

        def get_dropout():
            return nn.Dropout(dropout, inplace=True) if dropout > 0. else None
        # print("sizes: ", size_pairs, "for hidden ", hidden_dims, "and input ", data_dim, " and out ", out_dim)
        for index, (lhsize, rhsize) in enumerate(size_pairs):
            # print("in, out: ", lhsize, rhsize, file=sys.stderr)
            layerlist.append(nn.Linear(lhsize, rhsize))
            if batch_norm:
                layerlist.append(nn.BatchNorm1d(
                    rhsize, momentum=momentum, eps=eps))
            if layer_norm:
                layerlist.append(nn.LayerNorm(rhsize))
            if skip_last_activation and index == len(size_pairs) - 1:
                continue
            layerlist.append(activation())
            layerlist.append(get_dropout())
        self.layers = nn.Sequential(*list(filter(lambda x: x, layerlist)))

    def forward(self, x):
        return self.layers.forward(x)


class LayerSet(nn.Module):
    def __init__(self, data_dim, out_dim, hidden_dim=128, n_layers=3, batch_norm=False, layer_norm=True, dropout=0.1):
        super().__init__()
        self.data_dim = data_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.dropout = dropout

        def get_in(idx):
            return hidden_dim if idx > 0 else data_dim

        def get_out(idx):
            return hidden_dim if idx < n_layers - 1 else out_dim
        layerlist = []
        for idx in range(n_layers):
            ind = get_in(idx)
            outd = get_out(idx)
            layerlist.append(nn.Linear(ind, outd))
            # print(f"Layer {idx} from {ind} to {outd}")
            if batch_norm:
                layerlist.append(nn.BatchNorm1d(outd, momentum=0.01, eps=1e-3))
            if layer_norm:
                layerlist.append(nn.LayerNorm(outd))
        layerlist.append(nn.Mish())
        if dropout > 0.:
            layerlist.append(nn.Dropout(dropout, inplace=True))
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x):
        return self.layers.forward(x)
