try:
    import equiformer_pytorch
except ImportError:
    print("equiformer_pytorch not installed. Will not provide Equiformer model.")

import torch

class SpatialNDTransformer(torch.nn.Module):
    def __init__(self, coord_dim_lens=None, dim_head=None, num_tokens=None, num_positions=None, edge_dim=None, heads=None, **kwargs):
        super().__init__()
        def broadcast_default(x, default_value):
            if x is None:
                x = default_value
            if isinstance(x, int):
                x = (x,) * kwargs['num_degrees']
            return x
        coord_dim_lens = broadcast_default(coord_dim_lens, 1)
        dim_head = broadcast_default(dim_head, 16)
        heads = broadcast_default(heads, 2)
        if 'attend_self' not in kwargs:
            kwargs['attend_self'] = True
        if 'attend_self' not in kwargs:
            kwargs['attend_self'] = coord_dim_lens
        if 'num_degrees' not in kwargs:
            kwargs['num_degrees'] = len(coord_dim_lens)
        self.model = equiformer_pytorch.Equiformer(edge_dim=edge_dim, dim_head=dim_head, num_tokens=num_tokens, heads=heads, **kwargs)


    @staticmethod
    def make_nd_spatial(ndim=2, nfeatures=32768, hidden=None, **kwargs):
        if 'num_degrees' not in kwargs:
            kwargs['num_degrees'] = 1
        if not hidden:
            hidden = nfeatures
        model = SpatialNDTransformer(nfeatures, dim_in=nfeatures, **kwargs)

    def forward(self, feats, coords, mask=None):
        return self.model(feats, coords, mask=mask)

if __name__ == "__main__":
    tx = SpatialNDTransformer(dim=(2,), dim_in=(2,), num_degrees=1)
    coords = torch.Tensor([[0., 1.], [1., 0.], [4., -1.3]])
    data = torch.Tensor([[0., 1.], [1., 0.], [4., -1.3]])
    print("This won't work at the level of individual data: this model expects N dimensions for each feature. This is great for processing a whole slide of spatial tx as one entity.")
    print("But not great for denoising. So this model should be fed the output of an autoencoder, for instance.")
    output = tx(coords, data)
    print(output)
