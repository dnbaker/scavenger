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
    from equiformer_pytorch import Equiformer

    model = Equiformer(
    num_tokens = 24,
    dim = (4, 4, 2),               # dimensions per type, ascending, length must match number of degrees (num_degrees)
    dim_head = (4, 4, 4),          # dimension per attention head
    heads = (2, 2, 2),             # number of attention heads
    num_linear_attn_heads = 0,     # number of global linear attention heads, can see all the neighbors
    num_degrees = 3,               # number of degrees
    depth = 4,                     # depth of equivariant transformer
    attend_self = True,            # attending to self or not
    reduce_dim_out = True,         # whether to reduce out to dimension of 1, say for predicting new coordinates for type 1 features
    l2_dist_attention = False      # set to False to try out MLP attention
    )

    feats = torch.randint(0, 24, (1, 128))
    coors = torch.randn(1, 128, 3)
    mask  = torch.ones(1, 128).bool()

    out = model(feats, coors, mask) # (1, 128)

    example = equiformer_pytorch.Equiformer(dim=(2,), dim_head=(8,), heads=(2,), num_degrees=1, l2_dist_attention=True, reduce_dim_out=False)
    feats = torch.randn((1, 128))
    coors = torch.randn(1, 128, 2)
    mask  = torch.ones(1, 128).bool()
    out = example(feats, coors, mask)


    tx = SpatialNDTransformer(dim=(2,), dim_in=(2,), num_degrees=1)
    coords = torch.randn((4, 2, 2))
    data = torch.randn((4, 2)).square()
    print(coords.shape, "coords", data.shape, "data")
    print("This won't work at the level of individual data: this model expects N dimensions for each feature. This is great for processing a whole slide of spatial tx as one entity.")
    print("But not great for denoising. So this model should be fed the output of an autoencoder, for instance.")
    output = tx(data, coords)
    print(output)
