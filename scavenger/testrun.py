#!/usr/bin/env python3

import numpy as np
import simple_nb_vae
import torch
import scipy.sparse as sp


from argparse import ArgumentParser as AP

ap = AP()
ap.add_argument("--hidden", default="256,64,32")
ap.add_argument("--latent-dim", type=int, default=16)
ap.add_argument("--full-cov", action='store_true')
ap.add_argument("--zero-inflate", action='store_true')

args = ap.parse_args()
hidden = list(map(int, args.hidden.split(',')))
# print(hidden)

torch.manual_seed(3)
data = np.load("/Users/dnb13/Desktop/code/compressed_bundle/PBMC/pbmc.csr.npz")
mat = sp.csr_matrix((data['data'], data['indices'],
                    data['indptr']), shape=data['shape']).astype(np.float32)
f16 = torch.from_numpy(mat[:37, :].todense())

indices = np.arange(mat.shape[0])

randvals = torch.rand([mat.shape[0]])
train_vals = torch.where(randvals > .2)
test_vals = torch.where(torch.logical_and(randvals > .1, rand_vals <= 0.2))
validation_vals = torch.where(randvals < .1)

latent_dim = args.latent_dim
hidden_dims = hidden

model = simple_nb_vae.NBVAE(data_dim=f16.shape[1], latent_dim=latent_dim,
                            hidden_dim=hidden_dims, full_cov=args.full_cov, zero_inflate=args.zero_inflate)
out = model(f16)
# print("out:", out.shape)
unpacked_out = model.unpack(out)

latent, losses, zinb = unpacked_out

labeled_unpack = model.labeled_unpack(unpacked_out)

model.eval()

module = torch.jit.trace(model, f16, check_trace=False)
traced_module = module
orig_module = model

torch.manual_seed(0)
out_jit = module(f16)
torch.manual_seed(0)
out_orig = orig_module(f16)
assert torch.allclose(out_orig, out_jit)
hidden_dim = ",".join(map(str, hidden_dims))
module.save(f"nbvae.{latent_dim}.{hidden_dim}.pt")


# Now try to compile
if settings.compile:
    module = torch.compile(model)
    torch.manual_seed(0)
    out_compile = module(f16)
    assert torch.allclose(out_orig, out_compile)
