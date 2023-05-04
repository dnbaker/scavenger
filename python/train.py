#!/usr/bin/env python3
import sys
import os
import glob
import time

import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp

import simple_nb_vae

from argparse import ArgumentParser as AP

ap = AP()
ap.add_argument("--hidden", default="256,64,32")
ap.add_argument("--latent-dim", type=int, default=16)
ap.add_argument("--no-full-cov", action='store_true')
ap.add_argument("--zero-inflate", action='store_true')
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--batch-size", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--subsample-rows", type=float, default=1.)
ap.add_argument("--add-log1p-l2-recon-loss", action='store_true')
ap.add_argument("--compile", action='store_true')
ap.add_argument("--gradientfreq", '-G', type=int, help="Number of batches between calling backward.", default=1)

args = ap.parse_args()
args.full_cov = not args.no_full_cov

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

covstr = "full_cov" if args.full_cov else "diag_cov"
zi = ".zi" if args.zero_inflate else ""

settings = args

hidden = list(map(int, args.hidden.split(',')))

start_time = time.time()
print(f"Starting with args {args} at time {start_time}", file=sys.stderr)
# print(hidden)

torch.manual_seed(args.seed)
base = "/Users/dnb13/Desktop/code/compressed_bundle/PBMC"
labels = np.memmap(base + "/labels.u8.npy", np.uint8)
mat = sp.csr_matrix((np.memmap(base + "/pbmc.data.u16.npy", np.uint16), np.memmap(base + "/pbmc.indices.u16.npy", np.uint16), np.memmap(base + "/pbmc.indptr.u32.npy", np.uint32)), shape=np.fromfile(base + "/pbmc.shape.u32.npy", np.uint32))
'''
data = np.load("/Users/dnb13/Desktop/code/compressed_bundle/PBMC/pbmc.csr.npz")
mat = sp.csr_matrix((data['data'], data['indices'],
                    data['indptr']), shape=data['shape'])
'''

if args.subsample_rows < 1.:
    mat = mat[torch.rand(mat.shape[0]) < args.subsample_rows]
print(f"Mat of shape {mat.shape}", file=sys.stderr)

indices = np.arange(mat.shape[0])

rand_vals = torch.rand([mat.shape[0]])
train_vals = torch.where(rand_vals > .2)
test_vals = torch.where(torch.logical_and(rand_vals > .1, rand_vals <= 0.2))
validation_vals = torch.where(rand_vals < .1)

latent_dim = args.latent_dim
hidden_dims = hidden

model = simple_nb_vae.NBVAE(data_dim=mat.shape[1], latent_dim=latent_dim,
                            hidden_dim=hidden_dims, full_cov=args.full_cov, zero_inflate=args.zero_inflate)
print(model)
'''
out = model(f16)
# print("out:", out.shape)
unpacked_out = model.unpack(out)

latent, losses, zinb = unpacked_out

labeled_unpack = model.labeled_unpack(unpacked_out)
'''

num_train = len(train_vals[0])
num_batches = (num_train + args.batch_size - 1) // args.batch_size

model.eval()


opt = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch_id in range(args.epochs):
    randperm = torch.randperm(num_train)
    model.train()
    backprop_count = 0
    opt.zero_grad()
    for batch_id in range(num_batches):
        start = batch_id * args.batch_size
        end = start + args.batch_size
        idxtouse = randperm[start:end]
        submatrix = torch.from_numpy(mat[idxtouse].todense().astype(np.float32))
        unpacked_out = model.unpack(model(submatrix))
        latent, losses, zinb = unpacked_out
        model_loss, recon_loss = losses
        loss = model_loss.sum(axis=1) + recon_loss.sum(axis=1)
        if args.add_log1p_l2_recon_loss:
            sampled = zinb.sample()
            lsampled = torch.log1p(sampled)
            source = torch.log1p(submatrix)
            loss += (lsampled - source).square().sum(axis=1)
        backprop_count += 1
        if backprop_count == args.gradientfreq:
            loss.sum().backward()
            opt.step()
            backprop_count = 0
            opt.zero_grad()
            print(f"{batch_id}/{num_batches} at {epoch_id} has mean loss {loss.mean().item()}", file=sys.stderr)
    if backprop_count > 0:
        loss.sum().backward()
        opt.step()
        backprop_count = 0
        opt.zero_grad()
        print(f"{batch_id}/{num_batches} at {epoch_id} has mean loss {loss.mean().item()}", file=sys.stderr)
    # Now check test acc
    model.eval()
    num_test_batches = (len(test_vals[0]) +
                        args.batch_size - 1) // args.batch_size
    model_test_loss = None
    recon_test_loss = None
    for batch_id in range(num_test_batches):
        start = batch_id * args.batch_size
        end = start + args.batch_size
        submatrix = torch.from_numpy(mat[start:end, :].todense().astype(np.float32))
        latent, losses, zinb = model.unpack(model(submatrix))
        model_loss, recon_loss = losses
        if model_test_loss is None:
            model_test_loss = model_loss.sum(-1)
            recon_test_loss = recon_loss.sum(-1)
            continue
        model_test_loss += model_loss.sum(-1)
        recon_test_loss += recon_loss.sum(-1)
    print(f"[After epoch {epoch_id + 1} - Mean test loss: {model_test_loss.mean().item()} for model fit, {recon_test_loss.mean().item()} for reconstruction.", file=sys.stderr)
    torch.save(
        model, f"nbvae.{latent_dim}.{hidden_dims}.{covstr}.{epoch_id}of{args.epochs}{zi}.pt")


model.eval()

f16 = torch.from_numpy(mat[:37, :].todense().astype(np.float32))
module = torch.jit.trace(model, f16, check_trace=False)
traced_module = module
orig_module = model
torch.manual_seed(0)
out_jit = module(f16)
torch.manual_seed(0)
out_orig = orig_module(f16)
assert torch.allclose(out_orig, out_jit)
hidden_dim = ",".join(map(str, hidden_dims))
covstr = "full_cov" if args.full_cov else "diag_cov"
zi = ".zi" if args.zero_inflate else ""
module.save(
    f"nbvae.{latent_dim}.{hidden_dim}.{covstr}.{args.epochs}.{zi}jit.pt")
torch.save(
    model, f"nbvae.{latent_dim}.{hidden_dim}.{covstr}.{args.epochs}{zi}.pt")

set(map(os.remove, glob.iglob(f"nbvae.{latent_dim}.{hidden_dim}.{covstr}.*of{args.epochs}{zi}.pt")))

print(
    f"Finished with args {args} at time {time.time()} (after {time.time() - start_time})", file=sys.stderr)

# Now try to compile
if args.compile:
    module = torch.compile(model)
    torch.manual_seed(0)
    out_compile = module(f16)
    assert torch.allclose(out_orig, out_compile)
