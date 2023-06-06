

Contents:

1. VAE with negative binomial likelihood model with diagonal or full covariance latent model.
2. Batch correction:
  1. One-hot encoded class data helps correct for batch effects in reconstruction
  2. Allowance of softmax/uncertain classification allows semi-supervised learning.
  3. Optional classifier loss helps the model distinguish as well, giving us two locations for batch correction in the model.
3. Use of equivariant GNN for spatially-resolved patches. The aim is to use VAE-based embeddings in the spatial model.




Other ideas to try

0. Diffusion model for genetic pretraining + spatial joint modeling (like destvi)

1. Scaling loss by variance of genes (or a variation of it)

2. Pretrain individually, then fine-tune using local data pooled with graph nn

3. Predict what it should look like to match data across types.
