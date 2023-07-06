

##Contents

1. VAE with negative binomial likelihood model with diagonal or full covariance latent model.
2. Batch correction:
  1. One-hot encoded class data helps correct for batch effects in reconstruction
  2. Allowance of softmax/uncertain classification allows semi-supervised learning.
  3. Optional classifier loss helps the model distinguish as well, giving us two locations for batch correction in the model.
3. Use of equivariant GNN for spatially-resolved patches. The aim is to use VAE-based embeddings in the spatial model.
  1. Code in `scavenger/experimental/equiformer.py`.


## Quick-start

For count-based data, see `train.py` for an example. 

`scavenger.NBVAE` is the class you'll want to work with. Use `full_cov={True, False}` to choose between the isotropic variance and full covariance matrix options.

The isotropic case (diagonal covariance) is simpler in latent space and may pull our more independent information. But the full diagonal model has higher capacity.

```
from scavenger import NBVAE

data_dim = 32768 # Your number
latent_dim = 128
hidden_dim = [512, 256]
model = NBVAE(data_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, full_cov=True)


# Let data = batch (N, Dim)
packed_output = model(data)
latent, losses, zinb, class_info = packed_output
# zinb is the model for the data provided, latent is the latent representations
# latent_repr = (N, LatentDim)
# sampled_repr = (N, LatentDim) - latent + Gaussian noise
# nb_model: scavenger.ZINB model.
# logvar: log variance
# This is always used.

# full_cov: (N, LatentDim * LatentDim) - expanded covariance matrix.
# This is None if full covariance is not enabled.

# The diagonal of this matrix is the exponent of logvar
latent_repr, sampled_repr, nb_model, logvar, full_cov = latent

# Model loss: kl divergence of reparameterization
# Reconstruction loss: negative log-likelihood of model reconstruction.
model_loss, reconstruction_loss = losses[:2]


# If classification labels were provided, class_info will have (`class_logits`, `class_loss`).
# Otherwise, it will be None.
# You can use it to see how clearly the sample belonged to a particular group.
# And you can backpropagate from `class_loss` to teach the model to reconstruct categorical labels as well.

```


### Batch correction/integration
For batch correction, use `categorical_class_sizes=` when constructed NBVAE.

For example:

```python3
data_dim = 32768 # Your number
latent_dim = 128
hidden_dim = [512, 256]
model = NBVAE(data_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, full_cov=True)

dataset1, dataset2 = two_different_datasets()
shapes = [x.shape[0] for x in (dataset1, dataset2)]

# Use union or intersection for genes to get the same feature-set if necessary.
# Assume both datasets have the same features and are in row-major format.

merged_dataset = torch.vstack([dataset1, dataset2])

merged_labels = torch.vstack([torch.zeros(x, dtype=torch.float32).reshape(-1, 1) + xi for xi, x in enumerate(shapes)]).to(merged_dataset.dtype)

labeled_dataset = torch.hstack([merged_dataset, merged_labels])

# Now you can train:
packed_output = model(data)
labels = label_logits
latent, losses, zinb, class_info = model(data, labels=label_logits)
# labels can be one-hot or logits

latent_repr, sampled_repr, nb_model, logvar, full_cov = latent
model_loss, reconstruction_loss = losses[:2]
class_logits, class_loss = class_info
class_loss.sum(axis=1).backward()
```

By having the model learn the classes, it can try to distinguish batches/effect types.

I aim to test this using rnaseq expression atlases for normal background.

### Ideas to try

Other ideas to try

0. Diffusion model for genetic pretraining + spatial joint modeling (like destvi)

1. Scaling loss by variance of genes (or a variation of it)

2. Pretrain individually, then fine-tune using local data pooled with graph nn

3. Predict what it should look like to match data across types.


