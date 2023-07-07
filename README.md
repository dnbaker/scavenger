

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

The diagonal covariance is simpler in latent space and may pull our more independent information. But the full diagonal model has higher capacity and yields a much higher likelihood on real data.

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
For batch correction, use `categorical_class_sizes=[num_batches]` when constructed NBVAE, and add a one-hot encoded label.
If you have additional categorical labels (spatial data, atac-seq, microarray/short/long read, library prep), add them to the list, too.

For instance `categorical_class_sizes=[num_batches, num_experiment_types, num_library_preps]`.

Then, when calling forward on the model,

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

merged_labels = torch.vstack([torch.zeros(x, dtype=torch.long).reshape(-1, 1) + xi for xi, x in enumerate(shapes)]).to(merged_dataset.dtype)

# Now you can train:
# Get batches of data + labels

idx = sampled_set()
data = merged_data[idx]
labels = merged_labels[idx]

# labels can be one-hot or logits
# One-hot items (which are torch.long dtype) are treated as logits * 20, so 20,000 more likely to be the provided class.
# You can raise or lower this ratio with `temp=` for the forward call.
# Logits are used directly otherwise.
packed_output = model(data, labels)

labels = label_logits
# Get a tuple out
latent, losses, zinb, class_info = packed_output
# Or a dictionary, which is easier to reason with.
labeled_output = model.labeled_unpack(packed_output)
latent_repr, sampled_repr, nb_model, logvar, full_cov = latent
model_loss, reconstruction_loss = losses[:2]
class_logits, class_loss = class_info
total_loss = model_loss.sum() + reconstruction_loss.sum() + class_loss.sum()
total_loss.backward()
```

By having the model learn the classes, it can try to distinguish batches/effect types.

If you don't provide the class labels, the model will still generate logits for categorical labels, but it will only use count data to estimate. This gives it a light semi-supervised approach.

I aim to test this using rnaseq expression atlases for normal background (e.g., GTEx) for bulk data but leveraged for single-cell analysis.
