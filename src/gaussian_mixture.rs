use crate::nnutil;
use tch::{
    nn, nn::BatchNormConfig, nn::Module, nn::ModuleT, nn::SequentialT, Device, IndexOp, Kind,
    Tensor,
};

struct LatentGM {
    n_components: i64,
    latent_dim: i64,
    mixture_pi: nn::SequentialT, // logits for mixture components (batch, n_components)
    // p(y) = cat(mixture_pi)
    q_z_covar: nn::SequentialT, // From (n_latent) to (nchoose2(n_components) * latent_dim)
    q_z_means: nn::Linear,
    /*
    q_z_covariances: Tensor, // (n_comp, latent_dim, latent_dim)
    q_z_means: Tensor,
    q_z_variances: Tensor,

    p_z_covariances: Tensor, // (n_comp, latent_dim, latent_dim)
    p_z_means: Tensor,
    p_z_variances: Tensor,
    z_mean: Tensor,
    */
    // qz_given_x_y
    // layers goes from
    // input dim (x {counts} + categorical)
    // -> hidden dim
    // + then add a linear layer
    // from hidden dim to [n mixture components * latent dimension
}
// needs to generate:
// 1. kl_divergence_z
// 2. kl_divergence_y
// 3. q_y_probabilities
// 4. q_y_logits
// 5. p_y_probabilities
/*
    self.lower_bound, self.reconstruction_error,
    self.kl_divergence_z, self.kl_divergence_y,
    self.kl_divergence_neurons, self.q_y_probabilities,
    self.p_y_probabilities,
    self.q_y_logits, self.z_mean
*/

fn nchoose2(x: i64) -> i64 {
    x * (x - 1) / 2
}

impl LatentGM {
    pub fn new(
        &self,
        vs: &tch::nn::VarStore,
        latent_dim: i64,
        n_components: i64,
        full_cov: Option<bool>,
    ) -> Self {
        let full_cov = full_cov.unwrap_or(false);
        let mut mixture_pi = nn::seq_t();
        mixture_pi = mixture_pi.add(nn::linear(
            vs.root() / "mixture_pi",
            latent_dim,
            n_components,
            Default::default(),
        ));
        mixture_pi = mixture_pi.add_fn(|xs| xs.softmax(-1i64, xs.kind()));
        let num_covar_inputs = if full_cov {
            nchoose2(latent_dim)
        } else {
            latent_dim // diagonal cov
        } * n_components;
        let mut q_z_covar = nn::seq_t();
        q_z_covar = q_z_covar.add(nn::linear(
            vs.root() / "q_z_covar",
            latent_dim,
            num_covar_inputs,
            Default::default(),
        ));
        q_z_covar = q_z_covar.add_fn(|xs| xs.softplus());
        let q_z_means = nn::linear(
            vs.root() / "q_z_means",
            latent_dim,
            latent_dim,
            Default::default(),
        );
        Self {
            mixture_pi,
            latent_dim,
            n_components,
            q_z_covar,
            q_z_means,
        }
    }
}

// Need to do:
// 1. Add mean calculation to the model (use pytorch as base)
// 2. Do the sampling:
/*
                z_samples = q_z_given_x_y.sample(
                    self.n_iw_samples * self.n_mc_samples)
                z = tf.cast(
                    tf.reshape(
                        z_samples,
                        shape=[-1, self.latent_size],
                        name="SAMPLES"
                    ),
                    dtype=tf.float32
                )
*/
// 3.

// Dense outputs from a full-covariance gmm
// is of size n_components choose 2
// and
