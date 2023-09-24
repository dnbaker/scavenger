use crate::nnutil;
use tch::{
    nn, nn::BatchNormConfig, nn::Module, nn::ModuleT, nn::SequentialT, Device, IndexOp, Kind,
    Tensor,
};

struct LatentGM {
    n_components: i64,
    latent_dim: i64,
    prior_pi_logits: Tensor,
    mixture_pi: nn::SequentialT,
    // logits for mixture components (batch, n_components)
    // p(y) = cat(mixture_pi)
    // If Learned, then the parameters are determined by a linear layer from latent to this followed by a softmax.
    // If not, then
    q_z_covar: nn::SequentialT, // From (n_latent) to (nchoose2plusn(n_components) * latent_dim)
    q_z_means: nn::Linear,

    p_z_covar_given_y: nn::SequentialT,
    p_z_mean_given_y: nn::SequentialT, // using mean + covar, we then calculate mean of the distribution for priors later
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
fn nchoose2plusn(x: i64) -> i64 {
    nchoose2(x) + x
}

impl LatentGM {
    pub fn new(
        vs: &tch::nn::VarStore,
        latent_dim: i64,
        n_components: i64,
        full_cov: Option<bool>,
        kind: Option<tch::Kind>,
    ) -> Self {
        let full_cov = full_cov.unwrap_or(false);
        let kind = kind.unwrap_or(tch::Kind::Float);
        let mixture_pi = nn::seq_t()
            .add(nn::linear(
                vs.root() / "mixture_pi",
                latent_dim,
                n_components,
                Default::default(),
            ))
            .add_fn(|xs| xs.softmax(-1i64, xs.kind()));
        let prior_pi_logits = Tensor::randn(&[n_components], (kind, vs.device()));
        let num_covar_inputs = if full_cov {
            nchoose2plusn(latent_dim)
        } else {
            latent_dim // diagonal cov
        } * n_components;
        let num_input_features = latent_dim + n_components;
        let q_z_covar = nn::seq_t()
            .add(nn::linear(
                vs.root() / "q_z_covar",
                latent_dim,
                num_covar_inputs,
                Default::default(),
            ))
            .add_fn(|xs| xs.softplus());
        let p_z_covar_given_y = nn::seq_t()
            .add(nn::linear(
                vs.root() / "p_z_given_y",
                n_components,
                num_covar_inputs,
                Default::default(),
            ))
            .add_fn(|xs| xs.softplus());
        let p_z_mean_given_y = nn::seq_t().add(nn::linear(
            vs.root() / "p_z_given_y",
            n_components,
            latent_dim,
            Default::default(),
        ));
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
            p_z_mean_given_y,
            p_z_covar_given_y,
            q_z_covar,
            q_z_means,
            prior_pi_logits,
        }
    }
    // We don't need a kl categorical loss because we are using the model to select the mixture components.
}

mod tests {
    use crate::gaussian::*;
    #[test]
    fn test_stuff() {
        /*
        pub fn new(
            &self,
            vs: &tch::nn::VarStore,
            latent_dim: i64,
            n_components: i64,
            full_cov: Option<bool>,
            kind: Option<tch::Kind>,
        ) -> Self {
            let full_cov = full_cov.unwrap_or(false);

            */
        let vs = tch::nn::VarStore::new(Device::Cpu);
        let latent_mix = LatentGM::new(&vs, 128i64, 4i64, Some(true), Some(Kind::Float));
        /*
        vs: &tch::nn::VarStore,
        latent_dim: i64,
        n_components: i64,
        full_cov: Option<bool>,
        kind: Option<tch::Kind>,
        */
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
