use tch::{
    nn, nn::BatchNormConfig, nn::Module, nn::ModuleT, nn::SequentialT, Device, Kind, Tensor,
};

pub const VAR_EPS: f64 = 1e-5;

pub struct Vae {
    fc1: nn::Linear,
    fc21: nn::Linear,
    fc22: nn::Linear,
    fc3: nn::Linear,
    fc4: nn::Linear,
}

impl Vae {
    pub fn new(vs: &nn::Path) -> Self {
        Vae {
            fc1: nn::linear(vs / "fc1", 784, 400, Default::default()),
            fc21: nn::linear(vs / "fc21", 400, 20, Default::default()),
            fc22: nn::linear(vs / "fc22", 400, 20, Default::default()),
            fc3: nn::linear(vs / "fc3", 20, 400, Default::default()),
            fc4: nn::linear(vs / "fc4", 400, 784, Default::default()),
        }
    }

    pub fn encode(&self, xs: &Tensor) -> (Tensor, Tensor) {
        let h1 = xs.apply(&self.fc1).relu();
        (self.fc21.forward(&h1), self.fc22.forward(&h1))
    }

    pub fn decode(&self, zs: &Tensor) -> Tensor {
        zs.apply(&self.fc3).relu().apply(&self.fc4).sigmoid()
    }

    pub fn forward(&self, xs: &Tensor) -> (Tensor, Tensor, Tensor) {
        let (mu, logvar) = self.encode(&xs.view([-1, 784]));
        let std = (&logvar * 0.5).exp();
        let eps = std.randn_like();
        (self.decode(&(&mu + eps * std)), mu, logvar)
    }
}

#[derive(Debug, Clone)]
pub struct BroadcastingParameter {
    pub data: Vec<f64>,
}

impl BroadcastingParameter {
    pub fn value(&self, index: i64) -> f64 {
        *(self
            .data
            .get(index as usize)
            .unwrap_or(self.data.first().unwrap()))
    }
    pub fn is_single_value(&self) -> bool {
        self.data.len() == 1usize
    }
    pub fn from_vec(vals: &Vec<f64>) -> Self {
        Self::from(vals)
    }
}

impl std::convert::From<f64> for BroadcastingParameter {
    fn from(x: f64) -> Self {
        Self { data: vec![x] }
    }
}
impl std::convert::From<&[f64]> for BroadcastingParameter {
    fn from(x: &[f64]) -> Self {
        Self { data: x.to_vec() }
    }
}
impl std::convert::From<&Vec<f64>> for BroadcastingParameter {
    fn from(x: &Vec<f64>) -> Self {
        Self::from(&x[..])
    }
}

/*
pub struct BroadcastingParameter {
    pub data: Tensor,
}
impl BroadcastingParameter {
    pub fn value(&self, index: i64) -> Result<f64, tch::TchError>  {
        let size = self.data.size1().unwrap();
        if index < size {
            self.data.f_double_value(&[index])
        } else {
            Err(TchError::Shape(format!("index out of range, got {size:?} with index {index:?}")))
        }
    }
}
*/

#[derive(Clone, Debug, Copy)]
pub enum Activation {
    LeakyRelu,
    Relu,
    Gelu,
    Selu,
    Mish,
    // NoActivation,
}

fn leaky_relu(xs: &Tensor) -> Tensor {
    xs.maximum(&(xs * 0.2))
}

fn relu(xs: &Tensor) -> Tensor {
    xs.relu()
}
fn selu(xs: &Tensor) -> Tensor {
    xs.selu()
}
fn gelu(xs: &Tensor) -> Tensor {
    xs.gelu("none")
}

#[derive(Clone, Debug, Copy)]
pub struct ActivationPlan {
    activation: Activation,
    is_squared: bool,
}
impl std::default::Default for ActivationPlan {
    fn default() -> Self {
        Self {
            activation: Activation::Mish,
            is_squared: false,
        }
    }
}

impl ActivationPlan {
    pub fn apply(&self, input: &tch::Tensor) -> tch::Tensor {
        let mut output = match &self.activation {
            // Activation::NoActivation => input,
            Activation::LeakyRelu => leaky_relu(input),
            Activation::Relu => relu(input),
            Activation::Selu => selu(input),
            Activation::Gelu => gelu(input),
            Activation::Mish => input.mish(),
        };
        if self.is_squared {
            output.square_()
        } else {
            output
        }
    }
    pub fn apply_inplace(&self, input: &mut tch::Tensor) -> Tensor {
        match &self.activation {
            Activation::LeakyRelu => input.leaky_relu_(),
            Activation::Relu => input.relu_(),
            Activation::Selu => input.selu_(),
            Activation::Gelu => input.gelu_("none"),
            Activation::Mish => input.mish_(),
        }
    }
}
#[derive(Debug, Clone)]
pub enum FCLayerType {
    Encode,
    Decode,
    Neither,
}

pub struct ZINBVAESettings {
    pub d_in: i64,
    pub d_latent: i64,
    pub encoder_settings: FCLayerSetSettings,
    pub decoder_settings: FCLayerSetSettings,
    pub use_size_factor_key: bool,
}

impl ZINBVAESettings {
    pub fn new(
        d_in: i64,
        d_hidden: i64,
        n_layers: i64,
        d_latent: i64,
        dropout: Option<BroadcastingParameter>,
        use_size_factor_key: Option<bool>,
    ) -> Self {
        let encoder_settings = FCLayerSetSettings::new_simple(
            d_in,
            d_latent,
            Some(d_hidden),
            Some(n_layers),
            dropout.clone(),
        );
        let mut decoder_settings = FCLayerSetSettings::new_simple(
            d_in,
            d_latent,
            Some(d_hidden),
            Some(n_layers),
            dropout.clone(),
        );
        decoder_settings.no_dropout();
        Self {
            d_in,
            d_latent,
            encoder_settings,
            decoder_settings,
            use_size_factor_key: use_size_factor_key.unwrap_or(false),
        }
    }
}

pub struct ZINBVAE {
    settings: ZINBVAESettings,
    encoder: FCLayerSet,
    meanvar_encoder: tch::nn::Linear,
    decoder: ZINBDecoder,
    /*
        let meanvar_encoder = nn::linear(
            vs.root(),
            encoder_settings.d_out,
            latent_dim * 2i64,
            Default::default(),
        );
    */
}

impl ZINBVAE {
    pub fn new(vs: &nn::VarStore, settings: ZINBVAESettings) -> Self {
        let encoder = FCLayerSet::new(vs, settings.encoder_settings.clone());
        let decoder = ZINBDecoder::create(
            vs,
            FCLayerSet::new(vs, settings.decoder_settings.clone()),
            settings.d_in,
            settings.use_size_factor_key,
        );
        let meanvar_encoder = nn::linear(
            vs.root() / "meanvar_encoder",
            settings.d_latent,
            settings.d_latent * 2i64,
            Default::default(),
        );
        Self {
            settings,
            encoder,
            decoder,
            meanvar_encoder,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FCLayerSetSettings {
    pub d_in: i64,
    pub d_out: i64,
    pub d_hidden: i64,
    pub n_layers: i64,
    // Either a singular value, or one per layer
    pub dropout: BroadcastingParameter,
    pub batch_norm: Vec<bool>,
    pub layer_norm: Vec<bool>,
    pub activation: Vec<ActivationPlan>,
    pub layer_type: FCLayerType,
}

impl FCLayerSetSettings {
    pub fn default_dropout(value: Option<f64>) -> BroadcastingParameter {
        BroadcastingParameter::from(value.unwrap_or(0.1))
    }
    pub fn no_dropout(&mut self) {
        self.dropout = BroadcastingParameter::from(0.);
    }
    pub fn new_simple(
        d_in: i64,
        d_out: i64,
        d_hidden: Option<i64>,
        n_layers: Option<i64>,
        dropout: Option<BroadcastingParameter>,
    ) -> Self {
        let n_layers: usize = n_layers.unwrap_or(1) as usize;
        eprintln!("n layers: {n_layers}");
        let dropout = dropout.unwrap_or(Self::default_dropout(Some(0.125)));
        let layer_type = if dropout.data.iter().all(|x| *x == 0.) {
            FCLayerType::Decode
        } else {
            FCLayerType::Encode
        };
        Self {
            d_in,
            d_out,
            d_hidden: d_hidden.unwrap_or(128),
            n_layers: n_layers as i64,
            dropout,
            batch_norm: vec![false; n_layers],
            layer_norm: vec![true; n_layers],
            activation: vec![ActivationPlan::default(); n_layers],
            layer_type,
        }
    }
    pub fn set_layer_type(&mut self, new_type: FCLayerType) {
        self.layer_type = new_type;
    }
    pub fn make_layer(&self, vs: &nn::VarStore, layer_index: i64) -> SequentialT {
        let mut ret = nn::seq_t();
        let in_dim = if layer_index == 0 {
            self.d_in
        } else {
            self.d_hidden
        };
        let out_dim = if layer_index == self.n_layers - 1 {
            self.d_out
        } else {
            self.d_hidden
        };
        eprintln!("Layer {layer_index} in layer has {in_dim} in and {out_dim} out");
        let layer_type_str: &'static str = match self.layer_type {
            FCLayerType::Encode => "encode",
            FCLayerType::Decode => "decode",
            FCLayerType::Neither => "unknown",
        };
        ret = ret.add(nn::linear(
            vs.root() / format!("fclayer{in_dim}:{out_dim}:{layer_index}:{layer_type_str}:linear"),
            in_dim,
            out_dim,
            Default::default(),
        ));
        if self.batch_norm[layer_index as usize] {
            let config = BatchNormConfig {
                momentum: 0.01,
                eps: 1e-3,
                ..Default::default()
            };
            ret = ret.add(nn::batch_norm1d(
                vs.root()
                    / format!("fclayer{in_dim}:{out_dim}:{layer_index}:{layer_type_str}:batchnorm"),
                out_dim,
                config,
            ));
        }
        if self.layer_norm[layer_index as usize] {
            let layer_norm = nn::layer_norm(
                vs.root()
                    / format!("fclayer{in_dim}:{out_dim}:{layer_index}:{layer_type_str}:layernorm"),
                vec![out_dim],
                Default::default(),
            );
            ret = ret.add(layer_norm);
        }
        // Dropout doesn't need to be built, but we'll have to include it in the pass.
        let activation = std::sync::Arc::new(self.activation[layer_index as usize]);
        ret = ret.add_fn(move |xs| activation.apply(xs));
        let dropout_param = self.dropout.value(layer_index);
        if dropout_param > 0. && dropout_param < 1. {
            ret = ret.add_fn_t(move |xs, train| xs.dropout(dropout_param, train));
        }
        ret
    }
}

pub struct FCLayerSet {
    pub settings: FCLayerSetSettings,
    pub layers: SequentialT,
}

impl FCLayerSet {
    pub fn new(vs: &nn::VarStore, settings: FCLayerSetSettings) -> Self {
        let mut layers = nn::seq_t();
        for layer_index in 0..settings.n_layers {
            log::debug!("making layer {layer_index}");
            layers = layers.add(settings.make_layer(vs, layer_index));
        }
        Self { settings, layers }
    }
}

pub fn best_device_available() -> Device {
    if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else if tch::utils::has_mps() {
        Device::Mps
    } else {
        Device::Cpu
    }
}

pub struct ZINBDecoder {
    // Important: make sure there's no dropout here.
    pub fclayers: FCLayerSet,
    pub px_r_scale_dropout_decoder: tch::nn::Linear, // This layer emits x and r, and dropout. We then chunk it to extract x, r and dropout separately.
    use_size_factor_key: bool,
}

// For parsing from CSR
/*
#if 0
#endif
        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)
*/

/*
    pub fn sparse_csr_tensor(
        crow_indices: &Tensor,
        col_indices: &Tensor,
        values: &Tensor,
        options: (Kind, Device),
    ) -> Tensor {
        Tensor::f_sparse_csr_tensor(crow_indices, col_indices, values, options).unwrap()
    }
    pub fn sparse_csr_tensor_crow_col_value_size(
        crow_indices: &Tensor,
        col_indices: &Tensor,
        values: &Tensor,
        size: &[i64],
        options: (Kind, Device),
    ) -> Tensor {
        Tensor::f_sparse_csr_tensor_crow_col_value_size(
            crow_indices,
            col_indices,
            values,
            size,
            options,
        )
        .unwrap()
    }
*/

pub struct ZINB {
    mu: Tensor,
    theta: Tensor,
    zi_logits: Tensor,
    scale: Tensor,
}

impl ZINB {
    pub fn new(inputs: (Tensor, Tensor, Tensor, Tensor)) -> Self {
        Self {
            mu: inputs.2,        // rate
            theta: inputs.1,     // r
            zi_logits: inputs.3, // dropout, aka pi
            scale: inputs.0,     // scale
        }
    }
    pub fn scale_v(&self) -> &Tensor {
        &self.scale
    }
    pub fn theta_v(&self) -> &Tensor {
        &self.theta
    }
    pub fn pi(&self) -> &Tensor {
        &self.zi_logits
    }
    pub fn zi_logits_v(&self) -> &Tensor {
        &self.zi_logits
    }
    pub fn mu_v(&self) -> &Tensor {
        &self.mu
    }
    pub fn log_prob(&self, x: &Tensor) -> tch::Tensor {
        self.log_likelihood(x)
    }
    pub fn log_likelihood(&self, x: &Tensor) -> tch::Tensor {
        // eprintln!("Started");
        let softplus_pi = (-self.pi()).softplus();

        let theta_eps = &self.theta + VAR_EPS;
        let mu_theta_eps = &theta_eps + &self.mu;

        let log_theta_mu_eps = mu_theta_eps.log();
        let log_theta_eps = theta_eps.log();

        let pi_theta_log = -self.pi() + &self.theta * (&log_theta_eps - &log_theta_mu_eps);
        let log_mu_eps = (&self.mu + VAR_EPS).log();
        //eprintln!("Made pi theta and log mu");

        let case_zero = (pi_theta_log).softplus() - &softplus_pi;
        let case_non_zero = -softplus_pi
            + pi_theta_log
            + x * (log_mu_eps - log_theta_mu_eps)
            + (x + &self.theta).lgamma()
            - &self.theta.lgamma()
            - (x + 1).lgamma();
        //eprintln!("made case_non_zero. Shape: {:?}", case_non_zero.size2());

        let xlt_eps = x.less(tch::Scalar::from(VAR_EPS));
        let xge_eps = xlt_eps.logical_not();

        //eprintln!("Made cases");
        let mul_case_zero = case_zero * xlt_eps.internal_cast_float(/*non_blocking=*/ false);
        let mul_case_nonzero = case_non_zero * xge_eps.internal_cast_float(false);
        //eprintln!("Made mulcases");
        mul_case_zero + mul_case_nonzero
    }
}

impl ZINBDecoder {
    pub fn hidden_dim(&self) -> i64 {
        return self.fclayers.settings.d_hidden;
    }
    pub fn decode(&self, input: &tch::Tensor, library_size: &tch::Tensor, train: bool) -> ZINB {
        let px = self.fclayers.layers.forward_t(&input, train);
        //eprintln!("px generated");
        /*eprintln!(
            "Shape for decoder forward: {:?}",
            self.px_r_scale_dropout_decoder.ws.size2()
        );
        */
        //eprintln!("Shape for px forward: {:?}", px.size2());
        let r_dropout = self.px_r_scale_dropout_decoder.forward_t(&px, train);
        //eprintln!("dropout generated");
        let mut r_dropout = r_dropout.chunk(3, -1);
        let px_scale = r_dropout.remove(2);
        let px_scale = if self.use_size_factor_key {
            px_scale.softmax(-1, Kind::Float)
        } else {
            px_scale.softplus()
        };
        //eprintln!("Got px_scale of size {:?}", px_scale.size2());
        let dropout = r_dropout.remove(1);
        //eprintln!("Got dropout ");
        let r = r_dropout.remove(0).exp();
        //eprintln!("Got r");
        let px_rate = library_size.exp() * &px_scale;
        //eprintln!("Got px_rate");
        return ZINB::new((px_scale, r, px_rate, dropout));
    }
    pub fn new(vs: &nn::VarStore, fclayers: FCLayerSet, data_dim: i64) -> Self {
        Self::create(vs, fclayers, data_dim, false)
    }
    pub fn create(
        vs: &nn::VarStore,
        fclayers: FCLayerSet,
        data_dim: i64,
        use_size_factor_key: bool,
    ) -> Self {
        let px_r_scale_dropout_decoder = nn::linear(
            vs.root() / "px_r_scale_dropout_decoder",
            fclayers.settings.d_hidden,
            data_dim * 3,
            Default::default(),
        );
        Self {
            fclayers,
            use_size_factor_key,
            px_r_scale_dropout_decoder,
        }
    }
}
