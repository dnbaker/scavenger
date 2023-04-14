use tch::{
    nn, nn::BatchNormConfig, nn::Module, nn::ModuleT, nn::OptimizerConfig, nn::SequentialT, Device,
    Kind, Reduction, TchError, Tensor,
};

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
    pub fn from_value(val: f64) -> Self {
        Self { data: vec![val] }
    }
    pub fn from_vec(vals: Vec<f64>) -> Self {
        Self { data: vals }
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
        let activation = Activation::LeakyRelu;
        let is_squared = false;
        Self {
            activation,
            is_squared,
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
        };
        if self.is_squared {
            output.square_()
        } else {
            output
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
}

impl FCLayerSetSettings {
    pub fn default_dropout(value: Option<f64>) -> BroadcastingParameter {
        BroadcastingParameter::from_value(value.unwrap_or(0.1))
    }
    pub fn new_simple(
        d_in: i64,
        d_out: i64,
        d_hidden: Option<i64>,
        n_layers: Option<i64>,
        dropout: Option<BroadcastingParameter>,
    ) -> Self {
        let n_layers: usize = n_layers.unwrap_or(1) as usize;
        Self {
            d_in,
            d_out,
            d_hidden: d_hidden.unwrap_or(128),
            n_layers: n_layers as i64,
            dropout: dropout.unwrap_or(Self::default_dropout(Some(0.125))),
            batch_norm: vec![false; n_layers],
            layer_norm: vec![true; n_layers],
            activation: vec![ActivationPlan::default(); n_layers],
        }
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
        ret = ret.add(nn::linear(
            // *vs / format!("fclayer{in_dim}:{out_dim}:{layer_index}:linear"),
            vs.root(),
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
                vs.root() / format!("fclayer{in_dim}:{out_dim}:{layer_index}:batchnorm"),
                out_dim,
                config,
            ));
        }
        if self.layer_norm[layer_index as usize] {
            let layer_norm = nn::layer_norm(
                vs.root() / format!("layer_norm{layer_index}"),
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

pub enum ScaleAct {
    SoftMax,
    SoftPlus,
}

pub struct ZINBDecoder {
    // Important: make sure there's no dropout here.
    pub fclayers: FCLayerSet,
    pub px_r_scale_dropout_decoder: tch::nn::Linear, // This layer emits both x and r. We then chunk it to extract r and dropout separately.
    use_size_factor_key: bool,
}

/*
#if 0
#endif
        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)
*/

impl ZINBDecoder {
    fn hidden_dim(&self) -> i64 {
        return self.fclayers.settings.d_hidden;
    }
    pub fn decode(
        &self,
        input: &tch::Tensor,
        library_size: &tch::Tensor,
        train: bool,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let px = self.fclayers.layers.forward_t(&input, train);
        let r_dropout = self.px_r_scale_dropout_decoder.forward_t(&px, train);
        let mut r_dropout = input.chunk(3, -1);
        let px_scale = r_dropout.remove(2);
        let px_scale = if self.use_size_factor_key {
            px_scale.softmax(-1, Kind::Float)
        } else {
            px_scale.softplus()
        };
        let dropout = r_dropout.remove(1);
        let r = r_dropout.remove(0);
        let px_rate = library_size.exp() * &px_scale;
        return (px_scale, r, px_rate, dropout);
    }
    pub fn create(vs: &nn::VarStore, fclayers: FCLayerSet, use_size_factor_key: bool) -> Self {
        /*
        let mut scale_decoder = nn::seq_t().add(nn::linear(
            vs.root(),
            self.fclayers.settings.d_hidden,
            self.fclayers.settings.d_out,
            Default::default(),
        ));
        if !self.use_size_factor_key {
        scale_decoder = scale_decoder.add_fn(|xs, t| {
            xs.softmax(-1, Kind::Float);
        });
        } else {
        scale_decoder = scale_decoder.add_fn(|xs| {
            xs.softplus();
        });
        }
        */
        let px_r_scale_dropout_decoder = nn::linear(
            vs.root(),
            fclayers.settings.d_hidden,
            fclayers.settings.d_out * 3,
            Default::default(),
        );
        Self {
            fclayers,
            use_size_factor_key,
            px_r_scale_dropout_decoder,
        }
    }
}
