use tch::{
    nn, nn::BatchNormConfig, nn::Module, nn::OptimizerConfig, nn::SequentialT, Kind, Reduction,
    TchError, Tensor,
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

#[derive(Clone, Debug)]
pub struct FCLayerSetSettings {
    pub d_in: i32,
    pub d_out: i32,
    pub d_hidden: i32,
    pub n_layers: i32,
    // Either a singular value, or one per layer
    pub dropout: BroadcastingParameter,
    pub batch_norm: Vec<bool>,
    pub layer_norm: Vec<bool>,
    pub activation: Vec<bool>,
}

fn leaky_relu(xs: &Tensor) -> Tensor {
    xs.maximum(&(xs * 0.2))
}

impl FCLayerSetSettings {
    pub fn default_dropout(value: Option<f64>) -> BroadcastingParameter {
        BroadcastingParameter::from_value(value.unwrap_or(0.1))
    }
    pub fn new_simple(
        d_in: i32,
        d_out: i32,
        d_hidden: Option<i32>,
        n_layers: Option<i32>,
        dropout: Option<BroadcastingParameter>,
    ) -> Self {
        let n_layers: usize = n_layers.unwrap_or(1) as usize;
        Self {
            d_in,
            d_out,
            d_hidden: d_hidden.unwrap_or(128),
            n_layers: n_layers as i32,
            dropout: dropout.unwrap_or(Self::default_dropout(Some(0.125))),
            batch_norm: vec![true; n_layers],
            layer_norm: vec![false; n_layers],
            activation: vec![true; n_layers],
        }
    }
    pub fn make_layer(&self, vs: &nn::VarStore, layer_index: i32) -> SequentialT {
        let mut ret = nn::seq_t();
        let in_dim = (if layer_index == 0 {
            self.d_in
        } else {
            self.d_hidden
        }) as i64;
        let out_dim = (if layer_index == self.n_layers - 1 {
            self.d_out
        } else {
            self.d_hidden
        }) as i64;
        ret = ret.add(nn::linear(
            // *vs / format!("fclayer{in_dim}:{out_dim}:{layer_index}:linear"),
            vs.root(),
            in_dim,
            out_dim,
            Default::default(),
        ));
        /*
        if self.batch_norm[layer_index as usize] {
            let mut config = BatchNormConfig::default();
            config.momentum = 0.01;
            config.eps = 1e-3;
            ret = ret.add(nn::batch_norm1d(
                *vs / format!("fclayer{in_dim}:{out_dim}:{layer_index}:batchnorm"),
                out_dim,
                config,
            ));
        }
        */
        // Dropout doesn't need to be built, but we'll have to include it in the pass.
        ret = ret.add_fn(leaky_relu);
        let dropout_param = self.dropout.value(layer_index as i64);
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
