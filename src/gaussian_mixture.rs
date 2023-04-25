use crate::nnutil;
use itertools::Itertools;
use tch::{
    nn, nn::BatchNormConfig, nn::Module, nn::ModuleT, nn::SequentialT, Device, IndexOp, Kind,
    Tensor,
};

struct LatentGM {
    n_components: i64,
    mixture_pi: Tensor, // logits for mixture components (batch, n_components)
                        // p(y) = cat(mixture_pi)
}

impl LatentGM {}
