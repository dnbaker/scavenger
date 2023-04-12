use rust_scvi::nnutil::*;
use tch::{self, nn, *};

fn main() {
    let device = tch::Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let vae = Vae::new(&vs.root());
    println!("Hello, world!");
    let settings =
        FCLayerSetSettings::new_simple(128, 128, Some(128), Some(3), Default::default());
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let fclayers = FCLayerSet::new(&vs, settings);
}
