use rust_scvi::nnutil::*;
use tch::{nn,self};

fn main() {
    let device = tch::Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let vae = Vae::new(&vs.root());
    println!("Hello, world!");
}
