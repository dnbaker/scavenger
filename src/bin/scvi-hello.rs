use rust_scvi::nnutil::*;
use tch::{self, nn, nn::ModuleT, nn::OptimizerConfig, *};

const IMAGE_DIM: i64 = 784;
const LABELS: i64 = 10;

fn main() -> Result<(), TchError> {
    let mnist = vision::mnist::load_dir("data")?;
    let device = tch::Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let vae = Vae::new(&vs.root());
    println!("Hello, world!");
    let settings = FCLayerSetSettings::new_simple(
        IMAGE_DIM as i32,
        64,
        Some(LABELS as i32),
        Some(3),
        Default::default(),
    );
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let fclayers = FCLayerSet::new(&vs, settings);
    let net = &fclayers.layers;
    let mut opt = nn::Adam::default().build(&vs, 1e-4)?;
    println!("Net built");
    for epoch in 0..200 {
        println!("Epoch {epoch}");
        for (bimages, blabels) in mnist.train_iter(256).shuffle().to_device(vs.device()) {
            let loss = net
                .forward_t(&bimages, true)
                .cross_entropy_for_logits(&blabels);
            opt.backward_step(&loss);
        }
        let test_accuracy = net.batch_accuracy_for_logits(
            &mnist.test_images,
            &mnist.test_labels,
            vs.device(),
            1024,
        );
        println!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy);
    }
    Ok(())
}
