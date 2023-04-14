use rust_scvi::iter::Iter;
use rust_scvi::nnutil::*;
use std::collections::HashMap;
use tch::{self, nn, nn::ModuleT, nn::OptimizerConfig, *};

fn load_data() -> HashMap<String, Tensor> {
    let source = "/Users/dnb13/Desktop/code/compressed_bundle/PBMC/pbmc.sparse.npz";
    let tensors = tch::Tensor::read_npz(source).unwrap();
    tensors.into_iter().collect::<HashMap<String, Tensor>>()
}
const VAR_EPS: f64 = 1e-4;

fn sample(input: &Tensor) -> (Tensor, Tensor, Tensor) {
    let mut stdeps = input.chunk(2, -1);
    /*
    // let (mu, logvar) = stdeps;
     */
    let logvar = stdeps.remove(1);
    let mu = stdeps.remove(0);
    let std = (&logvar * 0.5).exp() + VAR_EPS;
    let eps = std.randn_like();
    (&mu + eps * std, mu, logvar)
}

fn main() -> Result<(), TchError> {
    let data = load_data();
    println!("Hello, world!");
    let latent_dim: i64 = 32;
    let labels = &data["labels"];
    let data = &data["data"];
    let data_dim = data.size2()?.1;
    tch::manual_seed(13i64);
    const NLAYERS: i64 = 3i64;
    const HIDDEN_DIM: i64 = 64i64;
    let encoder_settings = FCLayerSetSettings::new_simple(
        data_dim,
        latent_dim,
        Some(HIDDEN_DIM),
        Some(NLAYERS),
        Default::default(),
    );
    let decoder_settings = FCLayerSetSettings::new_simple(
        latent_dim,
        data_dim,
        Some(HIDDEN_DIM),
        Some(NLAYERS),
        Default::default(),
    );
    let vs = nn::VarStore::new(best_device_available());
    let meanvar_encoder = nn::linear(
        vs.root(),
        encoder_settings.d_out,
        latent_dim * 2i64,
        Default::default(),
    );
    eprintln!("Device: {:?}", vs.device());
    let fclayers = FCLayerSet::new(&vs, encoder_settings);
    let decoder_fclayers = FCLayerSet::new(&vs, decoder_settings);
    let net = nn::seq_t().add(fclayers.layers).add(meanvar_encoder);
    const BATCH_SIZE: i64 = 64;
    let num_rows = data.size2()?.0;
    let num_cols = data.size2()?.1;
    eprintln!("Dataset of size {num_rows}/{num_cols}");
    let mut opt = nn::Adam::default().build(&vs, 1e-4)?;
    println!("Net built");
    for epoch in 0..5 {
        println!("Epoch {epoch}");
        let mut rloss_sum = 0.;
        let mut klloss_sum = 0.;
        // let varsum = Tensor::zeros(&[latent_dim]);
        for (batch_index, (bdata, _)) in Iter::new(data, Some(labels), BATCH_SIZE)
            .shuffle()
            .to_device(vs.device())
            .enumerate()
        {
            let bdata = bdata.to_kind(tch::Kind::Float).log1p();
            let latent = net.forward_t(&bdata, true);
            let (sampled_data, mu, logvar) = sample(&latent);
            let decoded = decoder_fclayers.layers.forward_t(&sampled_data, true);
            let recon_loss = decoded.mse_loss(&bdata, tch::Reduction::Mean);
            let kl_loss: Tensor =
                -0.5 * (1i64 + &logvar - mu.pow_tensor_scalar(2) - logvar.exp()).sum(Kind::Float);
            rloss_sum += &recon_loss.double_value(&[]);
            klloss_sum += &kl_loss.double_value(&[]);
            let loss = recon_loss + kl_loss;
            opt.backward_step(&loss);
            if batch_index % 64 == 0 {
                let num_processed = (batch_index + 1) * BATCH_SIZE as usize;
                println!(
                    "epoch: {:4} after {num_processed} mean train error of this epoch: recon {:5.2}, kl {:5.2}, total {:5.2}",
                    epoch,
                    rloss_sum / (batch_index + 1) as f64,
                    klloss_sum / (batch_index + 1) as f64,
                    (klloss_sum + rloss_sum) / (batch_index + 1) as f64,
                );
            }
        }
        let loss_sum = rloss_sum + klloss_sum;
        println!("epoch: {:4} train error: {:5.2}", epoch, loss_sum);
    }
    Ok(())
}
