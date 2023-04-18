use clap::Parser;
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

#[derive(Parser)]
#[clap(author, version, about)]
struct Settings {
    #[clap(long, short)]
    #[clap(default_value = "10")]
    pub num_epochs: i32,

    #[clap(long, short)]
    #[clap(default_value = "1")]
    pub batch_size: i64,

    #[clap(long)]
    #[clap(short = 'l')]
    #[clap(default_value = "3")]
    pub n_layers: i64,

    #[clap(long)]
    #[clap(short = 'H')]
    #[clap(default_value = "64")]
    pub hidden_dim: i64,

    #[clap(long)]
    #[clap(short = 'L')]
    #[clap(default_value = "48")]
    pub latent_dim: i64,

    #[clap(long, short)]
    #[clap(default_value = "8")]
    pub report_index: i64,

    #[clap(long)]
    #[clap(default_value = "13")]
    pub seed: i64,

    #[clap(long, short)]
    #[clap(default_value = "10.")]
    pub kl_scale: f64,

    #[clap(long)]
    pub log1p: bool,
}

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
    let settings = Settings::parse();
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();
    let data = load_data();
    println!("Hello, world!");
    let latent_dim = settings.latent_dim;
    let labels = &data["labels"];
    let data = &data["data"];
    let data_dim = data.size2()?.1;
    tch::manual_seed(settings.seed);
    let n_layers = settings.n_layers;
    log::info!("Set seed");
    let encoder_settings = FCLayerSetSettings::new_simple(
        data_dim,
        latent_dim,
        Some(settings.hidden_dim),
        Some(n_layers),
        Default::default(),
    );
    log::info!("Made settings for enc");
    let mut decoder_settings = FCLayerSetSettings::new_simple(
        latent_dim,
        settings.hidden_dim,
        Some(settings.hidden_dim),
        Some(n_layers),
        Default::default(),
    );
    decoder_settings.no_dropout();
    log::info!("Created: enc/dec settings");
    let vs = nn::VarStore::new(best_device_available());
    log::info!("Created: varstore");
    let meanvar_encoder = nn::linear(
        vs.root(),
        encoder_settings.d_out,
        latent_dim * 2i64,
        Default::default(),
    );
    eprintln!("Device: {:?}", vs.device());
    let fclayers = FCLayerSet::new(&vs, encoder_settings);
    let zinb_decoder = ZINBDecoder::new(&vs, FCLayerSet::new(&vs, decoder_settings), data_dim);
    let mut net = nn::seq_t();
    if settings.log1p {
        net = net.add_fn(|xs| xs.log1p());
    }
    net = net.add(fclayers.layers).add(meanvar_encoder);
    let batch_size = settings.batch_size;
    let num_rows = data.size2()?.0;
    let num_cols = data.size2()?.1;
    eprintln!("Dataset of size {num_rows}/{num_cols}");
    let mut opt = nn::Adam::default().build(&vs, 1e-4)?;
    println!("Net built");
    for epoch in 0..settings.num_epochs {
        println!("Epoch {epoch}");
        let mut rloss_sum = 0.;
        let mut klloss_sum = 0.;
        let mut total_samples = 0;
        // let varsum = Tensor::zeros(&[latent_dim]);
        for (batch_index, (bdata, _)) in Iter::new(data, Some(labels), batch_size)
            .shuffle()
            .to_device(vs.device())
            .enumerate()
        {
            const TRAIN: bool = true;
            let current_bs = bdata.size2().unwrap().0 as i32;
            total_samples += current_bs;
            let bdata = bdata.to_kind(tch::Kind::Float).log1p();
            let latent = net.forward_t(&bdata, TRAIN);
            let (sampled_data, mu, logvar) = sample(&latent);
            let library_size = &bdata
                .sum_dim_intlist(Some([1].as_slice()), false, bdata.kind())
                .log()
                .unsqueeze(1i64);
            let decoded = zinb_decoder.decode(&sampled_data, library_size, TRAIN);
            let recon_loss = -decoded.log_prob(&bdata).sum(tch::Kind::Float);
            let kl_loss: Tensor =
                -0.5 * (1i64 + &logvar - mu.pow_tensor_scalar(2) - logvar.exp()).sum(Kind::Float);
            let current_recon_loss: f64 = recon_loss.double_value(&[]);
            let current_kl_loss: f64 = kl_loss.double_value(&[]);
            rloss_sum += current_recon_loss;
            klloss_sum += current_kl_loss;
            let loss = recon_loss + kl_loss * settings.kl_scale;
            opt.backward_step(&loss);
            if batch_index % settings.report_index as usize == 0 {
                let num_processed = (batch_index + 1) * batch_size as usize;
                log::info!(
                    "epoch: {:4} after {num_processed} mean train error of this epoch: recon {:5.2}, kl {:5.2}, total {:5.2}",
                    epoch,
                    rloss_sum / (total_samples as f64),
                    klloss_sum / (total_samples as f64),
                    (klloss_sum + rloss_sum) / (batch_index + 1) as f64,
                );
                log::info!(
                    "epoch: {:4} after {num_processed} mean train error of this batch: recon {:5.2}, kl {:5.2}, total {:5.2}",
                    epoch,
                    current_recon_loss / (current_bs as f64),
                    current_kl_loss / (current_bs as f64),
                    (current_kl_loss + current_recon_loss) / (current_bs as f64),
                );
            }
        }
        let loss_sum = rloss_sum + klloss_sum;
        println!("epoch: {:4} train error: {:5.2}", epoch, loss_sum);
    }
    vs.save(format!(
        "vae-pbmc.epochs.{}.hid.{}.latent.{}.nlayers.{}.seed.{}.ot",
        settings.num_epochs,
        settings.hidden_dim,
        settings.latent_dim,
        settings.n_layers,
        settings.seed,
    ))
    .unwrap();
    Ok(())
}
