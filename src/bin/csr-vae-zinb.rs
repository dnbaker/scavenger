use clap::Parser;
use scavenger::iter::{CSRMatrix, IterCSR};
use scavenger::nnutil::{self, *};
use std::collections::{BTreeSet, HashMap};
use tch::{self, nn, nn::ModuleT, nn::OptimizerConfig, Kind, Tensor};

fn load_data(source: std::path::PathBuf) -> HashMap<String, Tensor> {
    let tensors = tch::Tensor::read_npz(source).unwrap();
    tensors.into_iter().collect::<HashMap<String, Tensor>>()
}

#[derive(Parser)]
#[clap(author, version, about)]
struct Settings {
    #[clap(long, short)]
    #[clap(default_value = "10")]
    pub num_epochs: i32,

    #[clap(long, short)]
    #[clap(default_value = "16")]
    pub batch_size: i64,

    #[clap(long)]
    #[clap(short = 'l')]
    #[clap(default_value = "3")]
    pub n_layers: i64,

    #[clap(long)]
    #[clap(short = 'H')]
    #[clap(default_value = "256")]
    pub hidden_dim: i64,

    #[clap(long)]
    #[clap(short = 'L')]
    #[clap(default_value = "128")]
    pub latent_dim: i64,

    #[clap(long, short)]
    #[clap(default_value = "8")]
    pub report_index: i64,

    #[clap(long)]
    #[clap(default_value = "13")]
    pub seed: i64,

    #[clap(long, short)]
    #[clap(default_value = "0.1")]
    pub dropout: f64,

    #[clap(long, short)]
    #[clap(default_value = "1.")]
    pub kl_scale: f64,

    #[clap(long, short)]
    #[clap(default_value = "false")]
    pub add_classifier_loss: bool,

    #[clap(long, short)]
    #[clap(default_value = "false")]
    pub zero_inflate: bool,

    #[clap(long)]
    pub log1p: bool,

    pub data_path: Option<String>,

    #[clap(long, short)]
    #[clap(default_value = "false")]
    pub early_termination: bool,
}

fn main() -> Result<(), tch::TchError> {
    let settings = Settings::parse();
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();
    let data_path = settings.data_path.unwrap_or(String::from(
        "/Users/dnb13/Desktop/code/compressed_bundle/PBMC/pbmc.csr.npz",
    ));
    let data = load_data(std::path::PathBuf::from(&data_path[..]));
    let latent_dim = settings.latent_dim;
    let labels = data.get("labels");
    let num_labels = labels
        .map(|x| x.iter::<i64>().unwrap().collect::<BTreeSet<_>>().len() as i64)
        .unwrap_or(0i64);
    let device = nnutil::best_device_available();
    let mut vs = nn::VarStore::new(device);
    let csr_data = data.get("indptr").map(|indptr| CSRMatrix {
        data: data["data"].shallow_clone().to_kind(Kind::Float),
        indptr: indptr.shallow_clone(),
        indices: data["indices"].shallow_clone(),
        shape: Vec::<i64>::try_from(&data["shape"]).unwrap(),
        kind: Kind::Float,
    });
    if csr_data.is_none() {
        panic!("Testing csr format");
    }
    let csr_data = csr_data.unwrap();
    let shape = &csr_data.shape[..];
    let num_rows = shape[0] as i64;
    let num_cols = shape[1] as i64;
    let data_dim = num_cols;
    tch::manual_seed(settings.seed);
    let n_layers = settings.n_layers;
    log::info!("Set seed");
    let zinb_settings = ZINBVAESettings::new(
        data_dim,
        settings.hidden_dim,
        n_layers,
        latent_dim,
        Some(BroadcastingParameter::from(settings.dropout)),
        Default::default(),
        Some(settings.zero_inflate),
    );
    log::info!("Created: varstore");
    let vae = ZINBVAE::new(&vs, zinb_settings);
    eprintln!("Device: {:?}", vs.device());
    if settings.log1p {
        panic!("log1p not supported for this case because lazy");
    }
    let batch_size = settings.batch_size;
    let classifier_layer = nn::linear(
        vs.root() / "classifier",
        latent_dim,
        num_labels,
        Default::default(),
    );
    eprintln!("Dataset of size {num_rows}/{data_dim}");
    let mut opt = nn::Adam::default().build(&vs, 1e-4)?;
    for epoch in 0..settings.num_epochs {
        log::info!("Epoch {epoch}");
        let mut rloss_sum = 0.;
        let mut klloss_sum = 0.;
        let mut class_loss_sum = 0.;
        let mut total_samples = 0;
        /*
        for (batch_index, (bdata, blabels)) in Iter::new(data, labels, batch_size)
            .shuffle_sparse(is_sparse)
        */
        for (batch_index, (bdata, blabels)) in IterCSR::new(&csr_data, labels, batch_size)
            .to_device(vs.device())
            .enumerate()
        {
            let current_bs = bdata.size2().unwrap().0 as i32;
            total_samples += current_bs;
            // let bdata = bdata.to_kind(tch::Kind::Float).log1p();
            let data_output = vae.forward_t(&bdata, true);
            let (latent_data, losses, _zinb_model) = vae.unpack_output(&data_output);
            let (kl, recon) = losses;
            let recon_loss = recon.sum(tch::Kind::Float);
            let kl_loss: Tensor = kl.sum(Kind::Float);
            let current_recon_loss: f64 = recon_loss.double_value(&[]);
            let current_kl_loss: f64 = kl_loss.double_value(&[]);
            rloss_sum += current_recon_loss;
            klloss_sum += current_kl_loss;
            let mut loss = recon_loss + kl_loss * settings.kl_scale;
            let current_class_loss = if settings.add_classifier_loss {
                blabels
                    .map(|blabels| {
                        let classifier_loss = classifier_layer
                            .forward_t(&latent_data.0, true)
                            .cross_entropy_for_logits(&blabels)
                            .sum(tch::Kind::Float);
                        let classifier_double = classifier_loss.double_value(&[]);
                        class_loss_sum += classifier_double;
                        log::debug!(
                            "Classifier loss at {epoch}:{batch_index}: {}",
                            classifier_double
                        );
                        loss += classifier_loss;
                        classifier_double
                    })
                    .unwrap_or(0.)
            } else {
                0.
            };
            opt.backward_step(&loss);
            if batch_index % settings.report_index as usize == 0 {
                let num_processed = (batch_index + 1) * batch_size as usize;
                log::info!(
                    "epoch: {:4} after {num_processed} mean train error of this epoch: recon {:5.2}, kl {:5.2}, classification loss {:5.2} total {:5.2}",
                    epoch,
                    rloss_sum / (total_samples as f64),
                    klloss_sum / (total_samples as f64),
                    class_loss_sum / (total_samples as f64),
                    (klloss_sum + rloss_sum + class_loss_sum) / (batch_index + 1) as f64,
                );
                log::info!(
                    "epoch: {:4} after {num_processed} mean train error of this batch: recon {:5.2}, kl {:5.2}, classification {:5.2} total {:5.2}",
                    epoch,
                    current_recon_loss / (current_bs as f64),
                    current_kl_loss / (current_bs as f64),
                    current_class_loss / (current_bs as f64),
                    (current_kl_loss + current_recon_loss + current_class_loss) / (current_bs as f64),
                );
            }
            if settings.early_termination {
                break;
            }
        }
        let loss_sum = rloss_sum + klloss_sum + class_loss_sum;
        println!("epoch: {:4} train error: {:5.2}", epoch, loss_sum);
    }
    vs.save(format!(
        "weights/vae-pbmc.epochs.{}.hid.{}.latent.{}.nlayers.{}.seed.{}.ot",
        settings.num_epochs,
        settings.hidden_dim,
        settings.latent_dim,
        settings.n_layers,
        settings.seed,
    ))
    .unwrap();
    vs.freeze();
    let mut closure = |input: &[Tensor]| vec![vae.forward_t(&input[0], false)];
    let model = tch::CModule::create_by_tracing(
        &format!("{}NBVAE", if settings.zero_inflate { "ZI" } else { "" })[..],
        "forward",
        &[Tensor::randn([16, data_dim], (Kind::Float, device))],
        &mut closure,
    )?;
    model.save(format!(
        "{}nbvae.epochs{}.hid{}.latent{}.nlayers{}.seed{}.pt",
        if settings.zero_inflate { "zi" } else { "" },
        settings.num_epochs,
        settings.hidden_dim,
        settings.latent_dim,
        settings.n_layers,
        settings.seed,
    ))?;
    Ok(())
}
