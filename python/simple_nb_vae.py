import sys

import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

VAR_EPS = 1e-4


class ULayerSet(nn.Module):
    def __init__(self, data_dim, out_dim, hidden_dims=[-1, -1], batch_norm=False, layer_norm=True, dropout=0.1, momentum=0.01, eps=1e-3, activation=nn.Mish):
        super().__init__()
        print(data_dim, out_dim, "data, out")
        def default_size(x):
            return x if x > 0 else int(np.ceil(data_dim / np.sqrt(data_dim / out_dim)))
        self.data_dim = data_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.n_layers = len(hidden_dims) + 1
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.dropout = dropout
        layerlist = []
        hidden_empty = len(self.hidden_dims) == 0
        first_out = default_size(self.hidden_dims[0] if not hidden_empty else out_dim)
        size_pairs = [(data_dim, first_out)]
        def get_dropout():
            return nn.Dropout(dropout, inplace=True)
        for hidden_index in range(len(hidden_dims) - 1):
            lhsize, rhsize = map(default_size, hidden_dims[hidden_index:hidden_index + 2])
            size_pairs.append((lhsize, rhsize))
        size_pairs.append(tuple(map(default_size, (hidden_dims[-1], out_dim))))
        for lhsize, rhsize in size_pairs:
            print("in, out: ", lhsize, rhsize, file=sys.stderr)
            layerlist.append(nn.Linear(lhsize, rhsize))
            if batch_norm:
                layerlist.append(nn.BatchNorm1d(rhsize, momentum=momentum, eps=eps))
            if layer_norm:
                layerlist.append(nn.LayerNorm(rhsize))
            layerlist.append(activation())
            if dropout > 0.:
                layerlist.append(get_dropout())
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x):
        return self.layers.forward(x)



class LayerSet(nn.Module):
    def __init__(self, data_dim, out_dim, hidden_dim=128, n_layers=3, batch_norm=False, layer_norm=True, dropout=0.1):
        super().__init__()
        self.data_dim = data_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.dropout = dropout

        def get_in(idx):
            return hidden_dim if idx > 0 else data_dim

        def get_out(idx):
            return hidden_dim if idx < n_layers - 1 else out_dim
        layerlist = []
        for idx in range(n_layers):
            ind = get_in(idx)
            outd = get_out(idx)
            layerlist.append(nn.Linear(ind, outd))
            # print(f"Layer {idx} from {ind} to {outd}")
            if batch_norm:
                layerlist.append(nn.BatchNorm1d(outd, momentum=0.01, eps=1e-3))
            if layer_norm:
                layerlist.append(nn.LayerNorm(outd))
        layerlist.append(nn.Mish())
        if dropout > 0.:
            layerlist.append(nn.Dropout(dropout, inplace=True))
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x):
        return self.layers.forward(x)


def log_likelihood_nb(x, mu, theta, scale):
    log_theta_mu_eps = (theta + mu + VAR_EPS).log()
    log_mu_eps = (mu + VAR_EPS).log()
    ret = theta * ((theta + VAR_EPS).log() - log_theta_mu_eps)
    + x * (log_mu_eps - log_theta_mu_eps)
    + (x + theta).lgamma()
    - theta.lgamma()
    - (x + 1).lgamma()
    return ret


def log_likelihood_zinb(x, mu, theta, scale, zi_logits):
    if zi_logits is None:
        return log_likelihood_nb(x, mu, theta, scale)
    softplus_pi = F.softplus(-zi_logits)

    theta_eps = theta + VAR_EPS
    mu_theta_eps = theta_eps + mu

    log_theta_mu_eps = mu_theta_eps.log()
    log_theta_eps = theta_eps.log()

    pi_theta_log = -zi_logits + theta * (log_theta_eps - log_theta_mu_eps)
    log_mu_eps = (mu + VAR_EPS).log()

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    case_non_zero = (-softplus_pi
                     + pi_theta_log
                     + x * (log_mu_eps - log_theta_mu_eps)
                     + (x + theta).lgamma()
                     - theta.lgamma()
                     - (x + 1).lgamma())

    xlt_eps = x < VAR_EPS
    xge_eps = xlt_eps.logical_not()

    mul_case_zero = case_zero * xlt_eps.float()
    mul_case_nonzero = case_non_zero * xge_eps.float()
    return mul_case_zero + mul_case_nonzero


class ZINB:
    def __init__(self, mu, theta, scale, zi_logits=None):
        self.mu = mu
        self.theta = theta
        self.scale = scale
        self.zi_logits = zi_logits

    def log_prob(self, x):
        return log_likelihood_zinb(x, mu=self.mu, theta=self.theta, scale=self.scale, zi_logits=self.zi_logits)


# inputs ->
#  fclayers
#    latent dim
#      sample (+ noise)
#    reconstruct        -> negative binomial loss
#    kl divergence loss -> make this fit the latent space model

class NBVAE(nn.Module):
    def __init__(self, data_dim, latent_dim, *, hidden_dim=128, linear_settings={}, zero_inflate=False):
        super().__init__()
        self.current_seed = 0
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.zero_inflate = zero_inflate
        n_layers = linear_settings.get("n_layers", 3)
        batch_norm = linear_settings.get("batch_norm", False)
        layer_norm = linear_settings.get("layer_norm", True)
        enc_dropout = linear_settings.get("dropout", 0.1)
        dec_dropout = linear_settings.get(
            "dropout", linear_settings.get("decoder_dropout", 0.0))
        if isinstance(hidden_dim, int):
            self.encoder = LayerSet(data_dim, out_dim=latent_dim, n_layers=n_layers, hidden_dim=hidden_dim,
                                    batch_norm=batch_norm, layer_norm=layer_norm, dropout=enc_dropout)
            self.decoder = LayerSet(latent_dim, out_dim=hidden_dim, n_layers=n_layers, hidden_dim=hidden_dim,
                                    batch_norm=batch_norm, layer_norm=layer_norm, dropout=dec_dropout)
            last_layer_dim = hidden_dim
        else:
            print(f"Encoding with {hidden_dim}, ecoding with {hidden_dim[::-1]}", file=sys.stderr)
            self.encoder = ULayerSet(data_dim, out_dim=latent_dim, hidden_dims=hidden_dim,
                                    batch_norm=batch_norm, layer_norm=layer_norm, dropout=enc_dropout)
            last_layer_dim = hidden_dim[0] if hidden_dim[0] > 0 else int(np.ceil(data_dim / np.sqrt(data_dim / latent_dim)))
            print(last_layer_dim, "last layer dim for latent = ", latent_dim, " and data = ", data_dim, file=sys.stderr)
            self.decoder = ULayerSet(latent_dim, out_dim=last_layer_dim, hidden_dims=hidden_dim[::-1],
                                     batch_norm=batch_norm, layer_norm=layer_norm, dropout=enc_dropout)
        self.px_r_scale_dropout_decoder = nn.Linear(
            last_layer_dim, data_dim * self.dim_mul())
        self.meanvar_encoder = nn.Linear(latent_dim, latent_dim * 2)

    def dim_mul(self):
        return 2 + self.zero_inflate

    def next_seed(self):
        ret = self.current_seed
        self.current_seed += 1
        return ret

    def decode(self, latent, library_size):
        px = self.decoder(latent)
        output_data = self.px_r_scale_dropout_decoder(px)
        output_chunked = output_data.chunk(self.dim_mul(), -1)
        dropout = output_chunked[2] if self.zero_inflate else None
        scale = F.softplus(output_chunked[1])
        r = output_chunked[0].exp()
        px_rate = library_size.exp() * scale
        return ZINB(scale=scale, theta=r, mu=px_rate, zi_logits=dropout)

    def run(self, x):
        latent = self.encoder(x)
        meanvar = self.meanvar_encoder(latent)
        stdeps = meanvar.chunk(2, -1)
        mu, logvar = stdeps
        std = (logvar * 0.5).exp() + VAR_EPS
        eps = torch.randn_like(std)
        gen = mu + eps * std
        latent_outputs = (gen, mu, logvar)
        library_size = x.sum(dim=[1]).log().unsqueeze(1)
        decoded = self.decode(gen, library_size)
        kl_loss = -0.5 * (1 + logvar - torch.pow(mu, 2) - logvar.exp())
        nb_recon_loss = -decoded.log_prob(x)
        losses = (kl_loss, nb_recon_loss)
        return latent_outputs, losses, decoded


    def forward(self, x):
        latent, losses, data = self.run(x)
        packed_inputs = list(latent) + list(losses)
        packed_inputs += [data.mu, data.theta, data.scale]
        if self.zero_inflate:
            packed_inputs.append(data.zi_logits)
        # print([x.shape for x in packed_inputs])
        packed_result = torch.cat(packed_inputs, -1)
        return packed_result

    def unpack(self, packed_result):
        latent_data = packed_result[:,:self.latent_dim * 5]
        (latent, gen, mu, logvar, kl_loss) = latent_data.chunk(5, -1)
        n_data_chunks = 4 + self.zero_inflate
        full_data = packed_result[:,self.latent_dim * 5:].chunk(n_data_chunks, -1)
        (nb_recon_loss, mu, theta, scale) = full_data[:4]
        zi_logits = full_data[4] if self.zero_inflate else None
        return ((latent, gen, mu, logvar), (kl_loss, nb_recon_loss), ZINB(scale=scale, mu=mu, theta=theta, zi_logits=zi_logits))

    def labeled_unpack(self, unpacked_result):
        if not isinstance(unpacked_result, tuple):
            unpacked_result = self.unpack(unpacked_result)
        latent, losses, dist = unpacked_result
        return {"latent": {"sampled": latent[0], "mu": latent[1], "logvar": latent[2]}, "loss": {"kl": losses[0], "recon": losses[1]}, "dist": dist}
