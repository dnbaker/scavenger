import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Gamma, Poisson
import numpy as np

from scavenger.components import nchoose2, default_size, in_outs, ULayerSet, LayerSet

VAR_EPS = 1e-4



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

    def mean(self):
        if self.zi_logits is None:
            return self.mu
        return self.mu * self.zi_probs()

    def zi_probs(self):
        sigmoid_logits = F.sigmoid(self.zi_logits)
        return F.softmax(sigmoid_logits, dim=-1)

    def variance(self):
        # Perhaps not correct for zi case, but this isn't important.
        mean = self.mean()
        return mean + (torch.square(mean)) / self.theta

    def sample(self, shape=None):
        # concentration = theta
        # rate = theta / mu
        # Important remark: Gamma is parametrized by the rate = 1/scale!
        sample_shape = shape or torch.Size()
        gamma_d = Gamma(concentration=self.theta, rate=self.theta/self.mu)
        p_means = gamma_d.sample(sample_shape)
        counts = Poisson(
            torch.clamp(p_means, max=1e8)
        ).sample()
        # (Sample, Batch, Dim)
        if self.zi_logits is not None:
            is_zero = torch.rand_like(samp) <= self.zi_probs()
            counts = torch.where(is_zero, 0.0, counts)
        return counts


def tril2full_and_nonneg(tril, dim, nonneg_function=F.softplus):
    '''
        Input: Tensor, lower triangular covariance matrix
            (Batch, triangular size) [N Choose 2) + N]
        Output: Tensor, full covariance matrix
            (Batch, N, N)
    '''
    batch, *remaining = tril.shape
    xidx, yidx = torch.tril_indices(dim, dim)
    diag_idx_mask = torch.arange(xidx.shape[0])
    xdiag, ydiag = (z[diag_idx_mask] for z in (xidx, yidx))
    unpacked = torch.zeros([batch, dim, dim], dtype=tril.dtype)
    unpacked[:, xidx, yidx] = tril
    unpacked[:, xdiag, ydiag] = nonneg_function(unpacked[:, xdiag, ydiag])
    # Ensure positive definite covariance matrix by making variance non-negative
    # Can use softplus or exp, but softplus is probably more stable
    return unpacked


# inputs ->
#  fclayers
#    latent dim
#      sample (+ noise)
#    reconstruct        -> negative binomial loss
#    kl divergence loss -> make this fit the latent space model

class NBVAE(nn.Module):
    def __init__(self, data_dim, latent_dim, *, hidden_dim=128, linear_settings={}, zero_inflate=False, full_cov=None, categorical_class_sizes=[], expand_mul=4, nonneg_function=F.softplus):
        super().__init__()
        self.current_seed = 0
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.zero_inflate = zero_inflate
        self.full_cov = full_cov
        self.nonneg_function = nonneg_function
        if full_cov is None:
            full_cov = latent_dim < 512
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
            self.encoder = ULayerSet(data_dim, out_dim=latent_dim, hidden_dims=hidden_dim,
                                     batch_norm=batch_norm, layer_norm=layer_norm, dropout=enc_dropout)
            last_layer_dim = hidden_dim[0] if hidden_dim[0] > 0 else int(
                np.ceil(data_dim / np.sqrt(data_dim / latent_dim)))
            self.decoder = ULayerSet(latent_dim, out_dim=last_layer_dim, hidden_dims=hidden_dim[::-1],
                                     batch_norm=batch_norm, layer_norm=layer_norm, dropout=dec_dropout)
        self.px_r_scale_dropout_decoder = nn.Linear(
            last_layer_dim, data_dim * self.dim_mul())
        meanvar_out = self.num_latent_gaussian_parameters()
        self.meanvar_encoder = ULayerSet(latent_dim, meanvar_out, hidden_dims=[
                                         latent_dim * expand_mul], skip_last_activation=True)

        self.categorical_class_sizes = categorical_class_sizes
        num_classes = sum(categorical_class_sizes)
        # Give the classification projection the latent representation as well as the last layer's values
        # Could use an MLP, but I think it's better to make it linear since it's interpretable
        self.class_proj = nn.Linear(last_layer_dim + self.latent_dim, sum(categorical_class_sizes)) if num_classes else None


    def num_covariance_parameters_variables(self):
        return nchoose2(self.latent_dim) + self.latent_dim

    def num_latent_gaussian_parameters(self):
        if not self.full_cov:
            return self.latent_dim * 2
        num_covar = self.num_covariance_parameters_variables()
        num_mean = self.latent_dim
        return num_covar + num_mean

    def dim_mul(self):
        return 2 + self.zero_inflate

    '''
    def next_seed(self):
        ret = self.current_seed
        self.current_seed += 1
        return ret
    '''

    def decode(self, latent, library_size):
        px = self.decoder(latent)
        output_data = self.px_r_scale_dropout_decoder(px)
        output_chunked = output_data.chunk(self.dim_mul(), -1)
        dropout = output_chunked[2] if self.zero_inflate else None
        scale = F.softplus(output_chunked[1])
        r = output_chunked[0].exp()
        px_rate = library_size * scale

        # Logits for classification
        if hasattr(self, 'class_proj') and self.class_proj is not None:
            classification_logits = self.class_proj(torch.cat([px, latent]))
            classification_loss = F.binary_cross_entropy_with_logits(classification_logits, reduction='none')
        else:
            classification_logits = None
            classification_loss = None
        return {"model": ZINB(scale=scale, theta=r, mu=px_rate, zi_logits=dropout),
                "classification_loss": classification_loss,
                "classification_logits": classification_logits}

    def run(self, x):
        latent = self.encoder(x)
        meanvar = self.meanvar_encoder(latent)
        mu = meanvar[:, :self.latent_dim]
        if self.full_cov:
            full_cov = tril2full_and_nonneg(F.softplus(
                meanvar[:, self.latent_dim:]), self.latent_dim, self.nonneg_function)
            # dist = torch.distributions.MultivariateNormal(loc=mu, scale_tril=full_cov)
            # gen = dist.rsample()
            batch_size = x.shape[0]
            noise = torch.randn((batch_size, self.latent_dim))
            noise_squeeze = noise.unsqueeze(-1)
            mulout = torch.bmm(full_cov, noise_squeeze).squeeze()
            # print("mu: ", mu.shape, "fullcov", full_cov.shape, "noise_sq", noise_squeeze.shape, "mulout", mulout.shape)
            gen = mu + mulout
            latent_range = torch.arange(self.latent_dim)
            var = full_cov[:, latent_range, latent_range]
            logvar = var.log()
            # log_q_z = -.5*(noise.square() + logvar + log2pi)
            # log_p_z = -.5*(z**2 + log2pi)
            # the log2pi cancels out
            # log_q_z = -.5*(noise.square() + logvar)
            # log_p_z = -.5*(z**2)
            # kl_loss = -log_p_z + log_q_z
            # log_q_z = -.5*(noise.square() + logvar)
            # log_p_z = -.5*(z**2)
            kl_loss = .5 * (gen.square() - (noise.square() + logvar))
            # kl_loss = -log_p_z + log_q_z
            # https://arxiv.org/pdf/1906.02691.pdf, page 29
        else:
            full_cov = None
            var = self.nonneg_function(meanvar[:, self.latent_dim:])
            logvar = var.log()
            std = (logvar * 0.5).exp() + VAR_EPS
            eps = torch.randn_like(std)
            gen = mu + eps * std
            kl_loss = -0.5 * (1 + logvar - torch.pow(mu, 2) - logvar.exp())
        latent_outputs = (gen, mu, logvar)
        library_size = x.sum(dim=[1]).unsqueeze(1)
        decoded = self.decode(gen, library_size)
        classification_loss = decoded['classification_loss']
        class_info = {x: decoded.get(x) for x in ('classification_loss', 'classification_logits')}
        decoded = decoded['model']
        nb_recon_loss = -decoded.log_prob(x)
        losses = (kl_loss, nb_recon_loss)
        if classification_loss is not None:
            losses = tuple(list(losses) + [classification_loss])
        if full_cov is not None:
            full_cov = full_cov.reshape(full_cov.shape[0], -1)
        return latent_outputs, losses, decoded, full_cov, class_info

    def forward(self, x):
        latent, losses, data, full_cov, class_info = self.run(x)
        classification_loss = losses[2:]
        if len(classification_loss) == 0:
            classification_loss = None
        kl_loss, recon_loss = losses[:2]
        class_loss = losses[2] if len(losses) >= 3 else None
        packed_inputs = list(latent) + [kl_loss]
        if full_cov is not None:
            packed_inputs.append(full_cov)
        classification_logits = class_info.get('classification_logits')
        if classification_loss is not None:
            packed_inputs.append(classification_loss)
            assert classification_logits is not None
            packed_inputs.append(classification_logits)
        packed_inputs += list(losses)
        packed_inputs += [data.mu, data.theta, data.scale]
        if self.zero_inflate:
            packed_inputs.append(data.zi_logits)
        assert all(len(x.shape) == 2 for x in packed_inputs)
        packed_result = torch.cat(packed_inputs, -1)
        # print(f"packed_result: {packed_result}", file=sys.stderr)
        return packed_result

    def unpack(self, packed_result):
        total = packed_result.shape[1]
        # print("total:", packed_result.shape)
        end_of_latent_without_cov = self.latent_dim * 5
        end_of_latent = end_of_latent_without_cov + \
            (self.latent_dim ** 2 if self.full_cov else 0)
        start_of_class = end_of_latent
        end_of_class = end_of_latent + sum(self.categorical_class_sizes) * 2
        latent_data = packed_result[:, :end_of_latent]
        latent, gen, mu, logvar, kl_loss = latent_data[:, :end_of_latent_without_cov].chunk(
            5, -1)
        if self.full_cov:
            remaining = latent_data[:, end_of_latent_without_cov:]
            # print(
            #    f"Remaining: {remaining.shape}. numcov: {self.num_cov_variables()}")
            full_cov = remaining[:, :]
            assert full_cov.shape[1] == self.latent_dim ** 2, f"{self.latent_dim**2} vs {full_cov.shape}"
            assert kl_loss.shape[1] == self.latent_dim, f"{kl_loss.shape}, vs latent {self.latent_dim}"
        class_data = packed_result[:,start_of_class:end_of_class] if self.categorical_class_sizes else None
        if class_data is not None:
            class_loss, class_logits =  class_data.chunk(2, -1)
        else:
            class_loss = None
            class_logits = None
        n_data_chunks = 4 + self.zero_inflate
        full_data = packed_result[:, end_of_class:].chunk(n_data_chunks, -1)
        (nb_recon_loss, mu, theta, scale) = full_data[:4]
        zi_logits = full_data[4] if self.zero_inflate else None
        # print([x.shape for x in (latent, gen, mu, logvar, kl_loss)], "latent")
        # print([x.shape for x in (nb_recon_loss, scale, mu, theta)],
        #      "data: recon, scale, mu, theta")
        losses = (kl_loss, nb_recon_loss)
        if class_loss is not None:
            losses = kl_loss + (class_loss,)
        return ((latent, gen, mu, logvar), losses, ZINB(scale=scale, mu=mu, theta=theta, zi_logits=zi_logits), (class_logits, class_loss) if class_logits is not None else None)

    def labeled_unpack(self, unpacked_result):
        if not isinstance(unpacked_result, tuple):
            unpacked_result = self.unpack(unpacked_result)
        latent, losses, dist, class_info = unpacked_result
        ret = {"latent": {"sampled": latent[0], "mu": latent[1], "logvar": latent[2]}, "loss": {"kl": losses[0], "recon": losses[1]}, "dist": dist, "class": class_info}
        if len(losses) > 2:
            ret['loss']['class'] = losses[2]
        return ret


class UNetDiscriminator(nn.Module):
    # Simple U
    def __init__(self, data_dim, latent_dim, *, hidden_dim=128, linear_settings={}, categorical_classes=[]):
        super().__init__()
        self.current_seed = 0
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        if full_cov is None:
            full_cov = latent_dim < 512
        n_layers = linear_settings.get("n_layers", 3)
        batch_norm = linear_settings.get("batch_norm", False)
        layer_norm = linear_settings.get("layer_norm", True)
        enc_dropout = linear_settings.get("dropout", 0.1)
        dec_dropout = linear_settings.get(
            "dropout", linear_settings.get("decoder_dropout", 0.0))
        self.encoder = ULayerSet(data_dim, out_dim=latent_dim, hidden_dims=hidden_dim,
                                 batch_norm=batch_norm, layer_norm=layer_norm, dropout=enc_dropout)
        last_layer_dim = hidden_dim[0] if hidden_dim[0] > 0 else int(
            np.ceil(data_dim / np.sqrt(data_dim / latent_dim)))
        self.decoder = ULayerSet(latent_dim, out_dim=last_layer_dim, hidden_dims=hidden_dim[::-1],
                                 batch_norm=batch_norm, layer_norm=layer_norm, dropout=dec_dropout)
        self.project = nn.Linear(last_layer_dim, data_dim)

    def forward(self, inputs):
        latent = self.encoder(inputs)
        embedding = self.decoder(latent)
        final = self.project(embedding)


    @staticmethod
    def loss(data, labels, reduction=None, weights=None):
        return F.binary_cross_entropy(data, target, weight=weights, reduction=reduction)


