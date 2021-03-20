# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Models."""

# pytype: skip-file

import functools

from flax import linen as nn
from flax.nn import initializers
import jax
import jax.numpy as jnp

from jaxido import nvae
from jaxido import utils


class MultiGaussianLSTM(nn.Module):
  """Multi layer lstm with Gaussian output."""
  num_layers: int = 2
  hidden_size: int = 10
  output_size: int = 10
  dtype: int = jnp.float32

  def setup(self):
    self.embed = nn.Dense(self.hidden_size)
    self.mean = nn.Dense(self.output_size)
    self.logvar = nn.Dense(self.output_size)
    self.layers = [nn.recurrent.LSTMCell() for _ in range(self.num_layers)]

  def init_states(self, batch_size):
    init_fn = functools.partial(initializers.zeros, dtype=self.dtype)
    states = [None] * self.num_layers
    for i in range(self.num_layers):
      states[i] = nn.recurrent.LSTMCell.initialize_carry(
          self.make_rng('rng'),
          (batch_size,),
          self.hidden_size,
          init_fn=init_fn)
    return states

  def reparameterize(self, mu, logvar):
    var = jnp.exp(0.5 * logvar)
    epsilon = jax.random.normal(self.make_rng('rng'), var.shape)
    return mu + var * epsilon

  def __call__(self, x, states):
    x = self.embed(x)
    for i in range(self.num_layers):
      states[i], x = self.layers[i](states[i], x)
    mean = self.mean(x)
    logvar = self.logvar(x)
    z = self.reparameterize(mean, logvar)
    return states, (z, mean, logvar)


class EncDec(nn.Module):
  """Simple Encoder/Decoder video predictor."""
  training: bool
  stochastic: bool = True
  action_conditioned: bool = True
  learned_prior: bool = True
  z_dim: int = 10
  g_dim: int = 128
  rnn_size: int = 256
  n_past: int = 2
  beta: float = 1e-4
  dtype: int = jnp.float32

  def setup(self):
    self.encoder = nvae.NVAE_ENCODER(
        training=self.training,
        stage_sizes=[2, 2, 2, 2],
        num_classes=self.g_dim)
    self.decoder = nvae.NVAE_DECODER(
        training=self.training,
        stage_sizes=[2, 2, 2, 2],
        first_block_shape=(8, 8, 512),
        skip_type='residual')
    self.frame_predictor = MultiGaussianLSTM(
        hidden_size=self.rnn_size, output_size=self.g_dim, num_layers=2)
    self.posterior = MultiGaussianLSTM(
        hidden_size=self.rnn_size, output_size=self.z_dim, num_layers=1)
    self.prior = MultiGaussianLSTM(
        hidden_size=self.rnn_size, output_size=self.z_dim, num_layers=1)

  def __call__(self, video, actions, step):
    batch_size, video_len = video.shape[0], video.shape[1]
    frame_predictor_states = self.frame_predictor.init_states(batch_size)
    posterior_states = self.posterior.init_states(batch_size)
    prior_states = self.prior.init_states(batch_size)

    h_seq = [self.encoder(video[:, i]) for i in range(video_len)]

    mse = 0.0
    kld = 0.0
    preds, means, logvars = [], [], []
    x_pred = video[:, 0]
    _, skip = h_seq[0]
    for i in range(1, video_len):
      h_target = h_seq[i][0]
      h, s = h_seq[i-1]
      if self.training and i <= self.n_past:
        skip = s
      if not self.training and i > self.n_past:
        h, _ = self.encoder(x_pred)
      posterior_states, (z_t, mu, logvar) = self.posterior(
          h_target, posterior_states)

      if self.learned_prior:
        prior_states, (prior_z_t, prior_mu, prior_logvar) = self.prior(
            h, prior_states)
      else:
        prior_mu, prior_logvar = jnp.zeros_like(mu), jnp.zeros_like(logvar)
        prior_z_t = jax.random.normal(self.make_rng('rng'), z_t.shape)
      if not self.training:
        z_t = prior_z_t

      inp = [h]
      if self.action_conditioned:
        inp += [actions[:, i-1]]
      if self.stochastic:
        inp += [z_t]
      inp = jnp.concatenate(inp, axis=1)

      frame_predictor_states, (_, h_pred, _) = self.frame_predictor(
          inp, frame_predictor_states)
      h_pred = nn.sigmoid(h_pred)
      x_pred = self.decoder(h_pred, skip)
      mse += utils.l2_loss(x_pred, video[:, i])
      kld += utils.kl_divergence(
          mu, logvar, prior_mu, prior_logvar, batch_size)

      preds.append(x_pred)
      means.append(mu)
      logvars.append(logvar)

    preds = jnp.stack(preds, axis=1)
    means = jnp.stack(means, axis=1)
    logvars = jnp.stack(logvars, axis=1)

    loss = mse + kld * self.beta

    # Train Metrics
    metrics = {
        'hist/mean': means,
        'hist/logvars': logvars,
        'loss/mse': mse,
        'loss/kld': kld,
        'loss/all': loss,
    }

    return loss, preds, metrics
