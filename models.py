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

from fitvid import nvae
from fitvid import utils


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




class FitVid(nn.Module):
  """FitVid video predictor."""
  training: bool
  stochastic: bool = True
  action_conditioned: bool = True
  z_dim: int = 10
  g_dim: int = 128
  rnn_size: int = 256
  n_past: int = 2
  beta: float = 1e-4
  dtype: int = jnp.float32

  def setup(self):
    self.encoder = nvae.NVAE_ENCODER_VIDEO(
        training=self.training,
        stage_sizes=[2, 2, 2, 2],
        num_classes=self.g_dim)
    self.decoder = nvae.NVAE_DECODER_VIDEO(
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

  def get_input(self, hidden, action, z):
    inp = [hidden]
    if self.action_conditioned:
      inp += [action]
    if self.stochastic:
      inp += [z]
    return jnp.concatenate(inp, axis=1)

  def __call__(self, video, actions, step):
    batch_size, video_len = video.shape[0], video.shape[1]
    pred_s = self.frame_predictor.init_states(batch_size)
    post_s = self.posterior.init_states(batch_size)
    prior_s = self.prior.init_states(batch_size)
    kl = functools.partial(utils.kl_divergence, batch_size=batch_size)

    # encode frames
    hidden, skips = self.encoder(video)
    # Keep the last available skip only
    skips = {k: skips[k][:, self.n_past-1] for k in skips.keys()}

    kld, means, logvars = 0.0, [], []
    if self.training:
      h_preds = []
      for i in range(1, video_len):
        h, h_target = hidden[:, i-1], hidden[:, i]
        post_s, (z_t, mu, logvar) = self.posterior(h_target, post_s)
        prior_s, (_, prior_mu, prior_logvar) = self.prior(h, prior_s)

        inp = self.get_input(h, actions[:, i-1], z_t)
        pred_s, (_, h_pred, _) = self.frame_predictor(inp, pred_s)
        h_pred = nn.sigmoid(h_pred)
        h_preds.append(h_pred)
        means.append(mu)
        logvars.append(logvar)
        kld += kl(mu, logvar, prior_mu, prior_logvar)

      h_preds = jnp.stack(h_preds, axis=1)
      preds = self.decoder(h_preds, skips)

    else:  # eval
      preds, x_pred = [], None
      for i in range(1, video_len):
        h, h_target = hidden[:, i-1], hidden[:, i]
        if i > self.n_past:
          h = self.encoder(jnp.expand_dims(x_pred, 1))[0][:, 0]

        post_s, (_, mu, logvar) = self.posterior(h_target, post_s)
        prior_s, (z_t, prior_mu, prior_logvar) = self.prior(h, prior_s)

        inp = self.get_input(h, actions[:, i-1], z_t)
        pred_s, (_, h_pred, _) = self.frame_predictor(inp, pred_s)
        h_pred = nn.sigmoid(h_pred)
        x_pred = self.decoder(jnp.expand_dims(h_pred, 1), skips)[:, 0]
        preds.append(x_pred)
        means.append(mu)
        logvars.append(logvar)
        kld += kl(mu, logvar, prior_mu, prior_logvar)

      preds = jnp.stack(preds, axis=1)

    means = jnp.stack(means, axis=1)
    logvars = jnp.stack(logvars, axis=1)
    mse = utils.l2_loss(preds, video[:, 1:])
    loss = mse + kld * self.beta

    # Metrics
    metrics = {
        'hist/mean': means,
        'hist/logvars': logvars,
        'loss/mse': mse,
        'loss/kld': kld,
        'loss/all': loss,
    }

    return loss, preds, metrics

