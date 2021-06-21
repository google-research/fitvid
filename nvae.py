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

"""Flax implementation of NVAE like encoder decoder."""

# pylint:disable=g-bare-generic
# pytype: skip-file

import functools
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

ModuleDef = Any


class SEBlock(nn.Module):
  """Applies Squeeze-and-Excitation."""
  act: Callable = nn.relu
  axis: Tuple[int, int] = (1, 2)
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x):
    hidden_size = max(x.shape[-1] // 16, 4)
    y = x.mean(axis=self.axis, keepdims=True)
    y = nn.Dense(features=hidden_size, dtype=self.dtype, name='reduce')(y)
    y = self.act(y)
    y = nn.Dense(features=x.shape[-1], dtype=self.dtype, name='expand')(y)
    return nn.sigmoid(y) * x


class EncoderBlock(nn.Module):
  """NVAE ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  downsample: bool
  act: Callable = nn.swish

  @nn.compact
  def __call__(self, x):
    strides = (2, 2) if self.downsample else (1, 1)

    residual = x
    y = x
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), strides)(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    y = SEBlock()(y)

    if residual.shape != y.shape:
      print('E adjust')
      residual = self.conv(self.filters, (1, 1),
                           strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class DecoderBlock(nn.Module):
  """NVAE ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  upsample: bool
  expand: int = 4
  act: Callable = nn.swish

  def upsample_image(self, img, multiplier):
    shape = (img.shape[0],
             img.shape[1] * multiplier,
             img.shape[2] * multiplier,
             img.shape[3])
    return jax.image.resize(img, shape, jax.image.ResizeMethod.NEAREST)

  @nn.compact
  def __call__(self, x):
    if self.upsample:
      x = self.upsample_image(x, multiplier=2)

    residual = x
    y = x
    y = self.norm()(y)
    y = self.conv(self.filters * self.expand, (1, 1))(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters * self.expand, (5, 5))(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)
    y = SEBlock()(y)

    if residual.shape != y.shape:
      print('D adjust')
      residual = self.conv(self.filters, (1, 1), name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class ModularEncoder(nn.Module):
  """Modular Encoder."""
  training: bool
  stage_sizes: Sequence[int]
  encoder_block: Callable
  down_block: Callable
  num_classes: int
  num_filters: int = 64
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x):
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)
    norm = functools.partial(nn.BatchNorm,
                             use_running_average=not self.training,
                             momentum=0.9,
                             epsilon=1e-5,
                             axis_name='time',
                             dtype=self.dtype)

    skips = {}
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        print('E', i, j, x.shape)
        filters = self.num_filters * 2 ** i
        block = self.down_block if i > 0 and j == 0 else self.encoder_block
        x = block(filters=filters, conv=conv, norm=norm)(x)
        skips[(i, j)] = x

    print('E', i, j, x.shape)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    return x, skips


class ModularDecoder(nn.Module):
  """Modular Decoder."""
  training: bool
  skip_type: None
  stage_sizes: Sequence[int]
  decoder_block: Callable
  up_block: Callable
  first_block_shape: Sequence[int]
  num_filters: int = 64
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, skips):
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)
    norm = functools.partial(nn.BatchNorm,
                             use_running_average=not self.training,
                             momentum=0.9,
                             epsilon=1e-5,
                             axis_name='time',
                             dtype=self.dtype)

    filters = np.prod(np.array(self.first_block_shape))
    x = nn.Dense(filters, dtype=self.dtype)(x)
    x = jnp.reshape(x, (x.shape[0],) + self.first_block_shape)

    for i, block_size in enumerate(reversed(self.stage_sizes)):
      for j in range(block_size):
        print('D', i, j, x.shape)
        filters = self.num_filters * 2 ** (len(self.stage_sizes)-i-1)
        block = self.up_block if i > 0 and j == 0 else self.decoder_block
        x = block(filters=filters, conv=conv, norm=norm)(x)

        if self.skip_type == 'residual':
          x = x + skips[(len(self.stage_sizes) - i - 1, block_size - j - 1)]
        elif self.skip_type == 'concat':
          x = jnp.concatenate(
              [x, skips[(len(self.stage_sizes) - i - 1, block_size - j - 1)]],
              axis=-1)
        elif self.skip_type is not None:
          raise Exception('Unknown Skip Type.')

    print('D', i, j, x.shape)
    x = conv(3, (3, 3))(x)
    x = nn.sigmoid(x)
    x = jnp.asarray(x, self.dtype)
    return x

NVAE_ENCODER = functools.partial(
    ModularEncoder,
    encoder_block=functools.partial(EncoderBlock, downsample=False),
    down_block=functools.partial(EncoderBlock, downsample=True))

NVAE_DECODER = functools.partial(
    ModularDecoder,
    decoder_block=functools.partial(DecoderBlock, upsample=False),
    up_block=functools.partial(DecoderBlock, upsample=True))

NVAE_ENCODER_VMAP = nn.vmap(
    ModularEncoder,
    in_axes=1,
    out_axes=1,
    variable_axes={'params': None, 'batch_stats': None},
    split_rngs={'params': False, 'dropout': False, 'rng': False},
    axis_name='time')

NVAE_DECODER_VMAP = nn.vmap(
    ModularDecoder,
    in_axes=(1, None),
    out_axes=1,
    variable_axes={'params': None, 'batch_stats': None},
    split_rngs={'params': False, 'dropout': False, 'rng': False},
    axis_name='time')

NVAE_ENCODER_VIDEO = functools.partial(
    NVAE_ENCODER_VMAP,
    encoder_block=functools.partial(EncoderBlock, downsample=False),
    down_block=functools.partial(EncoderBlock, downsample=True))

NVAE_DECODER_VIDEO = functools.partial(
    NVAE_DECODER_VMAP,
    decoder_block=functools.partial(DecoderBlock, upsample=False),
    up_block=functools.partial(DecoderBlock, upsample=True))
