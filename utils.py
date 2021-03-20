# Copyright 2021 Google LLC
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

"""Utils."""

from typing import Any

from absl import logging
from flax import linen as nn
from flax import optim
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v2 as tf

from tensorboard.plugins.image import metadata


@struct.dataclass
class TrainState:
  step: int
  optimizer: optim.Optimizer
  model_state: Any


def flatten(x):
  return jnp.reshape(x, (x.shape[0], -1))


def discretize(x):
  z = nn.tanh(x)
  d = jnp.asarray(jnp.less(0.0, z), dtype=jnp.float32)
  return z + lax.stop_gradient(2.0 * d - 1.0 - z)


def kl_divergence(mean1, logvar1, mean2, logvar2, batch_size):
  kld = 0.5 * (-1.0 + logvar2 - logvar1 + jnp.exp(logvar1 - logvar2)
               + jnp.square(mean1 - mean2) * jnp.exp(-logvar2))
  return jnp.sum(kld) / batch_size


def l2_norm(tree):
  """Compute the l2 norm of a pytree of arrays."""
  leaves, _ = jax.tree_flatten(tree)
  return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))


def clip_grads(grad_tree, max_norm):
  """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
  norm = l2_norm(grad_tree)
  normalize = lambda g: jnp.where(norm < max_norm, g, g * (max_norm / norm))
  return jax.tree_map(normalize, grad_tree)


def image_float_to_uint(image):
  image = np.clip(image, 0.0, 1.0)
  image = (255 * image).astype(np.uint8)
  return image


def generate_rng_dict(base_rng):
  keys = ("params", "dropout", "rng")
  rngs = jax.random.split(base_rng, len(keys))
  return {key: rngs[i] for i, key in enumerate(keys)}


def cross_entropy_loss(logits, labels):
  labels = jax.nn.one_hot(labels, logits.shape[-1])
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  x_entropy = -jnp.sum(log_probs * labels, axis=-1)
  return jnp.mean(x_entropy)


def l1_loss(model_logits, ground_truth):
  return jnp.mean(jnp.absolute(model_logits - ground_truth))


def l2_loss(model_logits, ground_truth):
  return jnp.mean(jnp.square(model_logits - ground_truth))


def _sync_batch_stats(x):
  return lax.pmean(x, "batch")


def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  if "batch_stats" not in state.model_state.keys():
    return state

  avg = jax.pmap(_sync_batch_stats, "batch")
  new_model_state = state.model_state.copy({
      "batch_stats": avg(state.model_state["batch_stats"])})
  return state.replace(model_state=new_model_state)


def nested_dict_path_print(d):
  def pretty_dict(x, path):
    if not isinstance(x, dict):
      return f"{path}: {repr(x)}\n"
    rep = ""
    for key, val in x.items():
      rep += pretty_dict(val, f"{path}/{key}")
    return rep
  return pretty_dict(d, "")


def print_model_size(params, name=""):
  model_params_size = jax.tree_map(lambda x: x.size, params)
  total_params_size = sum(jax.tree_flatten(model_params_size)[0])
  logging.info("#" * 30)
  logging.info("###### %s Parameters: %d", name, total_params_size)
  logging.info("#" * 30)
  logging.info(nested_dict_path_print(model_params_size._dict))


def get_average_across_devices(x):
  x = jax.tree_map(lambda a: a.mean(axis=0), x)
  return jax.device_get(x)


def get_first_device(x):
  x = jax.tree_map(lambda a: a[0], x)
  return jax.device_get(x)


def get_all_devices(x):
  x = jax.tree_map(lambda a: jnp.reshape(a, (-1,) + a.shape[2:]), x)
  return jax.device_get(x)


def encode_gif(video, fps):
  """Encode a video into gif."""
  import subprocess
  l, h, w, c = video.shape
  ffmpeg = "ffmpeg"
  cmd = [
      ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-r",
      "%.02f" % fps, "-s",
      "%dx%d" % (w, h), "-pix_fmt", {
          1: "gray",
          3: "rgb24"
      }[c], "-i", "-", "-filter_complex",
      "[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse", "-r",
      "%.02f" % fps, "-f", "gif", "-"
  ]
  proc = subprocess.Popen(
      cmd,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE)
  for i in range(l):
    proc.stdin.write(video[i].tostring())
  out, err = proc.communicate()
  if proc.returncode:
    raise IOError("\n".join([" ".join(cmd), err.decode("utf8")]))
  del proc
  return out


def write_video_summaries(summary_writer, video_batch, num_samples, step):
  """Writes a video summary in gif and side by side images format."""
  video_batch = image_float_to_uint(video_batch)
  _, video_len, w, h, c = tuple(video_batch.shape)
  for i in range(num_samples):
    video = video_batch[i]
    summary = encode_gif(video, fps=15)
    tensor = tf.concat(
        [tf.as_string(w), tf.as_string(h), tf.convert_to_tensor(summary)],
        axis=0)
    md = metadata.create_summary_metadata(
        display_name=None, description=None)
    summary_writer.write(tag="gif_%d"%i, tensor=tensor, step=step, metadata=md)
    mo = np.reshape(video, [w * video_len, h, c])
    summary_writer.image(tag="sidebyside_%d"%i, image=mo, step=step)


def scheduler(
    factors="constant * linear_warmup * rsqrt_decay",
    base_learning_rate=0.5,
    warmup_steps=8000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000):
  """creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    factors: a string with factors separated by '*' that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: how many steps to warm up for in the warmup schedule.
    decay_factor: The amount to decay the learning rate by.
    steps_per_decay: How often to decay the learning rate.
    steps_per_cycle: Steps per cycle when using cosine decay.

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split("*")]

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == "constant":
        ret *= base_learning_rate
      elif name == "linear_warmup":
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == "rsqrt_decay":
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == "rsqrt_normalized_decay":
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == "decay_every":
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == "cosine_decay":
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError("Unknown factor %s." % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


def plot_1d_signals(signals, labels):
  """Plot a 1d signals and converts into an image."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  for x, l in zip(signals, labels):
    ax.plot(x, "--o", label=l)
  ax.legend()
  fig.canvas.draw()
  image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
  image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  return image
