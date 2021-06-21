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

"""Eval binary."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
from flax import jax_utils
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from fitvid import utils
from fitvid.train import evaluate
from fitvid.train import get_data
from fitvid.train import init_model_state
from fitvid.train import MODEL_CLS
from fitvid.train import write_summaries
import numpy as np
import tensorflow.compat.v2 as tf


FLAGS = flags.FLAGS


def eval_model():
  """Evaluates the latest model checkpoint."""
  rng_key = jax.random.PRNGKey(0)

  log_dir = os.path.join(FLAGS.output_dir, 'evaluate')
  summary_writer = tensorboard.SummaryWriter(log_dir)

  data_itr = get_data(False)
  batch = next(data_itr)
  sample = utils.get_first_device(batch)

  model = MODEL_CLS(n_past=FLAGS.n_past, training=False)
  state = init_model_state(rng_key, model, sample)

  last_checkpoint = int(state.step)
  rng_key = jax.random.split(rng_key, jax.local_device_count())
  while True:
    state = checkpoints.restore_checkpoint(FLAGS.output_dir, state)
    if int(state.step) == last_checkpoint:
      time.sleep(60)  # sleep for 60 secs
    else:
      last_checkpoint = int(state.step)
      state = jax_utils.replicate(state)
      metrics, gt, out_vid = evaluate(
          rng_key, state, model, data_itr, eval_steps=256 // FLAGS.batch_size)
      if jax.host_id() == 0:
        write_summaries(summary_writer, metrics, last_checkpoint, out_vid, gt)
        video_summary = np.concatenate([gt, out_vid], axis=3)
        with tf.io.gfile.GFile(f'{log_dir}/video.npy', 'w') as outfile:
          np.save(outfile, video_summary)


def main(argv):
  del argv  # Unused
  tf.enable_v2_behavior()
  eval_model()
  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()


if __name__ == '__main__':
  # We assume that checkpoints are in the output_dir
  flags.mark_flags_as_required(['output_dir'])
  app.run(main)
