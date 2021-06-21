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

"""Data."""

import functools
import random

from fitvid.randaug import randaugment
from flax import jax_utils
import jax
import numpy as np
import tensorflow as tf  # tf
import tensorflow_datasets as tfds


def rand_crop(seeds, video, width, height, wiggle):
  """Random crop of a video. Assuming height < width."""
  x_wiggle = wiggle
  crop_width = height - wiggle
  y_wiggle = width - crop_width
  xx = tf.random.stateless_uniform(
      [], seed=seeds[0], minval=0, maxval=x_wiggle, dtype=tf.int32)
  yy = tf.random.stateless_uniform(
      [], seed=seeds[1], minval=0, maxval=y_wiggle, dtype=tf.int32)
  return video[:, xx:xx+crop_width, yy:yy+crop_width, :]


def rand_aug(seeds, video, num_layers, magnitude):
  """RandAug for video with the same random seed for all frames."""
  image_aug = lambda a, x: randaugment(x, num_layers, magnitude, seeds)
  return tf.scan(image_aug, video)


def augment_dataset(dataset, augmentations):
  """Augment dataset with a list of augmentations."""
  def augment(seeds, features):
    video = tf.cast(features['video'], tf.uint8)
    for aug_fn in augmentations:
      video = aug_fn(seeds, video)
    video = tf.image.resize(video, (64, 64), antialias=True)
    features['video'] = video
    return features

  randds = tf.data.experimental.RandomDataset(1).batch(2).batch(4)
  dataset = tf.data.Dataset.zip((randds, dataset))
  dataset = dataset.map(
      augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset


def normalize_video(features):
  features['video'] = tf.cast(features['video'], tf.float32) / 255.0
  return features


def get_iterator(dataset, batch_size, is_train):
  """"Returns a performance optimized iterator from dataset."""
  local_device_count = jax.local_device_count()
  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  options.experimental_threading.max_intra_op_parallelism = 1
  dataset = dataset.with_options(options)
  dataset = dataset.map(normalize_video)
  dataset = dataset.repeat()
  if is_train:
    dataset = dataset.shuffle(batch_size * 64, seed=0)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(32)

  def prepare_tf_data(xs):
    def _prepare(x):
      x = x._numpy()
      return x.reshape((local_device_count, -1) + x.shape[1:])
    return jax.tree_map(_prepare, xs)

  iterator = map(prepare_tf_data, dataset)
  iterator = jax_utils.prefetch_to_device(iterator, 2)
  return iterator




def load_dataset_robonet(batch_size, video_len, is_train):
  """"Load RoboNet dataset."""

  def extract_features_robonet(features):
    dtype = tf.float32
    video = tf.cast(features['video'], dtype)
    actions = tf.cast(features['actions'], dtype)
    video /= 255.0
    return {
        'video': tf.identity(video[:video_len]),
        'actions': tf.identity(actions[:video_len-1]),
    }

  def robonet_filter_by_filename(features, filenames, white=True):
    in_list = tf.reduce_any(tf.math.equal(features['filename'], filenames))
    return in_list if white else tf.math.logical_not(in_list)

  def get_robonet_test_filenames():
    testfiles = None
    if testfiles is None:
      with tf.io.gfile.GFile('robonet_testset_filenames.txt', 'r') as f:
        testfiles = f.read()
    testfiles = ([x.encode('ascii') for x in testfiles.split('\n') if x])
    return testfiles

  dataset_builder = tfds.builder('robonet/robonet_64')
  num_examples = dataset_builder.info.splits['train'].num_examples
  split_size = num_examples // jax.host_count()
  start = jax.host_id() * split_size
  split = 'train[{}:{}]'.format(start, start + split_size)
  dataset = dataset_builder.as_dataset(split=split)
  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  options.experimental_threading.max_intra_op_parallelism = 1
  dataset = dataset.with_options(options)

  test_filenames = get_robonet_test_filenames()
  train_filter = functools.partial(
      robonet_filter_by_filename, filenames=test_filenames, white=not is_train)

  dataset = dataset.filter(train_filter)
  dataset = dataset.map(
      extract_features_robonet,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return get_iterator(dataset, batch_size, is_train)
