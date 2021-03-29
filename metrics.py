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

"""Metrics."""

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_gan as tfgan
import tensorflow_hub as hub


i3d_model = None
lpips_model = None


def flatten_video(video):
  return np.reshape(video, (-1,) + video.shape[2:])


def psnr(video_1, video_2):
  video_1 = flatten_video(video_1)
  video_2 = flatten_video(video_2)
  dist = tf.image.psnr(video_1, video_2, max_val=1.0)
  return np.mean(dist.numpy())


def ssim(video_1, video_2):
  video_1 = flatten_video(video_1)
  video_2 = flatten_video(video_2)
  dist = tf.image.ssim(video_1, video_2, max_val=1.0)
  return np.mean(dist.numpy())


def psnr_image(target_image, out_image):
  dist = tf.image.psnr(target_image, out_image, max_val=1.0)
  return np.mean(dist.numpy())


def psnr_per_frame(target_video, out_video):
  max_val = 1.0
  mse = np.mean(np.square(out_video - target_video), axis=(2, 3, 4))
  return 20 * np.log10(max_val) - 10.0 * np.log10(mse)


def lpips_image(generated_image, real_image):
  global lpips_model
  result = 0.0
  return result


def lpips(video_1, video_2):
  video_1 = flatten_video(video_1)
  video_2 = flatten_video(video_2)
  dist = lpips_image(video_1, video_2)
  return np.mean(dist.numpy())


def fvd_preprocess(videos, target_resolution):
  videos = tf.convert_to_tensor(videos * 255.0, dtype=tf.float32)
  videos_shape = videos.shape.as_list()
  all_frames = tf.reshape(videos, [-1] + videos_shape[-3:])
  resized_videos = tf.image.resize(all_frames, size=target_resolution)
  target_shape = [videos_shape[0], -1] + list(target_resolution) + [3]
  output_videos = tf.reshape(resized_videos, target_shape)
  scaled_videos = 2. * tf.cast(output_videos, tf.float32) / 255. - 1
  return scaled_videos


def create_id3_embedding(videos):
  """Get id3 embeddings."""
  global i3d_model
  module_spec = 'https://tfhub.dev/deepmind/i3d-kinetics-400/1'

  if not i3d_model:
    base_model = hub.load(module_spec)
    input_tensor = base_model.graph.get_tensor_by_name('input_frames:0')
    i3d_model = base_model.prune(input_tensor, 'RGB/inception_i3d/Mean:0')

  output = i3d_model(videos)
  return output


def calculate_fvd(real_activations, generated_activations):
  return tfgan.eval.frechet_classifier_distance_from_activations(
      real_activations, generated_activations)


def fvd(video_1, video_2):
  video_1 = fvd_preprocess(video_1, (224, 224))
  video_2 = fvd_preprocess(video_2, (224, 224))
  x = create_id3_embedding(video_1)
  y = create_id3_embedding(video_2)
  result = calculate_fvd(x, y)
  return result.numpy()


def inception_score(images):
  return tfgan.eval.inception_score(images)

