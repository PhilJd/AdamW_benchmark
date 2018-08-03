# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the CIFAR-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

import tensorflow as tf

import resnet
import weight_decay_helper

_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}


###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Returns a list of filenames."""
  data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

  assert os.path.exists(data_dir), (
      'Run cifar10_download_and_extract.py first to download and extract the '
      'CIFAR-10 data.')

  if is_training:
    return [
        os.path.join(data_dir, 'data_batch_%d.bin' % i)
        for i in range(1, _NUM_DATA_FILES + 1)
    ]
  else:
    return [os.path.join(data_dir, 'test_batch.bin')]


def parse_record(raw_record, is_training):
  """Parse CIFAR-10 image and label from a raw record."""
  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.decode_raw(raw_record, tf.uint8)

  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  label = tf.cast(record_vector[0], tf.int32)
  label = tf.one_hot(label, _NUM_CLASSES)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],
                           [_NUM_CHANNELS, _HEIGHT, _WIDTH])

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

  image = preprocess_image(image, is_training)

  return image, label


def preprocess_image(image, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, _HEIGHT + 8, _WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image


def input_fn(is_training, data_dir, batch_size, num_epochs=1,
             num_parallel_calls=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_parallel_calls: The number of records that are processed in parallel.
      This can be optimized per data set but for generally homogeneous data
      sets, should be approximately the number of available CPU cores.

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)

  return resnet.process_record_dataset(dataset, is_training, batch_size,
      _NUM_IMAGES['train'], parse_record, num_epochs, num_parallel_calls)


###############################################################################
# Running the model
###############################################################################
class Cifar10Model(resnet.Model):

  def __init__(self, resnet_size, data_format=None, num_classes=_NUM_CLASSES):
    """These are the parameters that work for CIFAR-10 data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
    """
    if resnet_size % 6 != 2:
      raise ValueError('resnet_size must be 6n + 2:', resnet_size)

    num_blocks = (resnet_size - 2) // 6

    super(Cifar10Model, self).__init__(
        resnet_size=resnet_size,
        num_classes=num_classes,
        num_filters=16,
        kernel_size=3,
        conv_stride=1,
        first_pool_size=None,
        first_pool_stride=None,
        second_pool_size=8,
        second_pool_stride=1,
        block_fn=resnet.building_block,
        block_sizes=[num_blocks] * 3,
        block_strides=[1, 2, 2],
        final_size=64,
        data_format=data_format)


def model_fn_factory(lr, weight_decay, optimizer_base):

  def cifar10_model_fn(features, labels, mode, params):
    """Model function for CIFAR-10."""
    features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])
    schedule_fn = resnet.schedule_with_warm_restarts(
        batch_size=params['batch_size'],
        num_images=_NUM_IMAGES['train'],
        t_0=100,
        m_mul=0.0)  # no restarts

    # Empirical testing showed that including batch_normalization variables
    # in the calculation of regularized loss helped validation accuracy
    # for the CIFAR-10 dataset, perhaps because the regularization prevents
    # overfitting on the small data set. We therefore include all vars when
    # regularizing and computing loss during training.
    def loss_filter_fn(name):
      return True
    return resnet.resnet_model_fn(features, labels, mode, Cifar10Model,
                                  resnet_size=params['resnet_size'],
                                  weight_decay=weight_decay,
                                  initial_learning_rate=lr,
                                  schedule_fn=schedule_fn,
                                  momentum=0.9,
                                  data_format=params['data_format'],
                                  loss_filter_fn=loss_filter_fn,
                                  decouple_weight_decay=FLAGS.decouple_weight_decay,
                                  optimizer_base=optimizer_base)
  return cifar10_model_fn


if __name__ == '__main__':
  #tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS = argparse.Namespace()
  FLAGS.resnet_size = 32
  FLAGS.train_epochs = 100
  FLAGS.epochs_per_eval = 10
  FLAGS.batch_size = 128
  FLAGS.data_dir = '/home/jundp/Data/cifar10'
  FLAGS.base_model_dir = '/home/jundp/Data/adamw/param_grid_warm_restarts/cifar10_'
  FLAGS.num_parallel_calls = 5
  FLAGS.data_format = 'channels_first'
  for opt_name, optimizer, decoupled in (('momentumw', tf.contrib.opt.MomentumWOptimizer, True),
                                         ('momentum', tf.train.MomentumOptimizer, False),
                                         ('adam', tf.train.AdamOptimizer, False),
                                         ('adamw', tf.contrib.opt.AdamWOptimizer, True)):
    FLAGS.decouple_weight_decay = decoupled
    for lr in [0.1, 0.01, 0.001, 1.0, 10.0]:
      for weight_decay in [0.00001, 0.0001, 0.001]:
        FLAGS.model_dir = (FLAGS.base_model_dir + opt_name + "lr_" +
                           str(lr) + "_weight_decay_" + str(weight_decay))
        if os.path.isdir(FLAGS.model_dir):
          continue  # don't train already trained models
        print(FLAGS.model_dir)
        model_fn = model_fn_factory(lr, weight_decay, optimizer)
        try:
            resnet.resnet_main(FLAGS, model_fn, input_fn)
        except tf.train.NanLossDuringTrainingError:
            print("Nan error.")

