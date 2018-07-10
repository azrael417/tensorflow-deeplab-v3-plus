"""Converts PASCAL dataset to TFRecords file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import os
import sys
import h5py as h5
import numpy as np

import tensorflow as tf

from utils import dataset_util

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='./',
                    help='Path to the directory containing the data.')

parser.add_argument('--train_fraction', type=float, default=0.8,
                    help='Fraction of dataset used for training.')

parser.add_argument('--output_path', type=str, default='./dataset',
                    help='Path to the directory to create TFRecords outputs.')


def dict_to_tf_example(file_path, image_path, label_path):
  """Convert image and label to tf.Example proto.

  Args:
    file_path:  Path to HDF5 file
    image_path: Path to a image inside HDF5 file.
    label_path: Path to its corresponding label in HDF5 file.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the size of image does not match with that of label.
  """
  with h5.File(file_path) as fid:
    image = fid[image_path][...]
    label = fid[label_path][...]

  if np.prod(image.shape[1:]) != np.prod(label.shape):
    raise ValueError('The size of image does not match with that of label.')

  channels, width, height = image.shape
  
  print(channels, width, height)

  #example = tf.train.Example(features=tf.train.Features(feature={
  #  'image/height': dataset_util.int64_feature(height),
  #  'image/width': dataset_util.int64_feature(width),
  #  'image/encoded': dataset_util.bytes_feature(encoded_jpg),
  #  'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
  #  'label/encoded': dataset_util.bytes_feature(encoded_label),
  #  'label/format': dataset_util.bytes_feature('png'.encode('utf8')),
  #}))
  #return example


def create_tf_record(output_filename,
                     examples):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    examples: Examples to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, example in enumerate(examples):
    if idx % 500 == 0:
      tf.logging.info('On image %d of %d', idx, len(examples))
    image_path = os.path.join('climate', 'data')
    label_path = os.path.join('climate', 'labels')
    
    try:
      tf_example = dict_to_tf_example(example, image_path, label_path)
      #writer.write(tf_example.SerializeToString())
    except ValueError:
      tf.logging.warning('Invalid example: %s, ignoring.', example)

  writer.close()


def main(unused_argv):
  if not os.path.exists(FLAGS.output_path):
    os.makedirs(FLAGS.output_path)

  tf.logging.info("Reading from dataset")
  data_dir = FLAGS.data_dir

  #check if directories exist
  if not os.path.isdir(data_dir):
    raise ValueError("Cannot access data")
  
  #setup output paths
  train_output_path = os.path.join(FLAGS.output_path, 'train')
  if not os.path.isdir(train_output_path):
    os.makedirs(train_output_path)
  
  valid_output_path = os.path.join(FLAGS.output_path, 'valid')
  if not os.path.isdir(valid_output_path):
    os.makedirs(valid_output_path)
  
  #read the list of examples
  examples = [os.path.join(FLAGS.data_dir,x) for x in os.listdir(FLAGS.data_dir)]

  #split into test and train
  num_examples = len(examples)
  
  #shuffle:
  np.random.seed(12345)
  np.random.shuffle(examples)
  
  #split
  train_examples = examples[:int(num_examples*FLAGS.train_fraction)]
  valid_examples = examples[int(num_examples*FLAGS.train_fraction):]

  #create symlinks to split dataset from examples
  for train_example in train_examples:
    output_name = os.path.join(train_output_path, os.path.basename(train_example))
    if not os.path.isfile(output_name):
      os.symlink(train_example, output_name)
    
  for valid_example in valid_examples:
    output_name = os.path.join(valid_output_path, os.path.basename(valid_example))
    if not os.path.isfile(output_name):
      os.symlink(valid_example, output_name)

    
  #create_tf_record(valid_output_path, valid_examples)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
