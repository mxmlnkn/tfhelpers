# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

# Lint as: python3
"""Imagenet datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import re

import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds


_DESCRIPTION = '''\
In contrast to the canonical tensorflow datasets' Imagenet2012, this allows
extracted folders as input if the TAR is not available or if the TAR is
already mounted with ratarmount.

ILSVRC 2012, aka ImageNet is an image dataset organized according to the
WordNet hierarchy. Each meaningful concept in WordNet, possibly described by
multiple words or word phrases, is called a "synonym set" or "synset". There are
more than 100,000 synsets in WordNet, majority of them are nouns (80,000+). In
ImageNet, we aim to provide on average 1000 images to illustrate each synset.
Images of each concept are quality-controlled and human-annotated. In its
completion, we hope ImageNet will offer tens of millions of cleanly sorted
images for most of the concepts in the WordNet hierarchy.

Note that labels were never publicly released for the test set, so we only
include splits for the training and validation sets here.
'''

# Web-site is asking to cite paper from 2015.
# http://www.image-net.org/challenges/LSVRC/2012/index#cite
_CITATION = '''\
@article{ILSVRC15,
Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
Title = {{ImageNet Large Scale Visual Recognition Challenge}},
Year = {2015},
journal   = {International Journal of Computer Vision (IJCV)},
doi = {10.1007/s11263-015-0816-y},
volume={115},
number={3},
pages={211-252}
}
'''

_LABELS_FNAME = 'image_classification/imagenet2012_labels.txt'

# This file contains the validation labels, in the alphabetic order of
# corresponding image names (and not in the order they have been added to the
# tar file).
_VALIDATION_LABELS_FNAME = 'image_classification/imagenet2012_validation_labels.txt'


# From https://github.com/cytsai/ilsvrc-cmyk-image-list
CMYK_IMAGES = [
    'n01739381_1309.JPEG',
    'n02077923_14822.JPEG',
    'n02447366_23489.JPEG',
    'n02492035_15739.JPEG',
    'n02747177_10752.JPEG',
    'n03018349_4028.JPEG',
    'n03062245_4620.JPEG',
    'n03347037_9675.JPEG',
    'n03467068_12171.JPEG',
    'n03529860_11437.JPEG',
    'n03544143_17228.JPEG',
    'n03633091_5218.JPEG',
    'n03710637_5125.JPEG',
    'n03961711_5286.JPEG',
    'n04033995_2932.JPEG',
    'n04258138_17003.JPEG',
    'n04264628_27969.JPEG',
    'n04336792_7448.JPEG',
    'n04371774_5854.JPEG',
    'n04596742_4225.JPEG',
    'n07583066_647.JPEG',
    'n13037406_4650.JPEG',
]

PNG_IMAGES = ['n02105855_2933.JPEG']


class Imagenet2012Folder(tfds.core.GeneratorBasedBuilder):
  """Imagenet 2012, aka ILSVRC 2012."""

  VERSION = tfds.core.Version(
      '1.0.0', 'New split API (https://tensorflow.org/datasets/splits)')

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  Download the dataset from: https://www.kaggle.com/c/imagenet-object-localization-challenge/data
  E.g. by using: kaggle competitions download -c imagenet-object-localization-challenge

  What you get is a zip with this structure:
    imagenet-object-localization-challenge.zip
      LOC_sample_submission.csv
      LOC_synset_mapping.txt
      LOC_train_solution.csv
      LOC_val_solution.csv
      imagenet_object_localization_patched2019.tar.gz
        ILSVRC
          Annotations
            CLS-LOC
              val
                ILSVRC2012_val_00025981.xml
                ILSVRC2012_val_00048606.xml
                ILSVRC2012_val_00025198.xml
                ...
              train
                n02276258
                  n02276258_3025.xml
                  n02276258_7567.xml
                  n02276258_4818.xml
                  ...
                n02412080
                  n02412080_16229.xml
                  n02412080_61170.xml
                  ..
                ..
          ImageSets
            CLS-LOC
              val.txt
              train_cls.txt
              test.txt
              train_loc.txt

          Data
            CLS-LOC
              val
                ILSVRC2012_val_00025981.JPEG
                ILSVRC2012_val_00048606.JPEG
                ILSVRC2012_val_00025198.JPEG
                ...
              train
                n02276258
                  n02276258_3025.JPEG
                  n02276258_7567.JPEG
                  n02276258_4818.JPEG
                  ...
                n02412080
                  n02412080_16229.JPEG
                  n02412080_61170.JPEG
                  ..

  In this case manual_dir should point to ILSVRC/Data/CLS-LOC containing the val and train subfolders.
  """

  def _info(self):
    names_file = tfds.core.get_tfds_path(_LABELS_FNAME)
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(encoding_format='jpeg'),
            'label': tfds.features.ClassLabel(names_file=names_file),
            'file_name': tfds.features.Text(),  # Eg: 'n15075141_54.JPEG'
        }),
        supervised_keys=('image', 'label'),
        homepage='http://image-net.org/',
        citation=_CITATION,
    )

  @staticmethod
  def _get_validation_labels(val_path):
    """Returns labels for validation.

    Args:
      val_path: path to folder containing validation images. It is used to
      retrieve the name of pictures and associate them to labels.

    Returns:
      dict, mapping from image name (str) to label (str).
    """
    labels_path = tfds.core.get_tfds_path(_VALIDATION_LABELS_FNAME)
    with tf.io.gfile.GFile(labels_path) as labels_f:
      # `splitlines` to remove trailing `\r` in Windows
      labels = labels_f.read().strip().splitlines()
    images = sorted((name for name in tf.io.gfile.listdir(val_path) if name.lower().endswith('.jpeg')))
    return dict(zip(images, labels))

  def _split_generators(self, dl_manager):
    train_path = os.path.join(dl_manager.manual_dir, 'train')
    val_path = os.path.join(dl_manager.manual_dir, 'val')
    # We don't import the original test split, as it doesn't include labels.
    # These were never publicly released.
    if not tf.io.gfile.exists(train_path) or not tf.io.gfile.exists(val_path):
      raise AssertionError(
          'ImageNet requires manual download of the data. Please download '
          'the train and val set and place them into: {}, {}'.format(
              train_path, val_path))
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                'folder': train_path,
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                'folder': val_path,
                'validation_labels': self._get_validation_labels(val_path),
            },
        ),
    ]

  def _fix_image(self, image_fname, image):
    if image_fname in CMYK_IMAGES:
      image = io.BytesIO(tfds.core.utils.jpeg_cmyk_to_rgb(image.read()))
    elif image_fname in PNG_IMAGES:
      image = io.BytesIO(tfds.core.utils.png_to_jpeg(image.read()))
    return image

  def _generate_examples(self, folder, validation_labels=None):
    """Yields examples."""
    if validation_labels:  # Validation split
      for key, example in self._generate_examples_validation(folder,
                                                             validation_labels):
        yield key, example
        return

    # Training split. Main archive contains archives names after a synset noun.
    # Each subfolder contains pictures associated to that synset, something like 'n01632458'.
    for fname in tf.io.gfile.listdir(folder):
      synset_folder = os.path.join(folder, fname)
      assert tf.io.gfile.isdir(synset_folder)
      assert re.fullmatch( 'n[0-9]{8}', fname )

      for image_fname in tf.io.gfile.listdir(synset_folder):
        image = tf.io.gfile.GFile(os.path.join(synset_folder, image_fname), 'rb')
        image = self._fix_image(image_fname, image)
        record = {
            'file_name': image_fname,
            'image': image,
            'label': fname,
        }
        yield image_fname, record

  def _generate_examples_validation(self, folder, labels):
    for fname in tf.io.gfile.listdir(folder):
      fobj = tf.io.gfile.GFile(os.path.join(folder, fname), 'rb')
      record = {
          'file_name': fname,
          'image': fobj,
          'label': labels[fname],
      }
      yield fname, record
