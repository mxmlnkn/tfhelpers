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

# https://www.tensorflow.org/datasets/add_dataset

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import gzip
import os
import tarfile
import tempfile

import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds


_DESCRIPTION = '''\
ILSVRC 2010 is an image dataset organized according to the
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
# http://www.image-net.org/challenges/LSVRC/2010/index#cite
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

# The ID to label mapping can be extracted from the ILSVRC2010_devkit-1.0.tar.gz,
# which contains data/meta.mat, using:
#   python3 -c '
#   import scipy.io
#   meta = scipy.io.loadmat( "meta.mat" )
#   with open( "ILSVRC2010_labels.txt", "wt" ) as file:
#     for synset in meta["synsets"]:
#       file.write( "{},{}\n".format( synset[0][0][0][0], synset[0][1][0] ) )
#   '
_LABELS_FNAME = 'ILSVRC2010_labels.txt'

# This file contains the validation labels, in the alphabetic order of
# corresponding image names (and not in the order they have been added to the
# tar file).
_VALIDATION_LABELS_FNAME = 'ILSVRC2010_validation_ground_truth.txt'
_TEST_LABELS_FNAME = 'ILSVRC2010_test_ground_truth.txt'


# These lists can be easily calculated using find, xargs, ratarmount, and ImageMagick's identify or jpeginfo.
# Note that identify takes multiple arguments and that should be made use of by using xargs' -n option
# because starting identify up takes a lot of time by itself.
# If your tar is on an SSD, prefer -P $( nprocs ) to speed up processing.
# If your tar is on an HDD, prefer -P 1 to avoid too much seeking
#   ratarmount ILSVRC2010_images_{train,val,test}.tar patch_images.tar /tmp/ILSVRC2010
#   time nFiles=$( find /media/ILSVRC2010/ -type f | wc -l ) # 90s
#   time find /media/ILSVRC2010/ -type f |
#       pv --line-mode --size $nFiles |
#       xargs -I{} -P 1 -n 100 jpeginfo -c {} 2>/dev/null |
#       'grep' -v OK > jpeginfo.log # 30min

# jpeginfo message: Corrupt JPEG data: 1 extraneous bytes before marker 0xd9  [WARNING]
#   -> only a warning. shouldn't be so bad
EXTRANEOUS_BYTES_IMAGES = [
    'n01669191_10054.JPEG',
    'n02882647_18811.JPEG',
]
# jpeginfo message: Not a JPEG file: starts with 0x1f 0x8b  [ERROR]
#   -> This is a gz compressed jpeg ... -.-
JPEG_GZ_IMAGES = [
    'n02487347_1956.JPEG',
]
PNG_IMAGES = [] # None found. Seems to only be an issue in ILSVRC2012
# jpeginfo message: Unsupported color conversion request  [ERROR]
CMYK_IMAGES = [
    'ILSVRC2010_test_00099846.JPEG',
    'ILSVRC2010_test_00149181.JPEG',
    'ILSVRC2010_val_00014712.JPEG',
    'ILSVRC2010_val_00040086.JPEG',
    'ILSVRC2010_val_00041418.JPEG',
    'n01739381_1309.JPEG',
    'n02747177_10752.JPEG',
    'n02872752_3721.JPEG',
    'n03109150_15092.JPEG',
    'n03329302_553.JPEG',
    'n03347037_9675.JPEG',
    'n03359436_9884.JPEG',
    'n03633091_5218.JPEG',
    'n03710637_5125.JPEG',
    'n04033995_2932.JPEG',
    'n04453156_5048.JPEG',
    'n04517823_2387.JPEG',
    'n04596742_4225.JPEG',
]


def script_folder():
    return os.path.join( os.getcwd(), os.path.dirname( __file__ ) )

class Imagenet2010(tfds.core.GeneratorBasedBuilder):
  """ILSVRC 2010"""

  VERSION = tfds.core.Version('1.0.0')

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  manual_dir should contain these files:
      ILSVRC2010_images_train.tar
      ILSVRC2010_images_val.tar
      ILSVRC2010_images_test.tar
      patch_images.tar
  In the past this dataset was downloadable from:
      http://www.image-net.org/challenges/LSVRC/2010/download-all-nonpub
      http://www.image-net.org/challenges/LSVRC/2010/ILSVRC2010_test_ground_truth.txt
      The validation ground truth can be found in the ILSVRC2010_devkit-1.0.tar.gz.
      The mapping between synsets and ground truth IDs can be found in ILSVRC2010_devkit-1.0/data/meta.mat
  """

  def _info(self):
    labels = np.loadtxt( os.path.join( script_folder(), 'ILSVRC2010_labels.txt' ),
                         delimiter = ',', dtype = str, encoding = 'utf-8' )
    self._idToSynset = dict( zip( labels[:,0].astype( int ), labels[:,1] ) )
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(encoding_format='jpeg'),
            'label': tfds.features.ClassLabel(names=list(self._idToSynset.values())),
            'file_name': tfds.features.Text(),  # Eg: 'n15075141_54.JPEG'
        }),
        supervised_keys=('image', 'label'),
        homepage='http://image-net.org/',
        citation=_CITATION,
    )

  def _get_labels(self, archive_path, labels_path):
    """Returns labels for validation and test.
    Args:
      archive_path: path to TAR file containing validation images. It is used to
      retrieve the name of pictures and associate them to labels.
    Returns:
      dict, mapping from image name (str) to label (str).
    """
    with tf.io.gfile.GFile(labels_path) as labels_f:
      # `splitlines` to remove trailing `\r` in Windows
      labels = [ self._idToSynset[int(x)] for x in labels_f.read().strip().splitlines() ]
    with tf.io.gfile.GFile(archive_path, 'rb') as tar_f_obj:
      tar = tarfile.open(mode='r:', fileobj=tar_f_obj)
      images = sorted([ tarinfo.name for tarinfo in tar if tarinfo.type == tarfile.REGTYPE])
    assert len(images) == len(labels)
    return dict(zip(images, labels))

  def _split_generators(self, dl_manager):
    train_path = os.path.join(dl_manager.manual_dir, 'ILSVRC2010_images_train.tar')
    test_path = os.path.join(dl_manager.manual_dir, 'ILSVRC2010_images_test.tar')
    val_path = os.path.join(dl_manager.manual_dir, 'ILSVRC2010_images_val.tar')
    patch_path = os.path.join(dl_manager.manual_dir, 'patch_images.tar')

    if tf.io.gfile.exists(patch_path):
        # Extract patches so all into one folder so that it is easier to look them up by name
        self.patch_images_folder_temp_object = tempfile.TemporaryDirectory()
        self.patch_images_folder = self.patch_images_folder_temp_object.name
        tar = tarfile.open(patch_path)
        for tarInfo in tar:
            image = tar.extractfile( tarInfo )
            if image:
                with open(os.path.join(self.patch_images_folder,
                                       os.path.basename(tarInfo.name)), 'wb') as file:
                    file.write( image.read() )
    else:
        self.patch_images_folder = None

    if not tf.io.gfile.exists(train_path) or \
       not tf.io.gfile.exists(test_path) or \
       not tf.io.gfile.exists(val_path):
      raise AssertionError(
          'ImageNet requires manual download of the data. Please download '
          'the train, val, and test set and place them into: {}, {}'.format(
              train_path, val_path))
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                'archive': dl_manager.iter_archive(train_path),
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                'archive': dl_manager.iter_archive(test_path),
                'labels': self._get_labels(test_path, os.path.join(script_folder(), _TEST_LABELS_FNAME)),
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                'archive': dl_manager.iter_archive(val_path),
                'labels': self._get_labels(val_path, os.path.join(script_folder(), _VALIDATION_LABELS_FNAME)),
            },
        ),
    ]

  def _fix_image(self, image_fname, image):
    patched_image_path = os.path.join(self.patch_images_folder, image_fname)
    if self.patch_images_folder and tf.io.gfile.exists(patched_image_path):
      # the patched images are all CMYK JPG, so no need to check against that!
      image = io.BytesIO(open(patched_image_path, 'rb').read())
    elif image_fname in CMYK_IMAGES:
      image = io.BytesIO(tfds.core.utils.jpeg_cmyk_to_rgb(image.read()))
    elif image_fname in JPEG_GZ_IMAGES:
      image = io.BytesIO(gzip.GzipFile(fileobj = image).read())
    elif image_fname in PNG_IMAGES:
      image = io.BytesIO(tfds.core.utils.png_to_jpeg(image.read()))

    return image

  def _generate_examples(self, archive, labels=None):
    """Yields examples."""
    if labels:  # Validation split
      for key, example in self._generate_examples_validation(archive, labels):
        yield key, example
    # Training split. Main archive contains archives names after a synset noun.
    # Each sub-archive contains pictures associated to that synset.
    for fname, fobj in archive:
      label = fname[:-4]  # fname is something like 'n01632458.tar'
      # TODO(b/117643231): in py3, the following lines trigger tarfile module
      # to call `fobj.seekable()`, which Gfile doesn't have. We should find an
      # alternative, as this loads ~150MB in RAM.
      fobj_mem = io.BytesIO(fobj.read())
      for image_fname, image in tfds.download.iter_archive(
          fobj_mem, tfds.download.ExtractMethod.TAR_STREAM):
        image = self._fix_image(image_fname, image)
        record = {
            'file_name': image_fname,
            'image': image,
            'label': label,
        }
        yield image_fname, record

  def _generate_examples_validation(self, archive, labels):
    for fname, fobj in archive:
      record = {
          'file_name': fname,
          'image': self._fix_image(fname, fobj),
          'label': labels[fname],
      }
      yield fname, record
