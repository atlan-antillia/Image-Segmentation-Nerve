# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The original segmentation dataset for Nerve Segmentation has been take from
#
# https://www.kaggle.com/competitions/ultrasound-nerve-segmentation/data
# Ultrasound Nerve Segmentation
# Identify nerve structures in ultrasound images of the neck

"""
Dataset Description
The task in this competition is to segment a collection of nerves called the Brachial Plexus (BP) in ultrasound images. 
You are provided with a large training set of images where the nerve has been manually annotated by humans. 
Annotators were trained by experts and instructed to annotate images where they felt confident about the existence of 
the BP landmark.

Please note these important points:

The dataset contains images where the BP is not present. Your algorithm should predict no pixel values in these cases.
As with all human-labeled data, you should expect to find noise, artifacts, and potential mistakes in the ground truth. 
Any individual mistakes (not affecting the broader integrity of the competition) will be left as is.
Due to the way the acquisition machine generates image frames, you may find identical images or very similar images.
In order to reduce the submission file sizes, this competition uses run-length encoding (RLE) on the pixel values. 
The details of how to use RLE are described on the 'Evaluation' page.
File descriptions
/train/ contains the training set images, named according to subject_imageNum.tif. Every image with the same subject 
number comes from the same person. This folder also includes binary mask images showing the BP segmentations.
/test/ contains the test set images, named according to imageNum.tif. You must predict the BP segmentation for these 
images and are not provided a subject number. There is no overlap between the subjects in the training and test sets.
train_masks.csv gives the training image masks in run-length encoded format. This is provided as a convenience to 
demonstrate how to turn image masks into encoded text values for submission.
sample_submission.csv shows the correct submission file format.

"""
# 2023/05/28 to-arai
# create_nerve_master.py

import os
import glob
import shutil
import traceback
import numpy as np

import cv2
import random

import os
import glob
import shutil
import traceback
import numpy as np
from PIL import Image

import cv2
import random

W = 256
H = 256

def create_nerve_master(input_dir, train_dir, test_dir):
  valid_mask_files = listup_valid_mask_files(input_dir)
  random.shuffle(valid_mask_files)
  num_files  = len(valid_mask_files)
  num_train  = int (num_files * 0.8)
  num_test   = int (num_files * 0.2)

  mask_train_files = valid_mask_files[0: num_train]
  mask_test_files  = valid_mask_files[num_train: num_files]
  print(" num files {}".format(num_files))
  print(" num train {}".format(num_train))
  print(" num test  {}".format(num_test))
  dataset_dirs = [train_dir,        test_dir]
  mask_files   = [mask_train_files, mask_test_files]
  for i, dataset_dir in enumerate(dataset_dirs):
     
    output_mask_dir  = os.path.join(dataset_dir, "masks")
    output_image_dir = os.path.join(dataset_dir, "images")
    if not os.path.exists(output_mask_dir):
      os.makedirs(output_mask_dir)
    if not os.path.exists(output_image_dir):
      os.makedirs(output_image_dir)

    for mask_file in mask_files[i]:
      image_file = mask_file.replace("_mask", "")
      print("=== mask file {}".format(mask_file))
      print("=== image file {}".format(image_file))

      resize_save_as_jpg(mask_file,  output_mask_dir, mask=True)
      resize_save_as_jpg(image_file, output_image_dir, mask=False)

def resize_save_as_jpg(image_file, output_dir, mask=False):
  print("=== resize_save_as jpg {}".format(image_file))
  if mask:
    image = Image.open(image_file).convert("L")
  else:
    image = Image.open(image_file)
  basename = os.path.basename(image_file)
  name     = basename.split(".")[0]
  w, h  = image.size
  SIZE = w
  if h > SIZE:
    SIZE = h
  # Create black background
  background = None
  if mask:
    background = Image.new("L", (SIZE, SIZE), 0)
  else:
    background = Image.new("RGB", (SIZE, SIZE), (128, 128, 128))
  x = int( (SIZE - w)/2 )
  y = int( (SIZE - h)/2 )
  if background == None:
    print("---background is None")
  background.paste(image, (x, y))
  #background.show()
  #input("---")

  background = background.resize((W, H)) #, Image.NEAREST)
  #angles = [0, 90, 180, 270]
  output_file = os.path.join(output_dir,  name + ".jpg")
  background.save(output_file)
  print("=== Saved {}".format(output_file))


def listup_valid_mask_files(input_dir):
  mask_pattern = input_dir + "/*_mask.tif"
  mask_files   = glob.glob(mask_pattern)
  valid_mask_files = []
  for mask_file in mask_files:
    image_file = mask_file.replace("_mask", "")
    mask = cv2.imread(mask_file)
        
    if np.any(mask==(255, 255, 255)):
      valid_mask_files.append(mask_file)
    else:
      pass
  return valid_mask_files     
    

if __name__ == "__main__":
  try:
    input_dir  = "./ultrasound-nerve-segmentation/train"
    output_dir = "./Nerve/"

    # Split nultrasound-nerve-segmentation/train
    #  to Nerve/train Neve/test
    train_dir = output_dir + "train"
    test_dir  = output_dir + "test"

    if os.path.exists(train_dir):
      shutil.rmtree(train_dir)
    if not os.path.exists(train_dir):
      os.makedirs(train_dir)
 
    if os.path.exists(test_dir):
      shutil.rmtree(test_dir)
    if not os.path.exists(test_dir):
      os.makedirs(test_dir)
    
    create_nerve_master(input_dir, train_dir, test_dir)

  except:
    traceback.print_exc()

