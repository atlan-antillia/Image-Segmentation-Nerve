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

# This is based on the code in the following web sites:

# 1. Keras U-Net starter - LB 0.277
# https://www.kaggle.com/code/keegil/keras-u-net-starter-lb-0-277/notebook

# 2. U-Net: Convolutional Networks for Biomedical Image Segmentation
# https://arxiv.org/pdf/1505.04597.pdf

# You can customize your TensorflowUnNet model by using a configration file
# Example: train.config

# 2023/05/28 Added dilation_rate parameter to Conv2D
# 2023/05/28 Modified to read loss and metrics from train_eval_infer.config file.

"""
[model]
image_width    = 256
image_height   = 256
image_channels = 3

num_classes    = 1
base_filters   = 16
num_layers     = 8
dropout_rate   = 0.05
learning_rate  = 0.001
"""

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import shutil
import sys
import glob
import traceback
import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Conv2D, Dropout, Conv2D, MaxPool2D

from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.activations import elu, relu
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from losses import dice_coef, basnet_hybrid_loss, sensitivity, specificity

from ConfigParser import ConfigParser

from EpochChangeCallback import EpochChangeCallback
from GrayScaleImageWriter import GrayScaleImageWriter

MODEL  = "model"
TRAIN  = "train"
INFER  = "infer"

BEST_MODEL_FILE = "best_model.h5"

class TensorflowUNet:

  def __init__(self, config_file):
    self.config    = ConfigParser(config_file)
    image_height   = self.config.get(MODEL, "image_height")

    image_width    = self.config.get(MODEL, "image_width")
    image_channels = self.config.get(MODEL, "image_channels")

    num_classes    = self.config.get(MODEL, "num_classes")
    base_filters   = self.config.get(MODEL, "base_filters")
    num_layers     = self.config.get(MODEL, "num_layers")
    
    self.model     = self.create(num_classes, image_height, image_width, image_channels, 
                            base_filters = base_filters, num_layers = num_layers)
    
    learning_rate  = self.config.get(MODEL, "learning_rate")

    self.optimizer = Adam(learning_rate = learning_rate, 
         beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, 
         amsgrad=False) 

    # <modified date="2023/05/28">
    #  Modified to read loss and metrics from train_eval_infer.config file.
    binary_crossentropy = tf.keras.metrics.binary_crossentropy
    binary_accuracy     = tf.keras.metrics.binary_accuracy

    # Default loss and metrics functions
    self.loss    = binary_crossentropy
    self.metrics = [binary_accuracy]
    
    # Read a loss function name from our config file, and eval it.
    # loss = "binary_crossentropy"
    self.loss  = eval(self.config.get(MODEL, "loss"))

    # Read a list of metrics function names, ant eval each of the list,
    # metrics = ["binary_accuracy"]
    metrics  = self.config.get(MODEL, "metrics")
    self.metrics = []
    for metric in metrics:
      self.metrics.append(eval(metric))
    
    print("--- loss    {}".format(self.loss))
    print("--- metrics {}".format(self.metrics))
    #</modified>

    self.model.compile(optimizer = self.optimizer, loss= self.loss, metrics = self.metrics)

    self.model.summary()    
    self.model_loaded = False


  def create(self, num_classes, image_height, image_width, image_channels,
            base_filters = 16, num_layers = 5):
    # inputs
    print("Input image_height {} image_width {} image_channels {}".format(image_height, image_width, image_channels))
    inputs = Input((image_height, image_width, image_channels))
    s= Lambda(lambda x: x / 255)(inputs)

    # Encoder
    dropout_rate = self.config.get(MODEL, "dropout_rate")
    enc         = []
    kernel_size = (3, 3)
    pool_size   = (2, 2)
    dilation    = (2, 2)
    strides     = (1, 1)
    try:
      dilation_ = self.config.get(MODEL, "dilation")
      (d1, d2) = dilation_
      if d1 == d2:
        dilation = dilation_
    except:
      pass


    for i in range(num_layers):
      filters = base_filters * (2**i)
      c = Conv2D(filters, kernel_size, strides=strides, activation=relu, 
                 kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(s)
      c = Dropout(dropout_rate * i)(c)
      c = Conv2D(filters, kernel_size, strides=strides, activation=relu, 
                 kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(c)
      if i < (num_layers-1):
        p = MaxPool2D(pool_size=pool_size)(c)
        s = p
      enc.append(c)
    
    enc_len = len(enc)
    enc.reverse()
    n = 0
    c = enc[n]
    
    # --- Decoder
    for i in range(num_layers-1):
      f = enc_len - 2 - i
      filters = base_filters* (2**f)
      u = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c)
      n += 1
      u = concatenate([u, enc[n]])
      u = Conv2D(filters, kernel_size, strides=strides, activation=relu, 
                 kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(u)
      u = Dropout(dropout_rate * f)(u)
      u = Conv2D(filters, kernel_size, strides=strides, activation=relu, 
                 kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(u)
      c  = u

    # outouts
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c)

    # create Model
    model = Model(inputs=[inputs], outputs=[outputs])

    return model


  def train(self, x_train, y_train): 
    batch_size = self.config.get(TRAIN, "batch_size")
    epochs     = self.config.get(TRAIN, "epochs")
    patience   = self.config.get(TRAIN, "patience")
    eval_dir   = self.config.get(TRAIN, "eval_dir")
    model_dir  = self.config.get(TRAIN, "model_dir")
    metrics    = ["accuracy", "val_accuracy"]
    try:
      metrics    = self.config.get(TRAIN, "metrics")
    except:
      pass
    if os.path.exists(model_dir):
      shutil.rmtree(model_dir)

    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
    weight_filepath   = os.path.join(model_dir, BEST_MODEL_FILE)

    early_stopping = EarlyStopping(patience=patience, verbose=1)
    check_point    = ModelCheckpoint(weight_filepath, verbose=1, save_best_only=True)
    epoch_change   = EpochChangeCallback(eval_dir, metrics)

    history = self.model.fit(x_train, y_train, 
                    validation_split=0.2, batch_size=batch_size, epochs=epochs, 
                    callbacks=[early_stopping, check_point, epoch_change],
                    verbose=1)

  def load_model(self) :
    rc = False
    if  not self.model_loaded:    
      model_dir  = self.config.get(TRAIN, "model_dir")
      weight_filepath = os.path.join(model_dir, BEST_MODEL_FILE)
      if os.path.exists(weight_filepath):
        self.model.load_weights(weight_filepath)
        self.model_loaded = True
        print("=== Loaded a weight_file {}".format(weight_filepath))
        rc = True
      else:
        message = "Not found a weight_file " + weight_filepath
        raise Exception(message)
    else:
      print("== Already loaded a weight file loaded ")
    return rc


  # 2023/05/05 Added newly.    
  def infer(self, input_dir, output_dir, expand=True):
    writer       = GrayScaleImageWriter()
    # We are intereseted in png and jpg files.
    # Sorry the bmp or tif files are ignored.
    image_files  = glob.glob(input_dir + "/*.png")
    image_files += glob.glob(input_dir + "/*.jpg")

    width        = self.config.get(MODEL, "image_width")
    height       = self.config.get(MODEL, "image_height")
    # 2023/05/24
    merged_dir   = None
    try:
      merged_dir = self.config.get(INFER, "merged_dir")
      if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
      if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)
    except:
      pass

    for image_file in image_files:
      basename = os.path.basename(image_file)
      name     = basename.split(".")[0]
      img      = cv2.imread(image_file, cv2.COLOR_BGR2RGB)
      h = img.shape[0]
      w = img.shape[1]
      # Any way, we have to resize input image to match the input size of our TensorflowUNet model.
      img         = cv2.resize(img, (width, height))
      predictions = self.predict([img], expand=expand)
      prediction  = predictions[0]
      image       = prediction[0]    
      # Resize the predicted image to be the original image size (w, h), and save it as a grayscale image.
      # Probably, this is a natural way for all humans. 
      mask = writer.save_resized(image, (w, h), output_dir, name)
      # 2023/05/24
      print("--- image_file {}".format(image_file))
      if merged_dir !=None:
        # Resize img to the original size (w, h)
        img   = cv2.resize(img, (w, h))
        print("=== mask {}".format(mask.shape))
        print("=== image {}".format(img.shape))

        img += mask
        merged_file = os.path.join(merged_dir, basename)
        cv2.imwrite(merged_file, img)     

  def predict(self, images, expand=True):
    self.load_model()

    predictions = []
    for image in images:
      #print("=== Input image shape {}".format(image.shape))
      if expand:
        image = np.expand_dims(image, 0)
      pred = self.model.predict(image)
      predictions.append(pred)
    return predictions    


  def evaluate(self, x_test, y_test): 
    self.load_model()

    score = self.model.evaluate(x_test, y_test, verbose=1)
    print("Test loss    :{}".format(score[0]))     
    print("Test accuracy:{}".format(score[1]))
     
    
if __name__ == "__main__":
  try:
    config_file    = "./model.config"
    config   = ConfigParser(config_file)

    width    = config.get(MODEL, "image_width")
    height   = config.get(MODEL, "image_height")
    channels = config.get(MODEL, "image_channels")

    if not (width == height and  height % 128 == 0 and width % 128 == 0):
      raise Exception("Image width should be a multiple of 128. For example 128, 256, 512")
    
    # Create a UNetMolde and compile
    model    = TensorflowUNet(config_file)
    
    """
    resized_image     = (height, width, channels)
    train_datapath    = "./stage1_train/"
    dataset           = NucleiDataset(resized_image)

    x_train, y_train  = dataset.create(train_datapath, has_mask=True)

    model.train(x_train, y_train)
    """

  except:
    traceback.print_exc()
    
