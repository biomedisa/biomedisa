#!/usr/bin/python3
##########################################################################
##                                                                      ##
##  Copyright (c) 2022 Philipp Lösel. All rights reserved.              ##
##                                                                      ##
##  This file is part of the open source project biomedisa.             ##
##                                                                      ##
##  Licensed under the European Union Public Licence (EUPL)             ##
##  v1.2, or - as soon as they will be approved by the                  ##
##  European Commission - subsequent versions of the EUPL;              ##
##                                                                      ##
##  You may redistribute it and/or modify it under the terms            ##
##  of the EUPL v1.2. You may not use this work except in               ##
##  compliance with this Licence.                                       ##
##                                                                      ##
##  You can obtain a copy of the Licence at:                            ##
##                                                                      ##
##  https://joinup.ec.europa.eu/page/eupl-text-11-12                    ##
##                                                                      ##
##  Unless required by applicable law or agreed to in                   ##
##  writing, software distributed under the Licence is                  ##
##  distributed on an "AS IS" basis, WITHOUT WARRANTIES                 ##
##  OR CONDITIONS OF ANY KIND, either express or implied.               ##
##                                                                      ##
##  See the Licence for the specific language governing                 ##
##  permissions and limitations under the Licence.                      ##
##                                                                      ##
##########################################################################

import sys, os
from keras_helper import *
import crop_helper as ch
from multiprocessing import Process
import subprocess

from tensorflow.python.framework.errors_impl import ResourceExhaustedError
import tensorflow as tf
import numpy as np
import h5py
import time

def conv_network(train, predict, path_to_model, compress, epochs, batch_size, path_to_labels,
            stride_size, channels, normalize, path_to_img, x_scale, y_scale, z_scale, class_weights, crop_data,
            flip_x, flip_y, flip_z, rotate, validation_split, early_stopping, val_tf, learning_rate,
            path_val_img, path_val_labels, validation_stride_size, validation_freq, validation_batch_size,
            debug_cropping):

    # get number of GPUs
    strategy = tf.distribute.MirroredStrategy()
    ngpus = int(strategy.num_replicas_in_sync)

    # batch size must be divisible by the number of GPUs and two
    rest = batch_size % (2*ngpus)
    if 2*ngpus - rest < rest:
        batch_size = batch_size + 2*ngpus - rest
    else:
        batch_size = batch_size - rest

    # validation batch size must be divisible by the number of GPUs and two
    rest = validation_batch_size % (2*ngpus)
    if 2*ngpus - rest < rest:
        validation_batch_size = validation_batch_size + 2*ngpus - rest
    else:
        validation_batch_size = validation_batch_size - rest

    # dimensions of patches for regular training
    z_patch, y_patch, x_patch = 64, 64, 64

    # adapt scaling and stridesize to patchsize
    stride_size = max(1, min(stride_size, 64))
    stride_size_refining = max(1, min(stride_size, 64))
    validation_stride_size = max(1, min(validation_stride_size, 64))
    x_scale = x_scale - (x_scale - 64) % stride_size
    y_scale = y_scale - (y_scale - 64) % stride_size
    z_scale = z_scale - (z_scale - 64) % stride_size

    if train:

        try:
            # train automatic cropping
            cropping_weights, cropping_config = None, None
            if crop_data:
                cropping_weights, cropping_config = ch.load_and_train(normalize, path_to_img, path_to_labels, path_to_model,
                            epochs, batch_size, validation_split, x_scale, y_scale, z_scale,
                            flip_x, flip_y, flip_z, rotate, path_val_img, path_val_labels)

            # train network
            train_semantic_segmentation(normalize, path_to_img, path_to_labels, x_scale, y_scale,
                            z_scale, crop_data, path_to_model, z_patch, y_patch, x_patch, epochs,
                            batch_size, channels, validation_split, stride_size, class_weights,
                            flip_x, flip_y, flip_z, rotate, early_stopping, val_tf, learning_rate,
                            path_val_img, path_val_labels, validation_stride_size, validation_freq,
                            validation_batch_size, cropping_weights, cropping_config)
        except InputError:
            print('Error:', InputError.message)
        except MemoryError:
            print('MemoryError')
        except ResourceExhaustedError:
            print('Error: GPU out of memory')
        except Exception as e:
            print('Error:', e)

    if predict:

        try:
            # get meta data
            hf = h5py.File(path_to_model, 'r')
            meta = hf.get('meta')
            configuration = meta.get('configuration')
            channels, x_scale, y_scale, z_scale, normalize, mu, sig = np.array(configuration)[:]
            channels, x_scale, y_scale, z_scale, normalize, mu, sig = int(channels), int(x_scale), \
                                    int(y_scale), int(z_scale), int(normalize), float(mu), float(sig)
            allLabels = np.array(meta.get('labels'))
            header = np.array(meta.get('header'))
            extension = str(np.array(meta.get('extension'), dtype=np.unicode_))
            crop_data = True if 'cropping_weights' in hf else False
            hf.close()
        except Exception as e:
            print('Error:', e)
            return

        # if header is not Amira falling back to Multi-TIFF
        if extension != '.am':
            extension = '.tif'
            header = None

        # create path_to_final
        filename = os.path.basename(path_to_img)
        filename = os.path.splitext(filename)[0]
        if filename[-4:] in ['.nii','.tar']:
            filename = filename[:-4]
        filename = 'final.' + filename
        path_to_final = path_to_img.replace(os.path.basename(path_to_img), filename + extension)

        try:
            # crop data
            region_of_interest = None
            if crop_data:
                region_of_interest = ch.crop_data(path_to_img, path_to_model, batch_size, debug_cropping)

            # load prediction data
            img, img_header, position, z_shape, y_shape, x_shape, region_of_interest = load_prediction_data(path_to_img,
                channels, x_scale, y_scale, z_scale, normalize, mu, sig, region_of_interest)

            # make prediction
            predict_semantic_segmentation(img, position, path_to_model, path_to_final,
                z_patch, y_patch, x_patch,  z_shape, y_shape, x_shape, compress, header,
                img_header, channels, stride_size, allLabels, batch_size, region_of_interest)

        except InputError:
            print('Error:', InputError.message)
        except MemoryError:
            print('MemoryError')
        except ResourceExhaustedError:
            print('GPU out of memory')
        except Exception as e:
            print('Error:', e)

if __name__ == '__main__':

    # time
    TIC = time.time()

    # get arguments
    predict = 1 if any(x in sys.argv for x in ['--predict','-p']) else 0
    train = 1 if any(x in sys.argv for x in ['--train','-t']) else 0

    # path to data
    path_to_img = sys.argv[1]
    if predict:
        path_to_labels = None
        path_to_model = sys.argv[2]
    if train:
        path_to_labels = sys.argv[2]
        path_to_model = path_to_img + '.h5'

    # parameters
    parameters = sys.argv
    balance = 1 if any(x in parameters for x in ['--balance','-b']) else 0                          # balance class members of training patches
    crop_data = 1 if any(x in parameters for x in ['--crop_data','-cd']) else 0                     # crop data automatically to region of interest
    flip_x = True if any(x in parameters for x in ['--flip_x']) else False                          # flip x-axis during training
    flip_y = True if any(x in parameters for x in ['--flip_y']) else False                          # flip y-axis during training
    flip_z = True if any(x in parameters for x in ['--flip_z']) else False                          # flip z-axis during training
    val_tf = True if any(x in parameters for x in ['--val_tf','-vt']) else False                    # use tensorflow standard accuracy on validation data
    debug_cropping = True if any(x in parameters for x in ['--debug_cropping','-dc']) else False    # debug cropping

    compress = True                 # compress segmentation result
    epochs = 200                    # epochs the network is trained
    channels = 1                    # use voxel coordinates
    normalize = 1                   # normalize images before training
    x_scale = 256                   # images are scaled at x-axis to this size before training
    y_scale = 256                   # images are scaled at y-axis to this size before training
    z_scale = 256                   # images are scaled at z-axis to this size before training
    rotate = 0                      # randomly rotate during training
    validation_split = 0.0          # percentage used for validation
    stride_size = 32                # stride size for patches
    validation_stride_size = 32     # stride size for validation patches
    validation_freq = 1             # epochs to be performed prior to validation
    batch_size = 24                 # batch size
    validation_batch_size = 24      # validation batch size
    learning_rate = 0.01            # learning rate
    path_val_img = None             # validation images
    path_val_labels = None          # validation labels
    early_stopping = 0              # early_stopping

    for k in range(len(parameters)):
        if parameters[k] in ['--epochs','-e']:
            epochs = int(parameters[k+1])
        if parameters[k] in ['--channels']:
            channels = int(parameters[k+1])
        if parameters[k] in ['--xsize','-xs']:
            x_scale = int(parameters[k+1])
        if parameters[k] in ['--ysize','-ys']:
            y_scale = int(parameters[k+1])
        if parameters[k] in ['--zsize','-zs']:
            z_scale = int(parameters[k+1])
        if parameters[k] in ['--rotate','-r']:
            rotate = int(parameters[k+1])
        if parameters[k] in ['--validation_split','-vs']:
            validation_split = float(parameters[k+1])
        if parameters[k] in ['--stride_size','-ss']:
            stride_size = int(parameters[k+1])
        if parameters[k] in ['--validation_stride_size','-vss']:
            validation_stride_size = int(parameters[k+1])
        if parameters[k] in ['--validation_freq','-vf']:
            validation_freq = int(parameters[k+1])
        if parameters[k] in ['--batch_size','-bs']:
            batch_size = int(parameters[k+1])
        if parameters[k] in ['--validation_batch_size','-vbs']:
            validation_batch_size = int(parameters[k+1])
        if parameters[k] in ['--learning_rate','-lr']:
            learning_rate = float(parameters[k+1])
        if parameters[k] in ['--val_images','-vi']:
            path_val_img = str(parameters[k+1])
        if parameters[k] in ['--val_labels','-vl']:
            path_val_labels = str(parameters[k+1])
        if parameters[k] in ['--early_stopping','-es']:
            early_stopping = int(parameters[k+1])

    # train network or predict segmentation
    conv_network(
        train, predict, path_to_model, compress,
        epochs, batch_size, path_to_labels, stride_size, channels,
        normalize, path_to_img, x_scale, y_scale, z_scale,
        balance, crop_data, flip_x, flip_y, flip_z, rotate,
        validation_split, early_stopping, val_tf, learning_rate,
        path_val_img, path_val_labels, validation_stride_size,
        validation_freq, validation_batch_size, debug_cropping
        )

    # calculation time
    t = int(time.time() - TIC)
    if t < 60:
        time_str = str(t) + " sec"
    elif 60 <= t < 3600:
        time_str = str(t // 60) + " min " + str(t % 60) + " sec"
    elif 3600 < t:
        time_str = str(t // 3600) + " h " + str((t % 3600) // 60) + " min " + str(t % 60) + " sec"
    print('Total calculation time:', time_str)

