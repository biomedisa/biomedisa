#!/usr/bin/python3
##########################################################################
##                                                                      ##
##  Copyright (c) 2020 Philipp Lösel. All rights reserved.              ##
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
from multiprocessing import Process
import subprocess

from tensorflow.python.framework.errors_impl import ResourceExhaustedError
import tensorflow as tf
import numpy as np
import h5py
import time

def conv_network(train, predict, path_to_model,
            compress, epochs, batch_size, path_to_labels,
            stride_size, channels, normalize, path_to_img,
            x_scale, y_scale, z_scale, class_weights, crop_data,
            flip_x, flip_y, flip_z, rotate, validation_split,
            early_stopping, val_dice, learning_rate,
            path_val_img, path_val_labels):

    # get number of GPUs
    strategy = tf.distribute.MirroredStrategy()
    ngpus = int(strategy.num_replicas_in_sync)

    success = False
    path_to_final = None
    batch_size -= batch_size % ngpus                # batch size must be divisible by number of GPUs
    z_patch, y_patch, x_patch = 64, 64, 64          # dimensions of patches for regular training
    patch_size = 64                                 # x,y,z-patch size for the refinement network

    # adapt scaling and stridesize to patchsize
    stride_size = max(1, min(stride_size, 64))
    stride_size_refining = max(1, min(stride_size, 64))
    x_scale = x_scale - (x_scale - 64) % stride_size
    y_scale = y_scale - (y_scale - 64) % stride_size
    z_scale = z_scale - (z_scale - 64) % stride_size

    if train:

        try:
            # train network
            train_semantic_segmentation(normalize, path_to_img, path_to_labels, x_scale, y_scale,
                            z_scale, crop_data, path_to_model, z_patch, y_patch, x_patch, epochs,
                            batch_size, channels, validation_split, stride_size, class_weights,
                            flip_x, flip_y, flip_z, rotate, early_stopping, val_dice, learning_rate,
                            path_val_img, path_val_labels)
        except InputError:
            print('Error:', InputError.message)
            return success, InputError.message, None, None
        except MemoryError:
            print('MemoryError')
            return success, 'MemoryError', None, None
        except ResourceExhaustedError:
            print('Error: GPU out of memory')
            return success, 'GPU out of memory', None, None
        except Exception as e:
            print('Error:', e)
            return success, e, None, None

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
            extension = str(np.array(meta.get('extension')))
            region_of_interest = np.array(meta.get('region_of_interest'))
            hf.close()
        except Exception as e:
            print('Error:', e)
            return success, 'Invalid Biomedisa Network', None

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

            # load prediction data
            img, img_header, position, z_shape, y_shape, x_shape, region_of_interest = load_prediction_data(path_to_img,
                channels, x_scale, y_scale, z_scale, normalize, mu, sig, region_of_interest)

            # make prediction
            predict_semantic_segmentation(img, position, path_to_model, path_to_final,
                z_patch, y_patch, x_patch,  z_shape, y_shape, x_shape, compress, header,
                img_header, channels, stride_size, allLabels, batch_size, region_of_interest)

        except InputError:
            return success, InputError.message, None, None
        except MemoryError:
            print('MemoryError')
            return success, 'MemoryError', None, None
        except ResourceExhaustedError:
            print('GPU out of memory')
            return success, 'GPU out of memory', None, None
        except Exception as e:
            print('Error:', e)
            return success, e, None, None

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
    crop_data = 1 if any(x in parameters for x in ['--crop']) else 0                                # rop data automatically to region of interest
    flip_x = True if any(x in parameters for x in ['--flip_x']) else False                          # flip x-axis during training
    flip_y = True if any(x in parameters for x in ['--flip_y']) else False                          # flip y-axis during training
    flip_z = True if any(x in parameters for x in ['--flip_z']) else False                          # flip z-axis during training
    early_stopping = True if any(x in parameters for x in ['--early_stopping','-es']) else False    # early_stopping
    val_dice = True if any(x in parameters for x in ['--val_dice','-vd']) else False                # use dice score on validation data

    compress = 6            # wheter final result should be compressed or not
    epochs = 200            # epochs the network is trained
    channels = 1            # use voxel coordinates
    normalize = 1           # normalize images before training
    x_scale = 256           # images are scaled at x-axis to this size before training
    y_scale = 256           # images are scaled at y-axis to this size before training
    z_scale = 256           # images are scaled at z-axis to this size before training
    rotate = 0              # randomly rotate during training
    validation_split = 0.0  # percentage used for validation
    stride_size = 32        # stride size for patches
    batch_size = 24         # batch size
    learning_rate = 0.01    # learning rate
    path_val_img = None     # validation images
    path_val_labels = None  # validation labels

    for k in range(len(parameters)):
        if parameters[k] in ['--compress','-c']:
            compress = int(parameters[k+1])
        if parameters[k] in ['--epochs','-e']:
            epochs = int(parameters[k+1])
        if parameters[k] in ['--channels']:
            channels = int(parameters[k+1])
        if parameters[k] in ['--normalize']:
            normalize = int(parameters[k+1])
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
        if parameters[k] in ['--batch_size','-bs']:
            batch_size = int(parameters[k+1])
        if parameters[k] in ['--learning_rate','-lr']:
            learning_rate = float(parameters[k+1])
        if parameters[k] in ['--val_images','-vi']:
            path_val_img = str(parameters[k+1])
        if parameters[k] in ['--val_labels','-vl']:
            path_val_labels = str(parameters[k+1])

    # train network or predict segmentation
    conv_network(
        train, predict, path_to_model, compress,
        epochs, batch_size, path_to_labels, stride_size, channels,
        normalize, path_to_img, x_scale, y_scale, z_scale,
        balance, crop_data, flip_x, flip_y, flip_z, rotate,
        validation_split, early_stopping, val_dice, learning_rate,
        path_val_img, path_val_labels
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

