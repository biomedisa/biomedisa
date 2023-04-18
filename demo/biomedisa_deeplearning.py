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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from keras_helper import *
import biomedisa
import biomedisa_features.crop_helper as ch
from multiprocessing import Process
import argparse
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
import tensorflow as tf
import numpy as np
import h5py
import time

def conv_network(args):

    # get number of GPUs
    strategy = tf.distribute.MirroredStrategy()
    ngpus = int(strategy.num_replicas_in_sync)

    # batch size must be divisible by the number of GPUs and two
    rest = args.batch_size % (2*ngpus)
    if 2*ngpus - rest < rest:
        args.batch_size = args.batch_size + 2*ngpus - rest
    else:
        args.batch_size = args.batch_size - rest

    # validation batch size must be divisible by the number of GPUs and two
    rest = args.validation_batch_size % (2*ngpus)
    if 2*ngpus - rest < rest:
        validation_batch_size = args.validation_batch_size + 2*ngpus - rest
    else:
        validation_batch_size = args.validation_batch_size - rest

    # dimensions of patches for regular training
    z_patch, y_patch, x_patch = 64, 64, 64

    # adapt scaling and stridesize to patchsize
    x_scale = args.x_scale - (args.x_scale - 64) % args.stride_size
    y_scale = args.y_scale - (args.y_scale - 64) % args.stride_size
    z_scale = args.z_scale - (args.z_scale - 64) % args.stride_size

    if args.train:

        try:
            # train automatic cropping
            cropping_weights, cropping_config = None, None
            if args.crop_data:
                cropping_weights, cropping_config = ch.load_and_train(args.normalize, [args.path_to_img], [args.path_to_labels], args.path_to_model,
                            args.epochs, args.batch_size, args.validation_split, x_scale, y_scale, z_scale,
                            args.flip_x, args.flip_y, args.flip_z, args.rotate, args.only, args.ignore, [args.val_images], [args.val_labels], True)

            # train automatic segmentation
            train_semantic_segmentation(args.normalize, [args.path_to_img], [args.path_to_labels], x_scale, y_scale,
                            z_scale, args.crop_data, args.path_to_model, z_patch, y_patch, x_patch, args.epochs,
                            args.batch_size, args.channels, args.validation_split, args.stride_size, args.balance,
                            args.flip_x, args.flip_y, args.flip_z, args.rotate, args.early_stopping, args.val_tf, args.learning_rate,
                            [args.val_images], [args.val_labels], args.validation_stride_size, args.validation_freq,
                            validation_batch_size, cropping_weights, cropping_config, args.only, args.ignore, args.network_filters, args.resnet)
        except InputError:
            print('Error:', InputError.message)
        except MemoryError:
            print('MemoryError')
        except ResourceExhaustedError:
            print('Error: GPU out of memory')
        except Exception as e:
            print('Error:', e)

    if args.predict:

        try:
            # get meta data
            hf = h5py.File(args.path_to_model, 'r')
            meta = hf.get('meta')
            configuration = meta.get('configuration')
            channels, x_scale, y_scale, z_scale, normalize, mu, sig = np.array(configuration)[:]
            channels, x_scale, y_scale, z_scale, normalize, mu, sig = int(channels), int(x_scale), \
                                    int(y_scale), int(z_scale), int(normalize), float(mu), float(sig)
            allLabels = np.array(meta.get('labels'))
            header = np.array(meta.get('header'))
            try:
                extension = meta.get('extension').asstr()[()]
            except:
                extension = str(np.array(meta.get('extension')))
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
        filename = os.path.basename(args.path_to_img)
        filename = os.path.splitext(filename)[0]
        if filename[-4:] in ['.nii','.tar']:
            filename = filename[:-4]
        args.path_to_cropped_image = args.path_to_img.replace(os.path.basename(args.path_to_img), filename + '.cropped.tif')
        filename = 'final.' + filename
        path_to_final = args.path_to_img.replace(os.path.basename(args.path_to_img), filename + extension)
        args.path_to_cleaned = args.path_to_img.replace(os.path.basename(args.path_to_img), filename + '.cleaned' + extension)
        args.path_to_filled = args.path_to_img.replace(os.path.basename(args.path_to_img), filename + '.filled' + extension)
        args.path_to_cleaned_filled = args.path_to_img.replace(os.path.basename(args.path_to_img), filename + '.cleaned.filled' + extension)

        try:
            # crop data
            region_of_interest = None
            if crop_data:
                region_of_interest = ch.crop_data(args.path_to_img, args.path_to_model, args.path_to_cropped_image,
                    args.batch_size, args.debug_cropping, args.save_cropped)

            # load prediction data
            img, img_header, position, z_shape, y_shape, x_shape, region_of_interest = load_prediction_data(args.path_to_img,
                channels, x_scale, y_scale, z_scale, normalize, mu, sig, region_of_interest)

            # make prediction
            predict_semantic_segmentation(args, img, position, args.path_to_model, path_to_final,
                z_patch, y_patch, x_patch, z_shape, y_shape, x_shape, args.compression, header,
                img_header, channels, args.stride_size, allLabels, args.batch_size, region_of_interest)

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

    # initialize arguments
    parser = argparse.ArgumentParser(description='Biomedisa deeplearning.',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required arguments
    parser.add_argument('path_to_img', type=str, metavar='PATH_TO_IMAGE',
                        help='Location of image data (tarball or directory)')
    parser.add_argument('path', type=str, metavar='PATH',
                        help='Location of label data during training (tarball or directory) or model for prediction (h5)')

    # optional arguments
    parser.add_argument('-v', '--version', action='version', version=f'{biomedisa.__version__}',
                        help='Biomedisa version')
    parser.add_argument('-p','--predict', action='store_true', default=False,
                        help='Automatic/predict segmentation')
    parser.add_argument('-t','--train', action='store_true', default=False,
                        help='Train neural network')
    parser.add_argument('-b','--balance', action='store_true', default=False,
                        help='Balance class members of training patches')
    parser.add_argument('-cd','--crop_data', action='store_true', default=False,
                        help='Crop data automatically to region of interest')
    parser.add_argument('--flip_x', action='store_true', default=False,
                        help='Randomly flip x-axis during training')
    parser.add_argument('--flip_y', action='store_true', default=False,
                        help='Randomly flip y-axis during training')
    parser.add_argument('--flip_z', action='store_true', default=False,
                        help='Randomly flip z-axis during training')
    parser.add_argument('-vt','--val_tf', action='store_true', default=False,
                        help='Use tensorflow standard accuracy on validation data')
    parser.add_argument('--no-compression', action='store_true', default=False,
                        help='Disable compression of segmentation results')
    parser.add_argument('-cs','--create_slices', action='store_true', default=False,
                        help='Create slices of segmentation results')
    parser.add_argument('--ignore', type=str, default='none',
                        help='Ignore specific label(s), e.g. 2,5,6')
    parser.add_argument('--only', type=str, default='all',
                        help='Segment only specific label(s), e.g. 1,3,5')
    parser.add_argument('-nf', '--network_filters', type=str, default='32-64-128-256-512-1024',
                        help='Number of filters per layer up to the deepest, e.g. 32-64-128-256-512-1024')
    parser.add_argument('-rn','--resnet', action='store_true', default=False,
                        help='Use U-resnet instead of standard U-net')
    parser.add_argument('--channels', type=int, default=1,
                        help='Use voxel coordinates')
    parser.add_argument('-dc','--debug_cropping', action='store_true', default=False,
                        help='Debug cropping')
    parser.add_argument('-sc','--save_cropped', action='store_true', default=False,
                        help='Save cropped image')
    parser.add_argument('-e','--epochs', type=int, default=200,
                        help='Epochs the network is trained')
    parser.add_argument('--normalize', type=int, default=1,
                        help='Normalize images before training')
    parser.add_argument('-r','--rotate', type=float, default=0.0,
                        help='Randomly rotate during training')
    parser.add_argument('-vs','--validation_split', type=float, default=0.0,
                        help='Percentage of data used for validation')
    parser.add_argument('-lr','--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('-ss','--stride_size', metavar="[1-64]", type=int, choices=range(1,65), default=32,
                        help='Stride size for patches')
    parser.add_argument('-vss','--validation_stride_size', metavar="[1-64]", type=int, choices=range(1,65), default=32,
                        help='Stride size for validation patches')
    parser.add_argument('-vf','--validation_freq', type=int, default=1,
                        help='Epochs performed before validation')
    parser.add_argument('-bs','--batch_size', type=int, default=24,
                        help='batch size')
    parser.add_argument('-vbs','--validation_batch_size', type=int, default=24,
                        help='validation batch size')
    parser.add_argument('-vi','--val_images', type=str, metavar='PATH', default=None,
                        help='Location of validation image data (tarball or directory)')
    parser.add_argument('-vl','--val_labels', type=str, metavar='PATH', default=None,
                        help='Location of validation label data (tarball or directory)')
    parser.add_argument('-c','--clean', nargs='?', type=float, const=0.1, default=None,
                        help='Remove outliers, e.g. 0.5 means that objects smaller than 50 percent of the size of the largest object will be removed')
    parser.add_argument('-f','--fill', nargs='?', type=float, const=0.9, default=None,
                        help='Fill holes, e.g. 0.5 means that all holes smaller than 50 percent of the entire label will be filled')
    parser.add_argument('-xs','--x_scale', type=int, default=256,
                        help='Images and labels are scaled at x-axis to this size before training')
    parser.add_argument('-ys','--y_scale', type=int, default=256,
                        help='Images and labels are scaled at y-axis to this size before training')
    parser.add_argument('-zs','--z_scale', type=int, default=256,
                        help='Images and labels are scaled at z-axis to this size before training')
    parser.add_argument('-es','--early_stopping', type=int, default=0,
                        help='Training is terminated when the accuracy has not increased in the epochs defined by this')
    args = parser.parse_args()

    if args.predict:
        args.path_to_labels = None
        args.path_to_model = args.path
    if args.train:
        args.path_to_labels = args.path
        args.path_to_model = args.path_to_img + '.h5'

    # compression
    if args.no_compression:
        args.compression = False
    else:
        args.compression = True

    # train network or predict segmentation
    conv_network(args)

    # calculation time
    t = int(time.time() - TIC)
    if t < 60:
        time_str = str(t) + " sec"
    elif 60 <= t < 3600:
        time_str = str(t // 60) + " min " + str(t % 60) + " sec"
    elif 3600 < t:
        time_str = str(t // 3600) + " h " + str((t % 3600) // 60) + " min " + str(t % 60) + " sec"
    print('Total calculation time:', time_str)

