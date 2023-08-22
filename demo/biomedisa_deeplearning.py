#!/usr/bin/python3
##########################################################################
##                                                                      ##
##  Copyright (c) 2023 Philipp Lösel. All rights reserved.              ##
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
from demo.keras_helper import *
import biomedisa
import biomedisa_features.crop_helper as ch
from multiprocessing import Process
import argparse
import tensorflow as tf
import numpy as np
import h5py
import time

class Biomedisa(object):
     pass

def deep_learning(img_data, label_data=None, path_to_img=None, path_to_labels=None, path_to_model=None,
    predict=False, train=False, balance=False, crop_data=False, flip_x=False, flip_y=False, flip_z=False,
    swapaxes=False, val_tf=False, no_compression=False, create_slices=False, ignore='none', only='all',
    network_filters='32-64-128-256-512', resnet=False, channels=1, debug_cropping=False,
    save_cropped=False, epochs=100, no_normalization=False, rotate=0.0, validation_split=0.0,
    learning_rate=0.01, stride_size=32, validation_stride_size=32, validation_freq=1,
    batch_size=24, validation_batch_size=24, val_images=None, val_labels=None, clean=None,
    fill=None, x_scale=256, y_scale=256, z_scale=256, no_scaling=False, early_stopping=0,
    pretrained_model=None, fine_tune=False, classification=False, workers=1, cropping_epochs=50,
    val_img_data=None, val_label_data=None, x_range=None, y_range=None, z_range=None):

    # time
    TIC = time.time()

    # build biomedisa objects
    results = None
    args = Biomedisa()
    key_copy = tuple(locals().keys())
    for arg in key_copy:
        args.__dict__[arg] = locals()[arg]

    # path to model
    if args.train and not args.path_to_model:
        current_time = time.strftime("%d-%m-%Y_%H-%M-%S", time.localtime())
        args.path_to_model = os.getcwd() + f'/biomedisa_{current_time}.h5'
    if args.predict and not args.path_to_model:
        raise Exception("'path_to_model' must be specified")

    # compression
    if args.no_compression:
        args.compression = False
    else:
        args.compression = True

    # normalization
    if args.no_normalization:
        args.normalize = 0
    else:
        args.normalize = 1

    # disable file saving when called as a function
    if img_data is not None:
        args.path_to_img = None
        args.create_slices = False
        args.path_to_cropped_image = None

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
        args.validation_batch_size = args.validation_batch_size + 2*ngpus - rest
    else:
        args.validation_batch_size = args.validation_batch_size - rest

    # dimensions of patches for regular training
    args.z_patch, args.y_patch, args.x_patch = 64, 64, 64

    # adapt scaling to stridesize and patchsize
    args.x_scale = args.x_scale - (args.x_scale - 64) % args.stride_size
    args.y_scale = args.y_scale - (args.y_scale - 64) % args.stride_size
    args.z_scale = args.z_scale - (args.z_scale - 64) % args.stride_size

    if args.train:

        # train automatic cropping
        args.cropping_weights, args.cropping_config = None, None
        if args.crop_data:
            args.cropping_weights, args.cropping_config = ch.load_and_train(args.normalize, [args.path_to_img], [args.path_to_labels], args.path_to_model,
                        args.cropping_epochs, args.batch_size, args.validation_split, args.x_scale, args.y_scale, args.z_scale,
                        args.flip_x, args.flip_y, args.flip_z, args.rotate, args.only, args.ignore,
                        [args.val_images], [args.val_labels],
                        img_data, label_data, None,
                        val_img_data, val_label_data, None, True)

        # train automatic segmentation
        train_semantic_segmentation([args.path_to_img], [args.path_to_labels],
                        [args.val_images], [args.val_labels], args,
                        img_data, label_data, None,
                        val_img_data, val_label_data, None)

    if args.predict:

        # get meta data
        hf = h5py.File(args.path_to_model, 'r')
        meta = hf.get('meta')
        configuration = meta.get('configuration')
        channels, args.x_scale, args.y_scale, args.z_scale, normalize, mu, sig = np.array(configuration)[:]
        channels, args.x_scale, args.y_scale, args.z_scale, normalize, mu, sig = int(channels), int(args.x_scale), \
                                int(args.y_scale), int(args.z_scale), int(normalize), float(mu), float(sig)
        allLabels = np.array(meta.get('labels'))
        header = np.array(meta.get('header'))
        try:
            extension = meta.get('extension').asstr()[()]
        except:
            extension = str(np.array(meta.get('extension')))
        crop_data = True if 'cropping_weights' in hf else False
        hf.close()

        # if header is not Amira falling back to Multi-TIFF
        if extension != '.am':
            extension = '.tif'
            header = None

        # create path_to_final
        if args.path_to_img:
            filename = os.path.basename(args.path_to_img)
            filename = os.path.splitext(filename)[0]
            if filename[-4:] in ['.nii','.tar']:
                filename = filename[:-4]
            args.path_to_cropped_image = args.path_to_img.replace(os.path.basename(args.path_to_img), filename + '.cropped.tif')
            filename = 'final.' + filename
            args.path_to_final = args.path_to_img.replace(os.path.basename(args.path_to_img), filename + extension)
            args.path_to_cleaned = args.path_to_img.replace(os.path.basename(args.path_to_img), filename + '.cleaned' + extension)
            args.path_to_filled = args.path_to_img.replace(os.path.basename(args.path_to_img), filename + '.filled' + extension)
            args.path_to_cleaned_filled = args.path_to_img.replace(os.path.basename(args.path_to_img), filename + '.cleaned.filled' + extension)

        # crop data
        region_of_interest, cropped_volume = None, None
        if crop_data:
            region_of_interest, cropped_volume = ch.crop_data(args.path_to_img, args.path_to_model, args.path_to_cropped_image,
                args.batch_size, args.debug_cropping, args.save_cropped, img_data, args.x_range, args.y_range, args.z_range)

        # load prediction data
        img, img_header, position, z_shape, y_shape, x_shape, region_of_interest = load_prediction_data(args.path_to_img,
            channels, args.x_scale, args.y_scale, args.z_scale, args.no_scaling, normalize, mu, sig, region_of_interest, img_data)

        # make prediction
        results = predict_semantic_segmentation(args, img, position, args.path_to_model,
            args.z_patch, args.y_patch, args.x_patch, z_shape, y_shape, x_shape, args.compression, header,
            img_header, channels, args.stride_size, allLabels, args.batch_size, region_of_interest,
            args.classification)

        # results
        if header is not None:
            results['labels_header'] = header
        if cropped_volume is not None:
            results['cropped_volume'] = cropped_volume

    # calculation time
    t = int(time.time() - TIC)
    if t < 60:
        time_str = str(t) + " sec"
    elif 60 <= t < 3600:
        time_str = str(t // 60) + " min " + str(t % 60) + " sec"
    elif 3600 < t:
        time_str = str(t // 3600) + " h " + str((t % 3600) // 60) + " min " + str(t % 60) + " sec"
    print('Total calculation time:', time_str)

    return results

if __name__ == '__main__':

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
                        help='Balance foreground and background training patches')
    parser.add_argument('-cd','--crop_data', action='store_true', default=False,
                        help='Crop data automatically to region of interest')
    parser.add_argument('--flip_x', action='store_true', default=False,
                        help='Randomly flip x-axis during training')
    parser.add_argument('--flip_y', action='store_true', default=False,
                        help='Randomly flip y-axis during training')
    parser.add_argument('--flip_z', action='store_true', default=False,
                        help='Randomly flip z-axis during training')
    parser.add_argument('-sa','--swapaxes', action='store_true', default=False,
                        help='Randomly swap two axes during training')
    parser.add_argument('-vt','--val_tf', action='store_true', default=False,
                        help='Use tensorflow standard accuracy on validation data')
    parser.add_argument('--no_compression', action='store_true', default=False,
                        help='Disable compression of segmentation results')
    parser.add_argument('-cs','--create_slices', action='store_true', default=False,
                        help='Create slices of segmentation results')
    parser.add_argument('--ignore', type=str, default='none',
                        help='Ignore specific label(s), e.g. 2,5,6')
    parser.add_argument('--only', type=str, default='all',
                        help='Segment only specific label(s), e.g. 1,3,5')
    parser.add_argument('-nf', '--network_filters', type=str, default='32-64-128-256-512',
                        help='Number of filters per layer up to the deepest, e.g. 32-64-128-256-512')
    parser.add_argument('-rn','--resnet', action='store_true', default=False,
                        help='Use U-resnet instead of standard U-net')
    parser.add_argument('--channels', type=int, default=1,
                        help='Use voxel coordinates')
    parser.add_argument('-dc','--debug_cropping', action='store_true', default=False,
                        help='Debug cropping')
    parser.add_argument('-sc','--save_cropped', action='store_true', default=False,
                        help='Save cropped image')
    parser.add_argument('-e','--epochs', type=int, default=100,
                        help='Epochs the network is trained')
    parser.add_argument('-ce','--cropping_epochs', type=int, default=50,
                        help='Epochs the network for auto-cropping is trained')
    parser.add_argument('-nn','--no_normalization', action='store_true', default=False,
                        help='Disable image normalization')
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
    parser.add_argument('-ns','--no_scaling', action='store_true', default=False,
                        help='Do not resize image and label data')
    parser.add_argument('-es','--early_stopping', type=int, default=0,
                        help='Training is terminated when the accuracy has not increased in the epochs defined by this')
    parser.add_argument('-pm','--pretrained_model', type=str, metavar='PATH', default=None,
                        help='Location of pretrained model (only encoder will be trained if specified)')
    parser.add_argument('-ft','--fine_tune', action='store_true', default=False,
                        help='Fine-tune a pretrained model. Choose a smaller learning rate, e.g. 0.0001')
    parser.add_argument('-cl','--classification', action='store_true', default=False,
                        help='Train a model for image classification. Validation works only with `-vt` option')
    parser.add_argument('-w','--workers', type=int, default=1,
                        help='Parallel workers for batch processing')
    parser.add_argument('-xr','--x_range', nargs="+", type=int, default=None,
                        help='Manually crop x-axis of image data for prediction, e.g. -xr 100 200')
    parser.add_argument('-yr','--y_range', nargs="+", type=int, default=None,
                        help='Manually crop y-axis of image data for prediction, e.g. -xr 100 200')
    parser.add_argument('-zr','--z_range', nargs="+", type=int, default=None,
                        help='Manually crop z-axis of image data for prediction, e.g. -xr 100 200')
    args = parser.parse_args()

    if args.predict:
        args.path_to_labels = None
        args.path_to_model = args.path
    if args.train:
        args.path_to_labels = args.path
        args.path_to_model = args.path_to_img + '.h5'

    kwargs = vars(args)
    del kwargs['path']

    # train network or predict segmentation
    deep_learning(None, **kwargs)

