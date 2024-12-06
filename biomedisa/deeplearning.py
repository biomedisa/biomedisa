#!/usr/bin/python3
##########################################################################
##                                                                      ##
##  Copyright (c) 2019-2024 Philipp Lösel. All rights reserved.         ##
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

import os
import biomedisa
import biomedisa.features.crop_helper as ch
from biomedisa.features.keras_helper import *
from biomedisa.features.biomedisa_helper import _error_, unique_file_path
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
import tensorflow as tf
import numpy as np
import traceback
import argparse
import h5py
import time
import subprocess
import glob
import tempfile
import tifffile

class Biomedisa(object):
     pass

def get_gpu_memory():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], encoding='utf-8')
        # Convert lines to list
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        return gpu_memory
    except:
        return None

def number_of_slices(file_path):
    with tifffile.TiffFile(file_path) as tiff:
        z_dim = len(tiff.pages)
    return z_dim

def deep_learning(img_data, label_data=None, val_img_data=None, val_label_data=None,
    path_to_images=None, path_to_labels=None, val_images=None, val_labels=None,
    path_to_model=None, predict=False, train=False, header_file=None,
    balance=False, crop_data=False, flip_x=False, flip_y=False, flip_z=False,
    swapaxes=False, train_dice=False, val_dice=True, compression=True, ignore='none', only='all',
    network_filters='32-64-128-256-512', resnet=False, debug_cropping=False,
    save_cropped=False, epochs=100, normalization=True, rotate=0.0, validation_split=0.0,
    learning_rate=0.01, stride_size=32, validation_stride_size=32, validation_freq=1,
    batch_size=None, x_scale=256, y_scale=256, z_scale=256, scaling=True, early_stopping=0,
    pretrained_model=None, fine_tune=False, workers=1, cropping_epochs=50,
    x_range=None, y_range=None, z_range=None, header=None, extension='.tif',
    img_header=None, img_extension='.tif', average_dice=False, django_env=False,
    path=None, success=True, return_probs=False, patch_normalization=False,
    z_patch=64, y_patch=64, x_patch=64, path_to_logfile=None, img_id=None, label_id=None,
    remote=False, queue=0, username=None, shortfilename=None, dice_loss=False,
    acwe=False, acwe_alpha=1.0, acwe_smooth=1, acwe_steps=3, clean=None, fill=None,
    separation=False, mask=None, refinement=False):

    # create biomedisa
    bm = Biomedisa()
    bm.process = 'deep_learning'
    results = None

    # time
    TIC = time.time()

    # transfer arguments
    key_copy = tuple(locals().keys())
    for arg in key_copy:
        bm.__dict__[arg] = locals()[arg]

    # normalization
    bm.normalize = 1 if bm.normalization else 0

    # use patch normalization instead of normalizing the entire volume
    if not bm.scaling:
        bm.normalize = 0
        bm.patch_normalization = True

    # region of interest
    if any([bm.x_range, bm.y_range, bm.z_range]):
        if not bm.x_range:
            bm.x_range = [0,None]
        if not bm.y_range:
            bm.y_range = [0,None]
        if not bm.z_range:
            bm.z_range = [0,None]

    # django environment
    if bm.django_env:

        # path to image data
        if bm.train:

            # training files
            bm.path_to_images = bm.path_to_images.split(',')[:-1]
            bm.path_to_labels = bm.path_to_labels.split(',')[:-1]

            # validation files
            if bm.val_images is not None:
                bm.val_images = bm.val_images.split(',')[:-1]
                bm.val_labels = bm.val_labels.split(',')[:-1]
            else:
                bm.val_images = [None]
                bm.val_labels = [None]

            # project name
            project = os.path.splitext(bm.shortfilename)[0]

            # path to model
            bm.path_to_model = biomedisa.BASE_DIR + f'/private_storage/images/{bm.username}/{project}.h5'
            if not bm.remote:
                bm.path_to_model = unique_file_path(bm.path_to_model)

        if bm.predict:
            project = os.path.splitext(os.path.basename(bm.path_to_model))[0]

        # create pid object
        from biomedisa.features.django_env import create_pid_object
        create_pid_object(os.getpid(), bm.remote, bm.queue, bm.img_id, (bm.path_to_model if bm.train else ''))

        # write in log file
        with open(bm.path_to_logfile, 'a') as logfile:
            print('%s %s %s %s' %(time.ctime(), bm.username, bm.shortfilename, 'Process was started.'), file=logfile)
            print(f'PROJECT:{project} PREDICT:{bm.predict} IMG:{bm.shortfilename}', file=logfile)
            if bm.train:
                print(f'IMG_LIST:{bm.path_to_images} LABEL_LIST:{bm.path_to_labels} VAL_IMG_LIST:{bm.val_images} VAL_LABEL_LIST:{bm.val_labels}', file=logfile)

    # path to model
    if bm.train and not bm.path_to_model:
        current_time = time.strftime("%d-%m-%Y_%H-%M-%S", time.localtime())
        bm.path_to_model = os.getcwd() + f'/biomedisa_{current_time}.h5'
    if bm.predict and not bm.path_to_model:
        raise Exception("'path_to_model' must be specified")

    # disable file saving when called as a function
    if bm.img_data is not None:
        bm.path_to_images = None
        bm.path_to_cropped_image = None

    # adapt scaling to stride size and patch size
    bm.stride_size = max(1, min(bm.stride_size, 64))
    bm.x_scale = bm.x_scale - (bm.x_scale - bm.x_patch) % bm.stride_size
    bm.y_scale = bm.y_scale - (bm.y_scale - bm.y_patch) % bm.stride_size
    bm.z_scale = bm.z_scale - (bm.z_scale - bm.z_patch) % bm.stride_size

    # adapt batch size to available gpu memory
    if bm.batch_size is None:
        bm.batch_size = 6
        gpu_memory = get_gpu_memory()
        if gpu_memory:
            if bm.predict:
                gpu_memory = gpu_memory[:1]
            if 14000 < np.sum(gpu_memory) < 28000:
                bm.batch_size = 12
            elif 28000 <= np.sum(gpu_memory):
                bm.batch_size = 24

    if bm.train:

        # path to results
        bm.path_to_final = None
        bm.path_to_cropped_image = None

        # get number of GPUs
        strategy = tf.distribute.MirroredStrategy()
        ngpus = int(strategy.num_replicas_in_sync)

        # batch size must be divisible by the number of GPUs and two
        rest = bm.batch_size % (2*ngpus)
        if 2*ngpus - rest < rest:
            bm.batch_size = bm.batch_size + 2*ngpus - rest
        else:
            bm.batch_size = bm.batch_size - rest

        if not bm.django_env:
            bm.path_to_images, bm.path_to_labels = [bm.path_to_images], [bm.path_to_labels]
            bm.val_images, bm.val_labels = [bm.val_images], [bm.val_labels]

        # train automatic cropping
        bm.cropping_weights, bm.cropping_config, bm.cropping_norm = None, None, None
        if bm.crop_data:
            bm.cropping_weights, bm.cropping_config, bm.cropping_norm = ch.load_and_train(bm)

        # train automatic segmentation
        train_segmentation(bm)

    if bm.predict:

        # get meta data
        hf = h5py.File(bm.path_to_model, 'r')
        meta = hf.get('meta')
        configuration = meta.get('configuration')
        channels, bm.x_scale, bm.y_scale, bm.z_scale, bm.normalize, mu, sig = np.array(configuration)[:]
        channels, bm.x_scale, bm.y_scale, bm.z_scale, bm.normalize, mu, sig = int(channels), int(bm.x_scale), \
                                int(bm.y_scale), int(bm.z_scale), int(bm.normalize), float(mu), float(sig)
        if 'normalization' in meta:
            normalization_parameters = np.array(meta['normalization'], dtype=float)
        else:
            normalization_parameters = np.array([[mu],[sig]])
        bm.allLabels = np.array(meta.get('labels'))
        if 'patch_normalization' in meta:
            bm.patch_normalization = bool(meta['patch_normalization'][()])
        if 'scaling' in meta:
            bm.scaling = bool(meta['scaling'][()])

        # check if amira header is available in the network
        if bm.header is None and meta.get('header') is not None:
            bm.header = [np.array(meta.get('header'))]
            bm.extension = '.am'

        # crop data
        crop_data = True if 'cropping_weights' in hf else False
        hf.close()

        # make temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:

            # extract image files from tar file
            bm.tarfile = False
            if bm.path_to_images is not None and (os.path.splitext(bm.path_to_images)[1]=='.tar' or bm.path_to_images[-7:]=='.tar.gz'):
                bm.tarfile = True
                path_to_result = os.path.dirname(bm.path_to_images) + '/final.'+os.path.basename(bm.path_to_images)
                if path_to_result[-3:]=='.gz':
                    path_to_result = path_to_result[:-3]
                if bm.django_env and not bm.remote:
                    path_to_result = unique_file_path(path_to_result)
                tar = tarfile.open(bm.path_to_images)
                tar.extractall(path=temp_dir)
                tar.close()
                bm.path_to_images = temp_dir
                bm.save_cropped, bm.acwe = False, False
                bm.clean, bm.fill = None, None

            # list of images
            path_to_finals = []
            if bm.path_to_images is not None and os.path.isdir(bm.path_to_images):
                # load list of volumetric image files
                files = []
                for data_type in ['.am','.hdr','.mhd','.mha','.nrrd','.nii','.nii.gz','.zip','.mrc']:
                    files += [file for file in glob.glob(bm.path_to_images+'/**/*'+data_type, recursive=True) if not os.path.basename(file).startswith('.')]
                for data_type in ['.tif','.tiff']:
                    files += [file for file in glob.glob(bm.path_to_images+'/**/*'+data_type, recursive=True) if not os.path.basename(file).startswith('.') and number_of_slices(file)>1]
                if len(files)==0: # assume directory of 2D slices
                    bm.path_to_images = [bm.path_to_images]
                else:
                    bm.path_to_images = files
            else:
                bm.path_to_images = [bm.path_to_images]

            # loop over all images
            for bm.path_to_image in bm.path_to_images:

                # create path_to_final
                if bm.path_to_image:
                    filename = os.path.basename(bm.path_to_image)
                    filename = os.path.splitext(filename)[0]
                    if filename[-4:] == '.nii':
                        filename = filename[:-4]
                    bm.path_to_cropped_image = os.path.dirname(bm.path_to_image) + '/' + filename + '.cropped.tif'
                    if bm.django_env and not bm.remote and not bm.tarfile:
                        bm.path_to_cropped_image = unique_file_path(bm.path_to_cropped_image)
                    filename = 'final.' + filename
                    if bm.refinement:
                        filename += '.refined'
                    bm.path_to_final = os.path.dirname(bm.path_to_image) + '/' + filename + bm.extension
                    if bm.django_env and not bm.remote and not bm.tarfile:
                        bm.path_to_final = unique_file_path(bm.path_to_final)

                # crop data
                region_of_interest, cropped_volume = None, None
                if any([bm.x_range, bm.y_range, bm.z_range]):
                    region_of_interest = [bm.z_range[0], bm.z_range[1], bm.y_range[0], bm.y_range[1], bm.x_range[0], bm.x_range[1]]
                elif crop_data:
                    region_of_interest, cropped_volume = ch.crop_data(bm)

                # make prediction
                results, bm = predict_segmentation(bm, region_of_interest,
                    channels, normalization_parameters)

                # results
                if cropped_volume is not None:
                    results['cropped_volume'] = cropped_volume

                # path to results
                if bm.path_to_image:
                    path_to_finals.append(bm.path_to_final)

            # write tar file and delete extracted image files
            if bm.tarfile and os.path.exists(temp_dir):
                with tarfile.open(path_to_result, 'w') as tar:
                    for file_path in path_to_finals:
                        file_name = os.path.basename(file_path)
                        tar.add(file_path, arcname=file_name)
                bm.path_to_final = path_to_result
                bm.path_to_cropped_image = None

    # computation time
    t = int(time.time() - TIC)
    if t < 60:
        time_str = str(t) + " sec"
    elif 60 <= t < 3600:
        time_str = str(t // 60) + " min " + str(t % 60) + " sec"
    elif 3600 < t:
        time_str = str(t // 3600) + " h " + str((t % 3600) // 60) + " min " + str(t % 60) + " sec"
    print('Total calculation time:', time_str)

    # django environment
    if bm.django_env:
        from biomedisa_app.config import config
        from biomedisa.features.django_env import post_processing
        post_processing(bm.path_to_final, time_str, config['SERVER_ALIAS'], bm.remote, bm.queue,
            img_id=bm.img_id, label_id=bm.label_id, path_to_model=bm.path_to_model,
            path_to_cropped_image=(bm.path_to_cropped_image if crop_data else None),
            train=bm.train, predict=bm.predict)

        # write in log file
        path_to_time = biomedisa.BASE_DIR + '/log/time.txt'
        with open(path_to_time, 'a') as timefile:
            if predict:
                message = 'Successfully segmented ' + bm.shortfilename
            else:
                message = 'Successfully trained ' + project
            print('%s %s %s %s on %s' %(time.ctime(), bm.username, message, time_str, config['SERVER_ALIAS']), file=timefile)

    return results

if __name__ == '__main__':

    # initialize arguments
    parser = argparse.ArgumentParser(description='Biomedisa deeplearning.',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # mutually exclusive group
    g = parser.add_mutually_exclusive_group()

    # required arguments
    parser.add_argument('path_to_images', type=str, metavar='PATH_TO_IMAGES',
                        help='Location of image data (tarball, directory, or file)')
    parser.add_argument('path', type=str, metavar='PATH',
                        help='Location of label data during training (tarball, directory, or file) or model for prediction (h5)')

    # optional arguments
    g.add_argument('-p','--predict', action='store_true', default=False,
                        help='Automatic/predict segmentation')
    g.add_argument('-t','--train', action='store_true', default=False,
                        help='Train neural network')
    parser.add_argument('-v', '--version', action='version', version=f'{biomedisa.__version__}',
                        help='Biomedisa version')
    parser.add_argument('-b','--balance', action='store_true', default=False,
                        help='Balance foreground and background training patches')
    parser.add_argument('-cd','--crop_data', action='store_true', default=False,
                        help='Crop data automatically to region of interest')
    parser.add_argument('--acwe', action='store_true', default=False,
                        help='Post-processing with active contour')
    parser.add_argument('--acwe_alpha', metavar='ALPHA', type=float, default=1.0,
                        help='Pushing force of active contour')
    parser.add_argument('--acwe_smooth', metavar='SMOOTH', type=int, default=1,
                        help='Smoothing steps of active contour')
    parser.add_argument('--acwe_steps', metavar='STEPS', type=int, default=3,
                        help='Iterations of active contour')
    parser.add_argument('-c','--clean', nargs='?', type=float, const=0.1, default=None,
                        help='Remove outliers, e.g. 0.5 means that objects smaller than 50 percent of the size of the largest object will be removed')
    parser.add_argument('-f','--fill', nargs='?', type=float, const=0.9, default=None,
                        help='Fill holes, e.g. 0.5 means that all holes smaller than 50 percent of the entire label will be filled')
    parser.add_argument('--flip_x', action='store_true', default=False,
                        help='Randomly flip x-axis during training')
    parser.add_argument('--flip_y', action='store_true', default=False,
                        help='Randomly flip y-axis during training')
    parser.add_argument('--flip_z', action='store_true', default=False,
                        help='Randomly flip z-axis during training')
    parser.add_argument('-sa','--swapaxes', action='store_true', default=False,
                        help='Randomly swap two axes during training')
    parser.add_argument('-td','--train_dice', action='store_true', default=False,
                        help='Monitor Dice score on training data')
    parser.add_argument('-nvd','--no-val_dice', dest='val_dice', action='store_false',
                        help='Disable monitoring of Dice score on validation data')
    parser.add_argument('-dl','--dice_loss', action='store_true', default=False,
                        help='Dice loss function')
    parser.add_argument('-ad','--average_dice', action='store_true', default=False,
                        help='Use averaged dice score of each label')
    parser.add_argument('-nc', '--no-compression', dest='compression', action='store_false',
                        help='Disable compression of segmentation results')
    parser.add_argument('-i', '--ignore', type=str, default='none',
                        help='Ignore specific label(s), e.g. 2,5,6')
    parser.add_argument('-o', '--only', type=str, default='all',
                        help='Segment only specific label(s), e.g. 1,3,5')
    parser.add_argument('-nf', '--network_filters', type=str, default='32-64-128-256-512',
                        help='Number of filters per layer up to the deepest, e.g. 32-64-128-256-512')
    parser.add_argument('-rn','--resnet', action='store_true', default=False,
                        help='Use U-resnet instead of standard U-net')
    parser.add_argument('-dc','--debug_cropping', action='store_true', default=False,
                        help='Debug cropping')
    parser.add_argument('-sc','--save_cropped', action='store_true', default=False,
                        help='Save automatically cropped image')
    parser.add_argument('-e','--epochs', type=int, default=100,
                        help='Epochs the network is trained')
    parser.add_argument('-ce','--cropping_epochs', type=int, default=50,
                        help='Epochs the network for auto-cropping is trained')
    parser.add_argument('-nn','--no-normalization', dest='normalization', action='store_false',
                        help='Disable normalization of 3D image volumes')
    parser.add_argument('-r','--rotate', type=float, default=0.0,
                        help='Randomly rotate during training')
    parser.add_argument('-vs','--validation_split', type=float, default=0.0,
                        help='Percentage of data used for training')
    parser.add_argument('-lr','--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('-ss','--stride_size', metavar="[1-64]", type=int, choices=range(1,65), default=32,
                        help='Stride size for patches')
    parser.add_argument('-vss','--validation_stride_size', metavar="[1-64]", type=int, choices=range(1,65), default=32,
                        help='Stride size for validation patches')
    parser.add_argument('-vf','--validation_freq', type=int, default=1,
                        help='Epochs performed before validation')
    parser.add_argument('-bs','--batch_size', type=int, default=None,
                        help='Number of samples processed in a batch')
    parser.add_argument('-vi','--val_images', type=str, metavar='PATH', default=None,
                        help='Location of validation image data (tarball, directory, or file)')
    parser.add_argument('-vl','--val_labels', type=str, metavar='PATH', default=None,
                        help='Location of validation label data (tarball, directory, or file)')
    parser.add_argument('-xs','--x_scale', type=int, default=256,
                        help='Images and labels are scaled at x-axis to this size before training')
    parser.add_argument('-ys','--y_scale', type=int, default=256,
                        help='Images and labels are scaled at y-axis to this size before training')
    parser.add_argument('-zs','--z_scale', type=int, default=256,
                        help='Images and labels are scaled at z-axis to this size before training')
    parser.add_argument('-ns','--no-scaling', dest='scaling', action='store_false',
                        help='Do not resize image and label data')
    parser.add_argument('-es','--early_stopping', type=int, default=0,
                        help='Training is terminated when the accuracy has not increased in the epochs defined by this')
    parser.add_argument('-pm','--pretrained_model', type=str, metavar='PATH', default=None,
                        help='Location of pretrained model (only encoder will be trained if specified)')
    parser.add_argument('-ft','--fine_tune', action='store_true', default=False,
                        help='Fine-tune the entire pretrained model. Choose a smaller learning rate, e.g. 0.0001')
    parser.add_argument('-w','--workers', type=int, default=1,
                        help='Parallel workers for batch processing')
    parser.add_argument('-xr','--x_range', nargs="+", type=int, default=None,
                        help='Manually crop x-axis of image data for prediction, e.g. -xr 100 200')
    parser.add_argument('-yr','--y_range', nargs="+", type=int, default=None,
                        help='Manually crop y-axis of image data for prediction, e.g. -yr 100 200')
    parser.add_argument('-zr','--z_range', nargs="+", type=int, default=None,
                        help='Manually crop z-axis of image data for prediction, e.g. -zr 100 200')
    parser.add_argument('-rp','--return_probs', action='store_true', default=False,
                        help='Return prediction probabilities for each label')
    parser.add_argument('-pn','--patch_normalization', action='store_true', default=False,
                        help='Scale each patch to mean zero and standard deviation')
    parser.add_argument('-xp','--x_patch', type=int, default=64,
                        help='X-dimension of patch')
    parser.add_argument('-yp','--y_patch', type=int, default=64,
                        help='Y-dimension of patch')
    parser.add_argument('-zp','--z_patch', type=int, default=64,
                        help='Z-dimension of patch')
    parser.add_argument('-iid','--img_id', type=str, default=None,
                        help='Image ID within django environment/browser version')
    parser.add_argument('-lid','--label_id', type=str, default=None,
                        help='Label ID within django environment/browser version')
    parser.add_argument('-re','--remote', action='store_true', default=False,
                        help='The interpolation is carried out on a remote server. Must be set up in config.py')
    parser.add_argument('-q','--queue', type=int, default=0,
                        help='Processing queue when using a remote server')
    parser.add_argument('-hf','--header_file', type=str, metavar='PATH', default=None,
                        help='Location of header file')
    parser.add_argument('-s','--separation', action='store_true', default=False,
                        help='Instance segmentation of objects such as cells or rock particles')
    parser.add_argument('-m','--mask', type=str, metavar='PATH', default=None,
                        help='Location of mask')
    parser.add_argument('-rf','--refinement', action='store_true', default=False,
                        help='Refine segmentation on full size data')
    parser.add_argument('-ext','--extension', type=str, default='.tif',
                        help='Save data for example as NRRD file using --extension=".nrrd"')
    bm = parser.parse_args()
    bm.success = True

    # prediction or training
    if not any([bm.train, bm.predict]):
        bm.predict = False
        bm.train = True
        if os.path.splitext(bm.path)[1] == '.h5':
            bm.predict = True
            bm.train = False
    if bm.predict:
        bm.path_to_labels = None
        bm.path_to_model = bm.path
    if bm.train:
        bm.path_to_labels = bm.path
        bm.path_to_model = bm.path_to_images + '.h5'

    # django environment
    if bm.img_id is not None:
        bm.django_env = True
        if bm.train:
            reference_image_path = bm.path_to_images.split(',')[:-1][-1]
        else:
            reference_image_path = bm.path_to_images
        bm.username = os.path.basename(os.path.dirname(reference_image_path))
        bm.shortfilename = os.path.basename(reference_image_path)
        bm.path_to_logfile = biomedisa.BASE_DIR + '/log/logfile.txt'
    else:
        bm.django_env = False

    kwargs = vars(bm)

    # train or predict segmentation
    try:
        deep_learning(None, **kwargs)
    except InputError:
        print(traceback.format_exc())
        bm = _error_(bm, f'{InputError.message}')
    except ch.InputError:
        print(traceback.format_exc())
        bm = _error_(bm, f'{ch.InputError.message}')
    except MemoryError:
        print(traceback.format_exc())
        bm = _error_(bm, 'MemoryError')
    except ResourceExhaustedError:
        print(traceback.format_exc())
        bm = _error_(bm, 'GPU out of memory. Reduce your batch size')
    except Exception as e:
        print(traceback.format_exc())
        bm = _error_(bm, e)

