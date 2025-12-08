##########################################################################
##                                                                      ##
##  Copyright (c) 2019-2025 Philipp Lösel. All rights reserved.         ##
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
from keras.optimizers import SGD, Adam
from keras.models import Model, load_model
from keras.layers import (
    Input, Conv3D, MaxPooling3D, UpSampling3D, Activation, Reshape,
    BatchNormalization, Concatenate, ReLU, Add, GlobalAveragePooling3D,
    Dense, Dropout, MaxPool3D, Flatten, Multiply)
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from biomedisa.features.DataGenerator import DataGenerator, welford_mean_std
from biomedisa.features.PredictDataGenerator import PredictDataGenerator
from biomedisa.features.biomedisa_helper import (unique, welford_mean_std,
    img_resize, load_data, save_data, set_labels_to_zero, id_generator, unique_file_path)
from biomedisa.features.remove_outlier import clean, fill
from biomedisa.features.active_contour import activeContour
from tifffile import TiffFile, imread, imwrite
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import cv2
import tarfile
from random import shuffle
import glob
import random
import numba
import re
import time
import h5py
import tempfile
import csv

class InputError(Exception):
    def __init__(self, message=None):
        self.message = message

def save_csv(path, history):
    # remove empty keys
    to_delete = []
    for key in history.keys():
        if len(history[key])==0:
            to_delete.append(key)
    for key in to_delete:
        del history[key]
    # open a file in write mode
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=history.keys())
        # write header
        writer.writeheader()
        # write data rows (use zip to iterate over values)
        for row in zip(*history.values()):
            writer.writerow(dict(zip(history.keys(), row)))

def save_history(history, path_to_model, validation_freq):
    # xticks
    x_labels = []
    for epoch in range(len(history['accuracy'])):
        x_labels.append(str(epoch * validation_freq + 1))
    x = np.arange(len(x_labels))
    step = len(x)//10 + 1
    tick_indices = x[::step]
    tick_labels = [x_labels[i] for i in tick_indices]
    plt.xticks(tick_indices, tick_labels)
    # standard accuracy history
    plt.plot(history['accuracy'])
    legend = ['Accuracy (train)']
    if 'val_accuracy' in history and len(history['val_accuracy'])>0:
        plt.plot(history['val_accuracy'])
        legend.append('Accuracy (test)')
    # dice history
    if 'dice' in history and len(history['dice'])>0:
        plt.plot(history['dice'])
        legend.append('Dice score (train)')
    if 'val_dice' in history and len(history['val_dice'])>0:
        plt.plot(history['val_dice'])
        legend.append('Dice score (test)')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(legend, loc='upper left')
    plt.tight_layout()  # To prevent overlapping of subplots
    plt.savefig(path_to_model.replace('.h5','_acc.png'), dpi=300, bbox_inches='tight')
    plt.clf()
    # loss history
    plt.plot(history['loss'])
    legend = ['train']
    if 'val_loss' in history and len(history['val_loss'])>0:
        plt.plot(history['val_loss'])
        legend.append('test')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(tick_indices, tick_labels)
    plt.legend(legend, loc='upper left')
    plt.tight_layout()  # To prevent overlapping of subplots
    plt.savefig(path_to_model.replace('.h5','_loss.png'), dpi=300, bbox_inches='tight')
    plt.clf()
    # save history as csv
    save_csv(path_to_model.replace('.h5','.csv'), history)

def predict_blocksize(labelData, x_puffer=25, y_puffer=25, z_puffer=25):
    zsh, ysh, xsh = labelData.shape
    argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = zsh, 0, ysh, 0, xsh, 0
    for k in range(zsh):
        y, x = np.nonzero(labelData[k])
        if x.any():
            argmin_x = min(argmin_x, np.amin(x))
            argmax_x = max(argmax_x, np.amax(x))
            argmin_y = min(argmin_y, np.amin(y))
            argmax_y = max(argmax_y, np.amax(y))
            argmin_z = min(argmin_z, k)
            argmax_z = max(argmax_z, k)
    zmin, zmax = argmin_z, argmax_z
    argmin_x = argmin_x - x_puffer if argmin_x - x_puffer > 0 else 0
    argmax_x = argmax_x + x_puffer if argmax_x + x_puffer < xsh else xsh
    argmin_y = argmin_y - y_puffer if argmin_y - y_puffer > 0 else 0
    argmax_y = argmax_y + y_puffer if argmax_y + y_puffer < ysh else ysh
    argmin_z = argmin_z - z_puffer if argmin_z - z_puffer > 0 else 0
    argmax_z = argmax_z + z_puffer if argmax_z + z_puffer < zsh else zsh
    return argmin_z,argmax_z,argmin_y,argmax_y,argmin_x,argmax_x

def set_image_dimensions(header, data):

    # read header as string
    b = header.tobytes()
    try:
        s = b.decode("utf-8")
    except:
        s = b.decode("latin1")

    # get image size in header
    lattice = re.search('define Lattice (.*)\n', s)
    lattice = lattice.group(1)
    xsh, ysh, zsh = lattice.split(' ')
    xsh, ysh, zsh = int(xsh), int(ysh), int(zsh)

    # new image size
    z,y,x = data.shape

    # change image size in header
    s = s.replace('%s %s %s' %(xsh,ysh,zsh), '%s %s %s' %(x,y,z),1)
    s = s.replace('Content "%sx%sx%s byte' %(xsh,ysh,zsh), 'Content "%sx%sx%s byte' %(x,y,z),1)

    # return header as array
    b2 = s.encode()
    new_header = np.frombuffer(b2, dtype=header.dtype)
    return new_header

def set_physical_size(header, img_header):

    # read img_header as string
    b = img_header.tobytes()
    try:
        s = b.decode("utf-8")
    except:
        s = b.decode("latin1")

    # get physical size from image header
    bounding_box = re.search('BoundingBox (.*),\n', s)
    bounding_box = bounding_box.group(1)
    i0, i1, i2, i3, i4, i5 = bounding_box.split(' ')
    bounding_box_i = re.search('&BoundingBox (.*),\n', s)
    bounding_box_i = bounding_box_i.group(1)

    # read label header as string
    b = header.tobytes()
    try:
        s = b.decode("utf-8")
    except:
        s = b.decode("latin1")

    # get physical size from label header
    bounding_box = re.search('BoundingBox (.*),\n', s)
    bounding_box = bounding_box.group(1)
    l0, l1, l2, l3, l4, l5 = bounding_box.split(' ')
    bounding_box_l = re.search('&BoundingBox (.*),\n', s)
    bounding_box_l = bounding_box_l.group(1)

    # change physical size in label header
    s = s.replace('%s %s %s %s %s %s' %(l0,l1,l2,l3,l4,l5),'%s %s %s %s %s %s' %(i0,i1,i2,i3,i4,i5),1)
    s = s.replace(bounding_box_l,bounding_box_i,1)

    # return header as array
    b2 = s.encode()
    new_header = np.frombuffer(b2, dtype=header.dtype)
    return new_header

@numba.jit(nopython=True)
def compute_position(position, zsh, ysh, xsh):
    zsh_h, ysh_h, xsh_h = zsh//2, ysh//2, xsh//2
    for k in range(zsh):
        for l in range(ysh):
            for m in range(xsh):
                x = (xsh_h-m)**2
                y = (ysh_h-l)**2
                z = (zsh_h-k)**2
                position[k,l,m] = x+y+z
    return position

def make_conv_block(nb_filters, input_tensor, block, dtype):
    def make_stage(input_tensor, stage):
        name = 'conv_{}_{}'.format(block, stage)
        x = Conv3D(nb_filters, (3, 3, 3), activation='relu',
                   padding='same', name=name, data_format="channels_last", dtype=dtype)(input_tensor)
        name = 'batch_norm_{}_{}'.format(block, stage)
        try:
            x = BatchNormalization(name=name, synchronized=True)(x)
        except:
            x = BatchNormalization(name=name)(x)
        x = Activation('relu')(x)
        return x

    x = make_stage(input_tensor, 1)
    x = make_stage(x, 2)
    return x

def make_conv_block_resnet(nb_filters, input_tensor, block, dtype):

    # Residual/Skip connection
    res = Conv3D(nb_filters, (1, 1, 1), padding='same',
        use_bias=False, name="Identity{}_1".format(block), dtype=dtype)(input_tensor)

    stage = 1
    name = 'conv_{}_{}'.format(block, stage)
    fx = Conv3D(nb_filters, (3, 3, 3), activation='relu',
        padding='same', name=name, data_format="channels_last", dtype=dtype)(input_tensor)
    name = 'batch_norm_{}_{}'.format(block, stage)
    try:
        fx = BatchNormalization(name=name, synchronized=True)(fx)
    except:
        fx = BatchNormalization(name=name)(fx)
    fx = Activation('relu')(fx)

    stage = 2
    name = 'conv_{}_{}'.format(block, stage)
    fx = Conv3D(nb_filters, (3, 3, 3), padding='same', name=name, data_format="channels_last", dtype=dtype)(fx)
    name = 'batch_norm_{}_{}'.format(block, stage)
    try:
        fx = BatchNormalization(name=name, synchronized=True)(fx)
    except:
        fx = BatchNormalization(name=name)(fx)

    out = Add()([res,fx])
    out = ReLU()(out)

    return out

def make_unet(bm, input_shape, nb_labels):
    # enable mixed_precision
    if bm.mixed_precision:
        dtype = "float16"
    else:
        dtype = "float32"

    # input
    nb_plans, nb_rows, nb_cols, _ = input_shape
    inputs = Input(input_shape, dtype=dtype)

    # configure number of layers and filters
    filters = bm.network_filters.split('-')
    filters = np.array(filters, dtype=int)
    latent_space_size = filters[-1]
    filters = filters[:-1]

    # initialize blocks
    convs = []

    # encoder
    i = 1
    for f in filters:
        if i==1:
            if bm.resnet:
                conv = make_conv_block_resnet(f, inputs, i, dtype)
            else:
                conv = make_conv_block(f, inputs, i, dtype)
        else:
            if bm.resnet:
                conv = make_conv_block_resnet(f, pool, i, dtype)
            else:
                conv = make_conv_block(f, pool, i, dtype)
        pool = MaxPooling3D(pool_size=(2, 2, 2))(conv)
        convs.append(conv)
        i += 1

    # latent space
    if bm.resnet:
        conv = make_conv_block_resnet(latent_space_size, pool, i, dtype)
    else:
        conv = make_conv_block(latent_space_size, pool, i, dtype)
    i += 1

    # decoder
    for k, f in enumerate(filters[::-1]):
        up = Concatenate()([UpSampling3D(size=(2, 2, 2))(conv), convs[-(k+1)]])
        if bm.resnet:
            conv = make_conv_block_resnet(f, up, i, dtype)
        else:
            conv = make_conv_block(f, up, i, dtype)
        i += 1

    # final layer and output
    conv = Conv3D(nb_labels, (1, 1, 1), name=f'conv_{i}_1')(conv)
    x = Reshape((nb_plans * nb_rows * nb_cols, nb_labels))(conv)
    x = Activation('softmax')(x)
    outputs = Reshape((nb_plans, nb_rows, nb_cols, nb_labels))(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def get_labels(arr, allLabels):
    final = np.zeros_like(arr)
    for k in unique(arr):
        final[arr == k] = allLabels[k]
    return final

def read_img_list(img_list, label_list, temp_img_dir, temp_label_dir):
    # read filenames
    img_names, label_names = [], []
    for img_name, label_name in zip(img_list, label_list):

        # check for tarball
        img_dir, img_ext = os.path.splitext(img_name)
        if img_ext == '.gz':
            img_ext = os.path.splitext(img_dir)[1]

        label_dir, label_ext = os.path.splitext(label_name)
        if label_ext == '.gz':
            label_ext = os.path.splitext(label_dir)[1]

        if (img_ext == '.tar' and label_ext == '.tar') or (os.path.isdir(img_name) and os.path.isdir(label_name)):

            # extract files
            if img_ext == '.tar':
                tar = tarfile.open(img_name)
                tar.extractall(path=temp_img_dir)
                tar.close()
                img_name = temp_img_dir
            if label_ext == '.tar':
                tar = tarfile.open(label_name)
                tar.extractall(path=temp_label_dir)
                tar.close()
                label_name = temp_label_dir

            for data_type in ['.am','.tif','.tiff','.hdr','.mhd','.mha','.nrrd','.nii','.nii.gz','.zip','.mrc']:
                img_names += [file for file in glob.glob(img_name+'/**/*'+data_type, recursive=True) if not os.path.basename(file).startswith('.')]
                label_names += [file for file in glob.glob(label_name+'/**/*'+data_type, recursive=True) if not os.path.basename(file).startswith('.')]
            img_names = sorted(img_names)
            label_names = sorted(label_names)
            if len(img_names)==0 or len(label_names)==0 or len(img_names)!=len(label_names):
                if len(img_names)!=len(label_names):
                    InputError.message = 'Number of image and label files must be the same'
                elif img_ext == '.tar' and len(img_names)==0:
                    InputError.message = 'Invalid image TAR file'
                elif label_ext == '.tar' and len(label_names)==0:
                    InputError.message = 'Invalid label TAR file'
                elif len(img_names)==0:
                    InputError.message = 'Invalid image data'
                else:
                    InputError.message = 'Invalid label data'
                raise InputError()
        else:
            img_names.append(img_name)
            label_names.append(label_name)
    return img_names, label_names

def load_training_data(bm, img_list, label_list, channels, img_in=None, label_in=None,
        normalization_parameters=None, allLabels=None, header=None, extension='.tif',
        x_puffer=25, y_puffer=25, z_puffer=25):

    # make temporary directories
    with tempfile.TemporaryDirectory() as temp_img_dir:
        with tempfile.TemporaryDirectory() as temp_label_dir:

            # read image lists
            if any(img_list):
                img_names, label_names = read_img_list(img_list, label_list, temp_img_dir, temp_label_dir)

            # load first label
            if any(img_list):
                label, header, extension = load_data(label_names[0], 'first_queue', True)
                if label is None:
                    InputError.message = f'Invalid label data "{os.path.basename(label_names[0])}"'
                    raise InputError()
            elif type(label_in) is list:
                label = label_in[0]
                label_names = [f'label_{i}' for i in range(1, len(label_in) + 1)]
            else:
                label = label_in
                label_names = ['label_1']
            label_dim = label.shape
            label = set_labels_to_zero(label, bm.only, bm.ignore)
            if any([bm.x_range, bm.y_range, bm.z_range]):
                if len(label_names)>1:
                    InputError.message = 'Training on region of interest is only supported for one volume.'
                    raise InputError()
                argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = \
                    bm.z_range[0], bm.z_range[1], bm.y_range[0], bm.y_range[1], bm.x_range[0], bm.x_range[1]
                label = label[argmin_z:argmax_z,argmin_y:argmax_y,argmin_x:argmax_x].copy()
            elif bm.crop_data:
                argmin_z,argmax_z,argmin_y,argmax_y,argmin_x,argmax_x = predict_blocksize(label, x_puffer, y_puffer, z_puffer)
                label = label[argmin_z:argmax_z,argmin_y:argmax_y,argmin_x:argmax_x].copy()
            if bm.scaling:
                label_values, counts = unique(label, return_counts=True)
                print(f'{os.path.basename(label_names[0])}:', 'Labels:', label_values[1:], 'Sizes:', counts[1:])
                label = img_resize(label, bm.z_scale, bm.y_scale, bm.x_scale, labels=True)

            # label channel must be 1 or 2 if using ignore mask
            if len(label.shape)>3 and label.shape[3]>1 and not bm.ignore_mask:
                InputError.message = 'Training labels must have one channel (gray values).'
                raise InputError()
            if len(label.shape)==3:
                label = label.reshape(label.shape[0], label.shape[1], label.shape[2], 1)

            # if header is not single data stream Amira Mesh falling back to Multi-TIFF
            if extension != '.am':
                extension, header = '.tif', None
            elif len(header) > 1:
                print('Warning! Multiple data streams are not supported. Falling back to TIFF.')
                extension, header = '.tif', None
            else:
                header = header[0]

            # load first image
            if any(img_list):
                img, _ = load_data(img_names[0], 'first_queue')
                if img is None:
                    InputError.message = f'Invalid image data "{os.path.basename(img_names[0])}"'
                    raise InputError()
            elif type(img_in) is list:
                img = img_in[0]
                img_names = [f'img_{i}' for i in range(1, len(img_in) + 1)]
            else:
                img = img_in
                img_names = ['img_1']

            # label and image dimensions must match
            if label_dim[:3] != img.shape[:3]:
                InputError.message = f'Dimensions of "{os.path.basename(img_names[0])}" and "{os.path.basename(label_names[0])}" do not match'
                raise InputError()

            # image channels must be >=1
            if len(img.shape)==3:
                img = img.reshape(img.shape[0], img.shape[1], img.shape[2], 1)
            if channels is None:
                channels = img.shape[3]
            if channels != img.shape[3]:
                InputError.message = f'Number of channels must be {channels} for "{os.path.basename(img_names[0])}"'
                raise InputError()

            # crop data
            if any([bm.x_range, bm.y_range, bm.z_range]) or bm.crop_data:
                img = img[argmin_z:argmax_z,argmin_y:argmax_y,argmin_x:argmax_x].copy()

            # resize image data
            if bm.scaling:
                img = img.astype(np.float32)
                img = img_resize(img, bm.z_scale, bm.y_scale, bm.x_scale)

            # scale data to the range from 0 to 1
            if not bm.patch_normalization:
                img = img.astype(np.float32)
                for ch in range(channels):
                    img[...,ch] -= np.amin(img[...,ch])
                    img[...,ch] /= np.amax(img[...,ch])

            # normalize first validation image
            if bm.normalize and np.any(normalization_parameters):
                img = img.astype(np.float32)
                for ch in range(channels):
                    mean, std = welford_mean_std(img[...,ch])
                    img[...,ch] = (img[...,ch] - mean) / std
                    img[...,ch] = img[...,ch] * normalization_parameters[1,ch] + normalization_parameters[0,ch]

            # get normalization parameters from first image
            if normalization_parameters is None:
                normalization_parameters = np.zeros((2,channels))
                if bm.normalize:
                    for ch in range(channels):
                        normalization_parameters[:,ch] = welford_mean_std(img[...,ch])

            # pad data
            if not bm.scaling:
                img_data_list = [img]
                label_data_list = [label]
                img_dtype = img.dtype
                # no-scaling for list of images needs negative values as it encodes padded areas as -1
                label_dtype = label.dtype
                if label_dtype==np.uint8:
                    label_dtype = np.int16
                elif label_dtype in [np.uint16, np.uint32]:
                    label_dtype = np.int32

            # loop over list of images
            if any(img_list) or type(img_in) is list:
                number_of_images = len(img_names) if any(img_list) else len(img_in)

                for k in range(1, number_of_images):

                    # load label data and pre-process
                    if any(label_list):
                        a, _ = load_data(label_names[k], 'first_queue')
                        if a is None:
                            InputError.message = f'Invalid label data "{os.path.basename(label_names[k])}"'
                            raise InputError()
                    else:
                        a = label_in[k]
                    label_dim = a.shape
                    a = set_labels_to_zero(a, bm.only, bm.ignore)
                    if bm.crop_data:
                        argmin_z,argmax_z,argmin_y,argmax_y,argmin_x,argmax_x = predict_blocksize(a, x_puffer, y_puffer, z_puffer)
                        a = np.copy(a[argmin_z:argmax_z,argmin_y:argmax_y,argmin_x:argmax_x], order='C')
                    if bm.scaling:
                        label_values, counts = unique(a, return_counts=True)
                        print(f'{os.path.basename(label_names[k])}:', 'Labels:', label_values[1:], 'Sizes:', counts[1:])
                        a = img_resize(a, bm.z_scale, bm.y_scale, bm.x_scale, labels=True)

                    # label channel must be 1 or 2 if using ignore mask
                    if len(a.shape)>3 and a.shape[3]>1 and not bm.ignore_mask:
                        InputError.message = 'Training labels must have one channel (gray values).'
                        raise InputError()
                    if len(a.shape)==3:
                        a = a.reshape(a.shape[0], a.shape[1], a.shape[2], 1)

                    # append label data
                    if bm.scaling:
                        label = np.append(label, a, axis=0)
                    else:
                        label_data_list.append(a)

                    # load image data and pre-process
                    if any(img_list):
                        a, _ = load_data(img_names[k], 'first_queue')
                        if a is None:
                            InputError.message = f'Invalid image data "{os.path.basename(img_names[k])}"'
                            raise InputError()
                    else:
                        a = img_in[k]
                    if label_dim[:3] != a.shape[:3]:
                        InputError.message = f'Dimensions of "{os.path.basename(img_names[k])}" and "{os.path.basename(label_names[k])}" do not match'
                        raise InputError()
                    if len(a.shape)==3:
                        a = a.reshape(a.shape[0], a.shape[1], a.shape[2], 1)
                    if a.shape[3] != channels:
                        InputError.message = f'Number of channels must be {channels} for "{os.path.basename(img_names[k])}"'
                        raise InputError()
                    if bm.crop_data:
                        a = np.copy(a[argmin_z:argmax_z,argmin_y:argmax_y,argmin_x:argmax_x], order='C')
                    if bm.scaling:
                        a = a.astype(np.float32)
                        a = img_resize(a, bm.z_scale, bm.y_scale, bm.x_scale)
                    if not bm.patch_normalization:
                        a = a.astype(np.float32)
                        for ch in range(channels):
                            a[...,ch] -= np.amin(a[...,ch])
                            a[...,ch] /= np.amax(a[...,ch])
                    if bm.normalize:
                        a = a.astype(np.float32)
                        for ch in range(channels):
                            mean, std = welford_mean_std(a[...,ch])
                            a[...,ch] = (a[...,ch] - mean) / std
                            a[...,ch] = a[...,ch] * normalization_parameters[1,ch] + normalization_parameters[0,ch]

                    # append image data
                    if bm.scaling:
                        img = np.append(img, a, axis=0)
                    else:
                        img_data_list.append(a)

    # pad and append data to a single volume
    if not bm.scaling and len(img_data_list)>1:
        target_y, target_x = 0, 0
        for img in img_data_list:
            target_y = max(target_y, img.shape[1])
            target_x = max(target_x, img.shape[2])
        img = np.empty((0, target_y, target_x, channels), dtype=img_dtype)
        label = np.empty((0, target_y, target_x, 2 if bm.ignore_mask else 1), dtype=label_dtype)
        for k in range(len(img_data_list)):
            pad_y = target_y - img_data_list[k].shape[1]
            pad_x = target_x - img_data_list[k].shape[2]
            pad_width = [(0, 0), (0, pad_y), (0, pad_x), (0, 0)]
            tmp = np.pad(img_data_list[k], pad_width, mode='constant', constant_values=0)
            img = np.append(img, tmp, axis=0)
            tmp = np.pad(label_data_list[k].astype(label_dtype), pad_width, mode='constant', constant_values=-1)
            label = np.append(label, tmp, axis=0)

    # limit intensity range
    if bm.normalize:
        img[img<0] = 0
        img[img>1] = 1

    if bm.separation:
        allLabels = np.array([0,1])
    else:
        # get labels
        if allLabels is None:
            allLabels = unique(label[...,0])
            index = np.argwhere(allLabels<0)
            allLabels = np.delete(allLabels, index)

        # labels must be in ascending order
        for k, l in enumerate(allLabels):
            label[...,0][label[...,0]==l] = k

    return img, label, allLabels, normalization_parameters, header, extension, channels

class CustomCallback(Callback):
    def __init__(self, id, epochs):
        self.epochs = epochs
        self.id = id

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        import django
        django.setup()
        from biomedisa_app.models import Upload
        image = Upload.objects.get(pk=self.id)
        if image.status == 3:
            self.model.stop_training = True
        else:
            keys = list(logs.keys())
            percentage = round((int(epoch)+1)*100/float(self.epochs))
            t = round(time.time() - self.epoch_time_start) * (self.epochs-int(epoch)-1)
            if t < 3600:
                time_remaining = str(t // 60) + 'min'
            else:
                time_remaining = str(t // 3600) + 'h ' + str((t % 3600) // 60) + 'min'
            image.message = 'Progress {}%, {} remaining'.format(percentage,time_remaining)
            if 'best_val_dice' in logs:
                best_val_dice = round(float(logs['best_val_dice'])*100,1)
                image.message += f', {best_val_dice}% accuracy'
            image.save()
            print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

class MetaData(Callback):
    def __init__(self, bm, path_to_model, configuration_data, allLabels,
        extension, header, crop_data, cropping_weights, cropping_config,
        normalization_parameters, cropping_norm, patch_normalization, scaling):

        self.path_to_model = path_to_model
        self.configuration_data = configuration_data
        self.normalization_parameters = normalization_parameters
        self.allLabels = allLabels
        self.extension = extension
        self.header = header
        self.crop_data = crop_data
        self.cropping_weights = cropping_weights
        self.cropping_config = cropping_config
        self.cropping_norm = cropping_norm
        self.patch_normalization = patch_normalization
        self.scaling = scaling
        self.separation = bm.separation
        self.patch_size = np.array([bm.z_patch, bm.y_patch, bm.x_patch])

    def on_epoch_end(self, epoch, logs={}):
        hf = h5py.File(self.path_to_model, 'r')
        if not '/meta' in hf:
            hf.close()
            hf = h5py.File(self.path_to_model, 'r+')
            group = hf.create_group('meta')
            group.create_dataset('configuration', data=self.configuration_data)
            group.create_dataset('normalization', data=self.normalization_parameters)
            group.create_dataset('labels', data=self.allLabels)
            group.create_dataset('patch_normalization', data=int(self.patch_normalization))
            group.create_dataset('scaling', data=int(self.scaling))
            group.create_dataset('separation', data=int(self.separation))
            group.create_dataset('patch_size', data=self.patch_size)
            if self.extension == '.am':
                group.create_dataset('extension', data=self.extension)
                group.create_dataset('header', data=self.header)
            if self.crop_data:
                cm_group = hf.create_group('cropping_meta')
                cm_group.create_dataset('configuration', data=self.cropping_config)
                cm_group.create_dataset('normalization', data=self.cropping_norm)
                cw_group = hf.create_group('cropping_weights')
                for iterator, arr in enumerate(self.cropping_weights):
                    cw_group.create_dataset(str(iterator), data=arr)
        hf.close()

class Metrics(Callback):
    def __init__(self, bm, img, label, list_IDs, dim_img, n_classes, train):
        self.dim_patch = (bm.z_patch, bm.y_patch, bm.x_patch)
        self.dim_img = dim_img
        self.list_IDs = list_IDs
        self.batch_size = bm.batch_size
        self.label = label
        self.img = img
        self.path_to_model = bm.path_to_model
        self.early_stopping = bm.early_stopping
        self.validation_freq = bm.validation_freq
        self.n_classes = n_classes
        self.n_channels = bm.channels
        self.average_dice = bm.average_dice
        self.django_env = bm.django_env
        self.patch_normalization = bm.patch_normalization
        self.train = train
        self.train_dice = bm.train_dice

    def on_train_begin(self, logs={}):
        self.history = {}
        self.history['val_accuracy'] = []
        self.history['accuracy'] = []
        self.history['val_dice'] = []
        self.history['dice'] = []
        self.history['val_loss'] = []
        self.history['loss'] = []

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.validation_freq == 0:

            result = np.zeros((*self.dim_img, self.n_classes), dtype=np.float32)

            len_IDs = len(self.list_IDs)
            n_batches = int(np.floor(len_IDs / self.batch_size))
            np.random.shuffle(self.list_IDs)

            # initialize validation loss
            val_loss = 0

            for batch in range(n_batches):
                # Generate indexes of the batch
                list_IDs_batch = self.list_IDs[batch*self.batch_size:(batch+1)*self.batch_size]

                # Initialization
                X_val = np.empty((self.batch_size, *self.dim_patch, self.n_channels), dtype=np.float32)

                # Generate data
                for i, ID in enumerate(list_IDs_batch):

                    # get patch indices
                    k = ID // (self.dim_img[1]*self.dim_img[2])
                    rest = ID % (self.dim_img[1]*self.dim_img[2])
                    l = rest // self.dim_img[2]
                    m = rest % self.dim_img[2]
                    tmp_X = self.img[k:k+self.dim_patch[0],l:l+self.dim_patch[1],m:m+self.dim_patch[2]]
                    if self.patch_normalization:
                        tmp_X = tmp_X.copy().astype(np.float32)
                        for ch in range(self.n_channels):
                            mean, std = welford_mean_std(tmp_X[...,ch])
                            tmp_X[...,ch] -= mean
                            tmp_X[...,ch] /= max(std, 1e-6)
                    X_val[i] = tmp_X

                # Prediction segmentation
                y_predict = np.asarray(self.model.predict(X_val, verbose=0, steps=None, batch_size=self.batch_size))

                for i, ID in enumerate(list_IDs_batch):

                    # get patch indices
                    k = ID // (self.dim_img[1]*self.dim_img[2])
                    rest = ID % (self.dim_img[1]*self.dim_img[2])
                    l = rest // self.dim_img[2]
                    m = rest % self.dim_img[2]
                    result[k:k+self.dim_patch[0],l:l+self.dim_patch[1],m:m+self.dim_patch[2]] += y_predict[i]

                    # calculate validation loss
                    if not self.train:
                        val_loss += categorical_crossentropy(self.label[k:k+self.dim_patch[0],l:l+self.dim_patch[1],m:m+self.dim_patch[2]], y_predict[i])

            # mean validation loss
            val_loss /= (n_batches*self.batch_size)

            # get result
            result = np.argmax(result, axis=-1)
            result = result.astype(np.uint8)
            result = result.reshape(*result.shape, 1)

            # calculate standard accuracy
            if not self.train:
                accuracy = np.sum(self.label==result) / float(self.label.size)

            # calculate dice score
            if self.average_dice:
                dice = 0
                for l in range(1, self.n_classes):
                    dice += 2 * np.logical_and(self.label==l, result==l).sum() / float((self.label==l).sum() + (result==l).sum())
                dice /= float(self.n_classes-1)
            else:
                dice = 2 * np.logical_and(self.label==result, (self.label+result)>0).sum() / \
                       float((self.label>0).sum() + (result>0).sum())

            if self.train:
                logs['dice'] = dice
            else:
                # save best model only
                if epoch == 0 or dice > max(self.history['val_dice']):
                    self.model.save(str(self.path_to_model))

                # add accuracy to history
                self.history['loss'].append(logs['loss'])
                self.history['accuracy'].append(logs['accuracy'])
                if self.train_dice:
                    self.history['dice'].append(logs['dice'])
                self.history['val_accuracy'].append(accuracy)
                self.history['val_dice'].append(dice)
                self.history['val_loss'].append(val_loss)

                # monitoring variables
                logs['val_loss'] = val_loss
                logs['val_accuracy'] = accuracy
                logs['val_dice'] = dice
                logs['best_acc'] = max(self.history['accuracy'])
                if self.train_dice:
                    logs['best_dice'] = max(self.history['dice'])
                logs['best_val_acc'] = max(self.history['val_accuracy'])
                logs['best_val_dice'] = max(self.history['val_dice'])

                # plot history in figure and save as numpy array
                save_history(self.history, self.path_to_model, self.validation_freq)

                # print accuracies
                print('\nValidation history:')
                print("train_acc: [" + " ".join(f"{x:.4f}" for x in self.history['accuracy']) + "]")
                if self.train_dice:
                    print("train_dice: [" + " ".join(f"{x:.4f}" for x in self.history['dice']) + "]")
                print("val_acc: [" + " ".join(f"{x:.4f}" for x in self.history['val_accuracy']) + "]")
                print("val_dice: [" + " ".join(f"{x:.4f}" for x in self.history['val_dice']) + "]")
                print('')

                # early stopping
                if self.early_stopping > 0 and max(self.history['val_dice']) not in self.history['val_dice'][-self.early_stopping:]:
                    self.model.stop_training = True

class HistoryCallback(Callback):
    def __init__(self, bm):
        self.path_to_model = bm.path_to_model
        self.train_dice = bm.train_dice
        self.val_img_data = bm.val_img_data
        self.validation_freq = bm.validation_freq

    def on_train_begin(self, logs={}):
        self.history = {}
        self.history['loss'] = []
        self.history['accuracy'] = []
        if self.train_dice:
            self.history['dice'] = []
        if self.val_img_data is not None:
            self.history['val_loss'] = []
            self.history['val_accuracy'] = []

    def on_epoch_end(self, epoch, logs={}):
        # append history
        self.history['loss'].append(logs['loss'])
        self.history['accuracy'].append(logs['accuracy'])
        if self.train_dice:
            self.history['dice'].append(logs['dice'])
        if self.val_img_data is not None:
            self.history['val_loss'].append(logs['val_loss'])
            self.history['val_accuracy'].append(logs['val_accuracy'])

        # plot history in figure and save as numpy array
        save_history(self.history, self.path_to_model, self.validation_freq)

def softmax(x):
    # Avoiding numerical instability by subtracting the maximum value
    exp_values = np.exp(x - np.max(x, axis=-1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
    return probabilities

@numba.jit(nopython=True)
def categorical_crossentropy(true_labels, predicted_probs):
    # Clip predicted probabilities to avoid log(0) issues
    predicted_probs = np.clip(predicted_probs, 1e-7, 1 - 1e-7)
    predicted_probs = -np.log(predicted_probs)
    zsh, ysh, xsh, _ = true_labels.shape
    # Calculate categorical crossentropy
    loss = 0
    for z in range(zsh):
        for y in range(ysh):
            for x in range(xsh):
                l = true_labels[z,y,x,0]
                loss += predicted_probs[z,y,x,l]
    loss = loss / float(zsh*ysh*xsh)
    return loss

def dice_coef(y_true, y_pred, smooth=1e-5):
    intersection = K.sum(Multiply()([y_true, y_pred]))
    dice = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    return dice

def dice_coef_loss(nb_labels):
    #y_pred_f = K.argmax(y_pred, axis=-1)
    #y_pred_f = K.cast(y_pred_f,'float32')
    #dice_coef(y_true[:,:,:,:,1], y_pred[:,:,:,:,1] * y_pred_f)
    def loss_fn(y_true, y_pred):
        dice = 0
        for index in range(1,nb_labels):
            dice += dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
        dice = dice / (nb_labels-1)
        #loss = -K.log(dice)
        loss = 1 - dice
        return loss
    return loss_fn

def loss_fn(y_true, y_pred):
    dice = 0
    for index in range(1,nb_labels):
        dice += dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    dice = dice / (nb_labels-1)
    #loss = -K.log(dice)
    loss = 1 - dice
    return loss

def custom_loss(y_true, y_pred):
    import tensorflow as tf #TODO: use keras backend
    # Extract labels and ignore mask
    labels = tf.cast(y_true[..., 0], tf.int32)  # First channel contains class labels
    ignore_mask = tf.cast(y_true[..., 1], tf.float32)  # Second channel contains mask (0 = ignore, 1 = include)

    # Convert integer labels to one-hot encoding
    y_true_one_hot = tf.one_hot(labels, depth=2)

    # Clip y_pred to avoid log(0)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

    # Compute categorical cross-entropy
    loss = -tf.reduce_sum(y_true_one_hot * tf.math.log(y_pred), axis=-1)

    # Apply ignore mask (ignore = 0 → loss is zero, include = 1 → loss is counted)
    loss = loss * ignore_mask

    # Return mean loss over valid (non-ignored) samples
    return tf.reduce_sum(loss) / tf.reduce_sum(ignore_mask)

def custom_accuracy(y_true, y_pred):
    import tensorflow as tf
    labels = tf.cast(y_true[..., 0], tf.int32)  # Extract actual values
    ignore_mask = y_true[..., 1]  # Extract mask (1 = include, 0 = ignore)

    # Convert predictions to discrete values (assuming regression: round values)
    y_pred_class = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    # Compute correct predictions (1 where correct, 0 where incorrect)
    correct_predictions = tf.cast(tf.equal(labels, y_pred_class), tf.float32)

    # Apply ignore mask
    masked_correct_predictions = correct_predictions * ignore_mask

    # Compute accuracy only over valid (non-ignored) pixels
    return tf.reduce_sum(masked_correct_predictions) / tf.reduce_sum(ignore_mask)

def train_segmentation(bm):

    # training data
    bm.img_data, bm.label_data, allLabels, normalization_parameters, bm.header, bm.extension, bm.channels = load_training_data(bm,
        bm.path_to_images, bm.path_to_labels, None, bm.img_data, bm.label_data, None, None, bm.header, bm.extension)

    # configuration data
    configuration_data = np.array([bm.channels,
        bm.x_scale, bm.y_scale, bm.z_scale, bm.normalize,
        normalization_parameters[0,0], normalization_parameters[1,0]])

    # img shape
    zsh, ysh, xsh, _ = bm.img_data.shape

    # validation data
    if any(bm.val_images) or bm.val_img_data is not None:
        bm.val_img_data, bm.val_label_data, _, _, _, _, _ = load_training_data(bm,
            bm.val_images, bm.val_labels, bm.channels, bm.val_img_data, bm.val_label_data, normalization_parameters, allLabels)

    elif bm.validation_split:
        split = round(zsh * bm.validation_split)
        bm.val_img_data = bm.img_data[split:].copy()
        bm.val_label_data = bm.label_data[split:].copy()
        bm.img_data = bm.img_data[:split].copy()
        bm.label_data = bm.label_data[:split].copy()
        zsh, ysh, xsh, _ = bm.img_data.shape

    # list of IDs
    list_IDs_fg, list_IDs_bg = [], []

    # get IDs of patches
    for k in range(0, zsh-bm.z_patch+1, bm.stride_size):
        for l in range(0, ysh-bm.y_patch+1, bm.stride_size):
            for m in range(0, xsh-bm.x_patch+1, bm.stride_size):
                patch = bm.label_data[k:k+bm.z_patch, l:l+bm.y_patch, m:m+bm.x_patch]
                index = k*ysh*xsh+l*xsh+m
                if not np.any(patch==-1): # ignore padded areas
                    if bm.balance:
                        if bm.separation:
                            centerLabel = bm.label_data[k+bm.z_patch//2,l+bm.y_patch//2,m+bm.x_patch//2]
                            if centerLabel>0 and np.any(np.logical_and(patch!=centerLabel, patch>0)):
                                list_IDs_fg.append(index)
                            elif centerLabel>0 and np.any(patch!=centerLabel):
                                list_IDs_bg.append(index)
                        elif np.any(patch>0):
                            list_IDs_fg.append(index)
                        else:
                            list_IDs_bg.append(index)
                    else:
                        if bm.separation:
                            centerLabel = bm.label_data[k+bm.z_patch//2,l+bm.y_patch//2,m+bm.x_patch//2]
                            if centerLabel>0 and np.any(patch!=centerLabel):
                                list_IDs_fg.append(index)
                        else:
                            list_IDs_fg.append(index)

    if bm.val_img_data is not None:

        # img_val shape
        zsh_val, ysh_val, xsh_val, _ = bm.val_img_data.shape

        # list of validation IDs
        list_IDs_val_fg, list_IDs_val_bg = [], []

        # get validation IDs of patches
        for k in range(0, zsh_val-bm.z_patch+1, bm.validation_stride_size):
            for l in range(0, ysh_val-bm.y_patch+1, bm.validation_stride_size):
                for m in range(0, xsh_val-bm.x_patch+1, bm.validation_stride_size):
                    patch = bm.val_label_data[k:k+bm.z_patch, l:l+bm.y_patch, m:m+bm.x_patch]
                    index = k*ysh_val*xsh_val+l*xsh_val+m
                    if not np.any(patch==-1): # ignore padded areas
                        if bm.balance and not bm.val_dice:
                            if bm.separation:
                                centerLabel = bm.val_label_data[k+bm.z_patch//2,l+bm.y_patch//2,m+bm.x_patch//2]
                                if centerLabel>0 and np.any(np.logical_and(patch!=centerLabel, patch>0)):
                                    list_IDs_val_fg.append(index)
                                elif centerLabel>0 and np.any(patch!=centerLabel):
                                    list_IDs_val_bg.append(index)
                            elif np.any(patch>0):
                                list_IDs_val_fg.append(index)
                            else:
                                list_IDs_val_bg.append(index)
                        else:
                            if bm.separation:
                                centerLabel = bm.val_label_data[k+bm.z_patch//2,l+bm.y_patch//2,m+bm.x_patch//2]
                                if centerLabel>0 and np.any(patch!=centerLabel):
                                    list_IDs_val_fg.append(index)
                            else:
                                list_IDs_val_fg.append(index)

    # remove padding label
    bm.label_data[bm.label_data<0]=0
    if bm.val_img_data is not None:
        bm.val_label_data[bm.val_label_data<0]=0

    # number of labels
    nb_labels = len(allLabels)

    # input shape
    input_shape = (bm.z_patch, bm.y_patch, bm.x_patch, bm.channels)

    # parameters
    params = {'shuffle': True,
              'batch_size': bm.batch_size,
              'dim': (bm.z_patch, bm.y_patch, bm.x_patch),
              'dim_img': (zsh, ysh, xsh),
              'n_classes': nb_labels,
              'n_channels': bm.channels,
              'augment': (bm.flip_x, bm.flip_y, bm.flip_z, bm.swapaxes, bm.rotate, bm.rotate3d),
              'patch_normalization': bm.patch_normalization,
              'separation': bm.separation,
              'ignore_mask': bm.ignore_mask,
              'downsample': bm.downsample
              }

    # data generator
    validation_generator = None
    training_generator = DataGenerator(bm.img_data, bm.label_data, list_IDs_fg, list_IDs_bg, True, **params)
    if bm.val_img_data is not None:
        if bm.val_dice:
            val_metrics = Metrics(bm, bm.val_img_data, bm.val_label_data, list_IDs_val_fg, (zsh_val, ysh_val, xsh_val), nb_labels, False)
        else:
            params['dim_img'] = (zsh_val, ysh_val, xsh_val)
            params['augment'] = (False, False, False, False, 0, False)
            if len(list_IDs_val_bg) == 0:
                params['shuffle'] = False
            validation_generator = DataGenerator(bm.val_img_data, bm.val_label_data, list_IDs_val_fg, list_IDs_val_bg, False, **params)

    # monitor dice score on training data
    if bm.train_dice:
        train_metrics = Metrics(bm, bm.img_data, bm.label_data, list_IDs_fg, (zsh, ysh, xsh), nb_labels, True)

    # custom objects
    if bm.dice_loss:
        custom_objects = {'dice_coef_loss': dice_coef_loss,'loss_fn': loss_fn}
    elif bm.ignore_mask:
        custom_objects={'custom_loss': custom_loss}
    else:
        custom_objects=None

    # Rename the function to appear as "accuracy" in logs
    if bm.ignore_mask:
        custom_accuracy.__name__ = "accuracy"
        metrics=[custom_accuracy]
    else:
        metrics=['accuracy']

    # loss function
    if bm.dice_loss:
        loss=dice_coef_loss(nb_labels)
    elif bm.ignore_mask:
        loss=custom_loss
    else:
        loss='categorical_crossentropy'

    # Backend-specific setup functions
    def setup_tensorflow_strategy():
        import tensorflow as tf
        cdo = tf.distribute.ReductionToOneDevice()
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=cdo)
        ngpus = int(strategy.num_replicas_in_sync)
        print(f'Using TensorFlow backend with {ngpus} GPUs')
        return strategy

    def setup_pytorch_devices(model):
        import torch
        ngpus = 1 if torch.cuda.is_available() else 0
        print(f'Using PyTorch backend with {ngpus} GPUs')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Access underlying PyTorch model (depends on Keras version)
        # Sometimes: model._model or model.keras_model._model
        pt_model = getattr(model, "_model", model) # fallback to model if no _model attr
        pt_model.to(device)
        return pt_model

    # Common model build function
    def build_model():
        if bm.pretrained_model:
            return load_model(bm.pretrained_model, custom_objects=custom_objects)
        else:
            return make_unet(bm, input_shape, nb_labels)

    # Common optimizer setup
    def build_optimizer():
        from keras.optimizers.schedules import InverseTimeDecay
        lr_schedule = InverseTimeDecay(
            initial_learning_rate=bm.learning_rate,
            decay_rate=1e-6,
            decay_steps=1,
            staircase=False)
        if bm.optimizer in ['adam','Adam']:
            optimizer = Adam(learning_rate=lr_schedule, epsilon=1e-4) #lr=0.0001
        else:
            if bm.optimizer not in ['sgd','SGD']:
                print(f"Warning: unsupported optimizer {bm.optimizer}. Falling back to SGD.")
            optimizer = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
        if bm.mixed_precision:
            from keras.mixed_precision import LossScaleOptimizer
            optimizer = LossScaleOptimizer(optimizer, dynamic=False, initial_scale=128)
        return optimizer

    # Configure backend
    backend_name = K.backend()
    if backend_name == 'tensorflow':
        strategy = setup_tensorflow_strategy()
        with strategy.scope():
            model = build_model()
            #model.summary()
            optimizer = build_optimizer()
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    elif backend_name == 'torch':
        model = build_model()
        #model.summary()
        pt_model = setup_pytorch_devices(model)
        optimizer = build_optimizer()
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    else:
        raise RuntimeError(f"Unsupported backend: {backend_name}")

    # save meta data
    meta_data = MetaData(bm, bm.path_to_model, configuration_data, allLabels,
        bm.extension, bm.header, bm.crop_data, bm.cropping_weights, bm.cropping_config,
        normalization_parameters, bm.cropping_norm, bm.patch_normalization, bm.scaling)

    # model checkpoint
    if bm.val_img_data is not None:
        if bm.val_dice:
            callbacks = [val_metrics, meta_data]
        else:
            model_checkpoint_callback = ModelCheckpoint(
                filepath=str(bm.path_to_model),
                save_weights_only=False,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)
            callbacks = [model_checkpoint_callback, HistoryCallback(bm), meta_data]
            if bm.early_stopping > 0:
                callbacks.insert(0, EarlyStopping(monitor='val_accuracy', mode='max', patience=bm.early_stopping))
    else:
        callbacks = [ModelCheckpoint(filepath=str(bm.path_to_model)), HistoryCallback(bm), meta_data]

    # monitor dice score on training data
    if bm.train_dice:
        callbacks = [train_metrics] + callbacks

    # custom callback
    if bm.django_env and not bm.remote:
        callbacks.insert(-1, CustomCallback(bm.img_id, bm.epochs))

    # train model
    model.fit(training_generator,
        epochs=bm.epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        initial_epoch=bm.initial_epoch)

def load_prediction_data(bm, channels, normalization_parameters,
    region_of_interest, img, img_header, load_blockwise=False, z=None):

    # read image data
    if img is None:
        if load_blockwise:
            img_header = None
            tif = TiffFile(bm.path_to_image)
            img = imread(bm.path_to_image, key=range(z,min(len(tif.pages),z+bm.z_patch)))
            if len(img.shape)==2:
                img = img.reshape(1,img.shape[0],img.shape[1])
            if img.shape[0] < bm.z_patch:
                rest = bm.z_patch - img.shape[0]
                tmp = imread(bm.path_to_image, key=range(len(tif.pages)-rest,len(tif.pages)))
                if len(tmp.shape)==2:
                    tmp = tmp.reshape(1,tmp.shape[0],tmp.shape[1])
                img = np.append(img, tmp[::-1], axis=0)
        else:
            img, img_header = load_data(bm.path_to_image, 'first_queue')

    # verify validity
    if img is None:
        InputError.message = f'Invalid image data: {os.path.basename(bm.path_to_image)}.'
        raise InputError()

    # preserve original image data for post-processing
    img_data = None
    if bm.acwe:
        img_data = img.copy()

    # image data must have number of channels >=1
    if len(img.shape)==3:
        img = img.reshape(img.shape[0], img.shape[1], img.shape[2], 1)
    if img.shape[3] != channels:
        InputError.message = f'Number of channels must be {channels}.'
        raise InputError()

    # original image shape
    z_shape, y_shape, x_shape, _ = img.shape

    # automatic cropping of image to region of interest
    if np.any(region_of_interest):
        min_z, max_z, min_y, max_y, min_x, max_x = region_of_interest[:]
        img = np.copy(img[min_z:max_z,min_y:max_y,min_x:max_x], order='C')
        region_of_interest = np.array([min_z,max_z,min_y,max_y,min_x,max_x,z_shape,y_shape,x_shape])
        z_shape, y_shape, x_shape = max_z-min_z, max_y-min_y, max_x-min_x

    # resize image data
    if bm.scaling:
        img = img.astype(np.float32)
        img = img_resize(img, bm.z_scale, bm.y_scale, bm.x_scale)

    # scale image data
    if not bm.patch_normalization:
        img = img.astype(np.float32)
        for ch in range(channels):
            img[...,ch] -= np.amin(img[...,ch])
            img[...,ch] /= np.amax(img[...,ch])

    # normalize image data
    if bm.normalize:
        img = img.astype(np.float32)
        for ch in range(channels):
            mean, std = welford_mean_std(img[...,ch])
            img[...,ch] = (img[...,ch] - mean) / std
            img[...,ch] = img[...,ch] * normalization_parameters[1,ch] + normalization_parameters[0,ch]

        # limit intensity range
        img[img<0] = 0
        img[img>1] = 1

    return img, img_header, z_shape, y_shape, x_shape, region_of_interest, img_data

def append_ghost_areas(bm, img):
    # append ghost areas to make image dimensions divisible by patch size (mirror edge areas)
    zsh, ysh, xsh, _ = img.shape
    z_rest = bm.z_patch - (zsh % bm.z_patch)
    if z_rest == bm.z_patch:
        z_rest = -zsh
    else:
        img = np.append(img, img[-z_rest:][::-1], axis=0)
    y_rest = bm.y_patch - (ysh % bm.y_patch)
    if y_rest == bm.y_patch:
        y_rest = -ysh
    else:
        img = np.append(img, img[:,-y_rest:][:,::-1], axis=1)
    x_rest = bm.x_patch - (xsh % bm.x_patch)
    if x_rest == bm.x_patch:
        x_rest = -xsh
    else:
        img = np.append(img, img[:,:,-x_rest:][:,:,::-1], axis=2)
    return img, z_rest, y_rest, x_rest

def gradient(volData):
    grad = np.zeros(volData.shape, dtype=np.uint8)
    tmp = np.abs(volData[:-1] - volData[1:])
    tmp[tmp>0]=1
    grad[:-1] += tmp
    grad[1:] += tmp
    tmp = np.abs(volData[:,:-1] - volData[:,1:])
    tmp[tmp>0]=1
    grad[:,:-1] += tmp
    grad[:,1:] += tmp
    tmp = np.abs(volData[:,:,:-1] - volData[:,:,1:])
    tmp[tmp>0]=1
    grad[:,:,:-1] += tmp
    grad[:,:,1:] += tmp
    grad[grad>0]=1
    return grad

@numba.jit(nopython=True)
def scale_probabilities(final):
    zsh, ysh, xsh, nb_labels = final.shape
    for k in range(zsh):
        for l in range(ysh):
            for m in range(xsh):
                scale_factor = 0
                for n in range(nb_labels):
                    scale_factor += final[k,l,m,n]
                scale_factor = max(1, scale_factor)
                for n in range(nb_labels):
                    final[k,l,m,n] /= scale_factor
    return final

def predict_segmentation(bm, region_of_interest, channels, normalization_parameters):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ngpus = comm.Get_size()

    # optional result paths
    if bm.path_to_image:
        filename, bm.extension = os.path.splitext(bm.path_to_final)
        if bm.extension == '.gz':
            bm.extension = '.nii.gz'
            filename = filename[:-4]
        path_to_cleaned = filename + '.cleaned' + bm.extension
        path_to_filled = filename + '.filled' + bm.extension
        path_to_cleaned_filled = filename + '.cleaned.filled' + bm.extension
        path_to_refined = filename + '.refined' + bm.extension
        path_to_acwe = filename + '.acwe' + bm.extension
        path_to_probs = filename + '.probs.tif'

    # configure backend
    backend_name = K.backend()
    if backend_name == 'tensorflow':
        import tensorflow as tf
        # Set the visible GPU by ID
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Restrict TensorFlow to only use the specified GPU
                tf.config.experimental.set_visible_devices(gpus[rank % len(gpus)], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[rank % len(gpus)], True)
                print(f"[Rank {rank}] Using TensorFlow GPU {rank % len(gpus)}")
            except RuntimeError as e:
                print(e)
    elif backend_name == 'torch':
        import torch
        if  torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # Assign GPU based on MPI rank
            gpu_id = rank % torch.cuda.device_count()
            # Set the current CUDA device
            torch.cuda.set_device(gpu_id)
            device = torch.device("cuda")
            # Optional: also set environment variable for consistency with other libraries
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"[Rank {rank}] Using PyTorch GPU {gpu_id}")
        else:
            device = torch.device("cpu")
    else:
        raise RuntimeError(f"Unsupported backend: {backend_name}")

    # initialize results
    results = {}

    # number of labels
    nb_labels = len(bm.allLabels)
    results['allLabels'] = bm.allLabels

    # custom objects
    custom_objects = {"SyncBatchNormalization": BatchNormalization}
    if bm.dice_loss:
        def loss_fn(y_true, y_pred):
            dice = 0
            for index in range(1, nb_labels):
                dice += dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
            dice = dice / (nb_labels-1)
            #loss = -K.log(dice)
            loss = 1 - dice
            return loss
        custom_objects.update({'dice_coef_loss': dice_coef_loss,'loss_fn': loss_fn})
    elif bm.ignore_mask:
        custom_objects['custom_loss'] = custom_loss

    # load model
    model = load_model(bm.path_to_model, custom_objects=custom_objects, compile=False)
    if backend_name == 'torch':
        model.to(device)

    # check if data can be loaded blockwise to save host memory
    load_blockwise = False
    if not bm.scaling and not bm.normalize and bm.path_to_image and not np.any(region_of_interest) and \
      os.path.splitext(bm.path_to_image)[1] in ['.tif', '.tiff'] and not bm.acwe:

        # get image shape
        tif = TiffFile(bm.path_to_image)
        zsh = len(tif.pages)
        ysh, xsh = tif.pages[0].shape

        # load mask
        '''if bm.separation or bm.refinement:
            mask, _ = load_data(bm.mask)
            mask = mask.reshape(zsh, ysh, xsh, 1)
            mask, _, _, _ = append_ghost_areas(bm, mask)
            mask = mask.reshape(mask.shape[:-1])'''

        # determine new image size after appending ghost areas to make image dimensions divisible by patch size
        z_rest = bm.z_patch - (zsh % bm.z_patch)
        if z_rest == bm.z_patch:
            z_rest = -zsh
        else:
            zsh +=  z_rest
        y_rest = bm.y_patch - (ysh % bm.y_patch)
        if y_rest == bm.y_patch:
            y_rest = -ysh
        else:
            ysh +=  y_rest
        x_rest = bm.x_patch - (xsh % bm.x_patch)
        if x_rest == bm.x_patch:
            x_rest = -xsh
        else:
            xsh +=  x_rest

        # get Ids of patches
        '''list_IDs = []
        for k in range(0, zsh-bm.z_patch+1, bm.stride_size):
            for l in range(0, ysh-bm.y_patch+1, bm.stride_size):
                for m in range(0, xsh-bm.x_patch+1, bm.stride_size):
                    if bm.separation:
                        centerLabel = mask[k+bm.z_patch//2,l+bm.y_patch//2,m+bm.x_patch//2]
                        patch = mask[k:k+bm.z_patch, l:l+bm.y_patch, m:m+bm.x_patch]
                        if centerLabel>0 and np.any(patch!=centerLabel):
                            list_IDs.append(k*ysh*xsh+l*xsh+m)
                    elif bm.refinement:
                        if np.any(mask[k:k+bm.z_patch, l:l+bm.y_patch, m:m+bm.x_patch]):
                            list_IDs.append(k*ysh*xsh+l*xsh+m)
                    else:
                        list_IDs.append(k*ysh*xsh+l*xsh+m)'''

        # make length of list divisible by batch size
        '''max_i = len(list_IDs)
        rest = bm.batch_size - (len(list_IDs) % bm.batch_size)
        list_IDs = list_IDs + list_IDs[:rest]'''

        # prediction
        if zsh*ysh*xsh > 256**3:
            load_blockwise = True

    # load image data and calculate patch IDs
    if not load_blockwise:

        # load prediction data
        img, bm.img_header, z_shape, y_shape, x_shape, region_of_interest, bm.img_data = load_prediction_data(
            bm, channels, normalization_parameters, region_of_interest, bm.img_data, bm.img_header)

        # append ghost areas
        img, z_rest, y_rest, x_rest = append_ghost_areas(bm, img)

        # img shape
        zsh, ysh, xsh, _ = img.shape

        # list of IDs
        list_IDs = []

        # get Ids of patches
        for k in range(0, zsh-bm.z_patch+1, bm.stride_size):
            for l in range(0, ysh-bm.y_patch+1, bm.stride_size):
                for m in range(0, xsh-bm.x_patch+1, bm.stride_size):
                    list_IDs.append(k*ysh*xsh+l*xsh+m)

        # make length of list divisible by batch size
        rest = bm.batch_size - (len(list_IDs) % bm.batch_size)
        list_IDs = list_IDs + list_IDs[:rest]

        # number of patches
        nb_patches = len(list_IDs)

    # load all patches on GPU memory
    if not load_blockwise and nb_patches < 400 and not bm.separation and not bm.refinement:
      if rank==0:

        # parameters
        params = {'dim': (bm.z_patch, bm.y_patch, bm.x_patch),
                  'dim_img': (zsh, ysh, xsh),
                  'batch_size': bm.batch_size,
                  'n_channels': channels,
                  'patch_normalization': bm.patch_normalization}

        # data generator
        predict_generator = PredictDataGenerator(img, list_IDs, **params)

        # predict probabilities
        probabilities = model.predict(predict_generator, verbose=0, steps=None)

        # create final
        final = np.zeros((zsh, ysh, xsh, nb_labels), dtype=np.float32)
        nb = 0
        for k in range(0, zsh-bm.z_patch+1, bm.stride_size):
            for l in range(0, ysh-bm.y_patch+1, bm.stride_size):
                for m in range(0, xsh-bm.x_patch+1, bm.stride_size):
                    final[k:k+bm.z_patch, l:l+bm.y_patch, m:m+bm.x_patch] += probabilities[nb]
                    nb += 1

        # calculate result
        label = np.argmax(final, axis=-1).astype(np.uint8)

    else:
        # stream data batchwise to GPU to reduce memory usage
        X = np.empty((bm.batch_size, bm.z_patch, bm.y_patch, bm.x_patch, channels), dtype=np.float32)

        # allocate final probabilities array
        if rank==0 and bm.return_probs:
            if load_blockwise:
                if not os.path.exists(path_to_probs[:-4]):
                    os.mkdir(path_to_probs[:-4])
            else:
                final = np.zeros((zsh, ysh, xsh, nb_labels), dtype=np.float32)

        # allocate final result array
        if rank==0:
            label = np.zeros((zsh, ysh, xsh), dtype=np.uint8)

        # predict segmentation block by block
        z_indices = range(0, zsh-bm.z_patch+1, bm.stride_size)
        for j, z in enumerate(z_indices):
          # handle len(z_indices) % ngpus != 0
          if len(z_indices)-1-j < ngpus:
            nprocs = len(z_indices)-j
          else:
            nprocs = ngpus
          if j % ngpus == rank:

            # load blockwise from TIFF
            if load_blockwise:
                img, _, _, _, _, _, _ = load_prediction_data(bm,
                    channels, normalization_parameters,
                    region_of_interest, bm.img_data, bm.img_header, load_blockwise, z)
                img, _, _, _ = append_ghost_areas(bm, img)

                # load mask block
                if bm.separation or bm.refinement:
                    mask = imread(bm.mask, key=range(z,min(len(tif.pages),z+bm.z_patch)))
                    if len(mask.shape)==2:
                        mask = mask.reshape(1,mask.shape[0],mask.shape[1])
                    # pad zeros to make dimensions divisible by patch dimensions
                    pad_z = bm.z_patch - mask.shape[0]
                    pad_y = (bm.y_patch - (mask.shape[1] % bm.y_patch)) % bm.y_patch
                    pad_x = (bm.x_patch - (mask.shape[2] % bm.x_patch)) % bm.x_patch
                    pad_width = [(0, pad_z), (0, pad_y), (0, pad_x)]
                    mask = np.pad(mask, pad_width, mode='constant', constant_values=0)

            # list of IDs
            list_IDs_block = []

            # get Ids of patches
            k = 0 if load_blockwise else z
            for l in range(0, ysh-bm.y_patch+1, bm.stride_size):
                for m in range(0, xsh-bm.x_patch+1, bm.stride_size):
                    if bm.separation:
                        centerLabel = mask[k+bm.z_patch//2,l+bm.y_patch//2,m+bm.x_patch//2]
                        patch = mask[k:k+bm.z_patch, l:l+bm.y_patch, m:m+bm.x_patch]
                        if centerLabel>0 and np.any(patch!=centerLabel):
                            list_IDs_block.append(z*ysh*xsh+l*xsh+m)
                    elif bm.refinement:
                        if np.any(mask[k:k+bm.z_patch, l:l+bm.y_patch, m:m+bm.x_patch]):
                            list_IDs_block.append(z*ysh*xsh+l*xsh+m)
                    else:
                        list_IDs_block.append(z*ysh*xsh+l*xsh+m)

            # make length of list divisible by batch size
            max_i_block = len(list_IDs_block)
            rest = bm.batch_size - (len(list_IDs_block) % bm.batch_size)
            list_IDs_block = list_IDs_block + list_IDs_block[:rest]

            # number of patches
            nb_patches = len(list_IDs_block)

            # allocate block array
            if bm.separation:
                block_label = np.zeros((bm.z_patch, ysh, xsh), dtype=np.uint8)
            else:
                block_probs = np.zeros((bm.z_patch, ysh, xsh, nb_labels), dtype=np.float32)

            # get one batch of image patches
            for step in range(nb_patches//bm.batch_size):
                for i, ID in enumerate(list_IDs_block[step*bm.batch_size:(step+1)*bm.batch_size]):

                    # get patch indices
                    k=0 if load_blockwise else ID // (ysh*xsh)
                    rest = ID % (ysh*xsh)
                    l = rest // xsh
                    m = rest % xsh

                    # get patch
                    tmp_X = img[k:k+bm.z_patch,l:l+bm.y_patch,m:m+bm.x_patch]
                    if bm.patch_normalization:
                        tmp_X = tmp_X.copy().astype(np.float32)
                        for ch in range(channels):
                            mean, std = welford_mean_std(tmp_X[...,ch])
                            tmp_X[...,ch] -= mean
                            tmp_X[...,ch] /= max(std, 1e-6)
                    X[i] = tmp_X

                # predict batch
                Y = model.predict(X, verbose=0, steps=None, batch_size=bm.batch_size)

                # loop over result patches
                for i, ID in enumerate(list_IDs_block[step*bm.batch_size:(step+1)*bm.batch_size]):
                    rest = ID % (ysh*xsh)
                    l = rest // xsh
                    m = rest % xsh
                    if step*bm.batch_size+i < max_i_block:
                        if bm.separation:
                            patch = np.argmax(Y[i], axis=-1).astype(np.uint8)
                            block_label[:,l:l+bm.y_patch,m:m+bm.x_patch] += gradient(patch)
                        else:
                            block_probs[:,l:l+bm.y_patch,m:m+bm.x_patch] += Y[i]

            # communicate results
            if bm.separation:
                if rank==0:
                    label[z:z+bm.z_patch] += block_label
                    for source in range(1, nprocs):
                        receivedata = np.empty_like(block_label)
                        for i in range(bm.z_patch):
                            comm.Recv([receivedata[i], MPI.BYTE], source=source, tag=i)
                        block_start = z_indices[j+source]
                        label[block_start:block_start+bm.z_patch] += receivedata
                else:
                    for i in range(bm.z_patch):
                        comm.Send([block_label[i].copy(), MPI.BYTE], dest=0, tag=i)
            else:
                if rank==0:
                    for source in range(nprocs):
                        if source>0:
                            probs = np.empty_like(block_probs)
                            for i in range(bm.z_patch):
                                comm.Recv([probs[i], MPI.FLOAT], source=source, tag=i)
                        else:
                            probs = block_probs

                        # overlap in z direction
                        if bm.stride_size < bm.z_patch:
                            if j+source>0:
                                probs[:-bm.stride_size] += overlap
                            overlap = probs[bm.stride_size:].copy()

                        # block z dimension
                        block_z = z_indices[j+source]
                        if j+source==len(z_indices)-1: # last block
                            block_zsh = bm.z_patch
                            block_z_rest = z_rest if z_rest>0 else -block_zsh
                        else:
                            block_zsh = min(bm.stride_size, bm.z_patch)
                            block_z_rest = -block_zsh

                        # calculate result
                        label[block_z:block_z+block_zsh] = np.argmax(probs[:block_zsh], axis=-1).astype(np.uint8)

                        # return probabilities
                        if bm.return_probs:
                            if load_blockwise:
                                block_output = scale_probabilities(probs[:block_zsh])
                                block_output = block_output[:-block_z_rest,:-y_rest,:-x_rest]
                                imwrite(path_to_probs[:-4] + f"/block-{j+source}.tif", block_output)
                            else:
                                final[block_z:block_z+block_zsh] = probs[:block_zsh]
                else:
                    for i in range(bm.z_patch):
                        comm.Send([block_probs[i].copy(), MPI.FLOAT], dest=0, tag=i)
    if rank==0:

        # refine mask data with result
        '''if bm.refinement:
            # loop over boundary patches
            for i, ID in enumerate(list_IDs):
                if i < max_i:
                    k = ID // (ysh*xsh)
                    rest = ID % (ysh*xsh)
                    l = rest // xsh
                    m = rest % xsh
                    mask[k:k+bm.z_patch, l:l+bm.y_patch, m:m+bm.x_patch] = label[k:k+bm.z_patch, l:l+bm.y_patch, m:m+bm.x_patch]
            label = mask'''

        # remove ghost areas
        if bm.return_probs and not load_blockwise:
            final = final[:-z_rest,:-y_rest,:-x_rest]
        label = label[:-z_rest,:-y_rest,:-x_rest]
        zsh, ysh, xsh = label.shape

        # return probabilities
        if bm.return_probs and not load_blockwise:
            probabilities = scale_probabilities(final)
            if bm.scaling:
                probabilities = img_resize(probabilities, z_shape, y_shape, x_shape, interpolation=cv2.INTER_LINEAR)
            if np.any(region_of_interest):
                min_z,max_z,min_y,max_y,min_x,max_x,original_zsh,original_ysh,original_xsh = region_of_interest[:]
                tmp = np.zeros((original_zsh, original_ysh, original_xsh, nb_labels), dtype=np.float32)
                tmp[min_z:max_z,min_y:max_y,min_x:max_x] = probabilities
                probabilities = np.copy(tmp, order='C')
            results['probs'] = probabilities

        # rescale final to input size
        if bm.scaling:
            label = img_resize(label, z_shape, y_shape, x_shape, labels=True)

        # revert automatic cropping
        if np.any(region_of_interest):
            min_z,max_z,min_y,max_y,min_x,max_x,original_zsh,original_ysh,original_xsh = region_of_interest[:]
            tmp = np.zeros((original_zsh, original_ysh, original_xsh), dtype=np.uint8)
            tmp[min_z:max_z,min_y:max_y,min_x:max_x] = label
            label = np.copy(tmp, order='C')

        # get result
        if not bm.separation:
            label = get_labels(label, bm.allLabels)
        results['regular'] = label

        # load header from file
        if bm.header_file and os.path.exists(bm.header_file):
            _, bm.header = load_data(bm.header_file)
            # update file extension
            if bm.header is not None and bm.path_to_image:
                bm.extension = os.path.splitext(bm.header_file)[1]
                if bm.extension == '.gz':
                    bm.extension = '.nii.gz'
                bm.path_to_final = os.path.splitext(bm.path_to_final)[0] + bm.extension
                if bm.django_env and not bm.remote and not bm.tarfile:
                    bm.path_to_final = unique_file_path(bm.path_to_final)

        # handle amira header
        if bm.header is not None:
            if bm.extension == '.am':
                bm.header = set_image_dimensions(bm.header[0], label)
                if bm.img_header is not None:
                    try:
                        bm.header = set_physical_size(bm.header, bm.img_header[0])
                    except:
                        pass
                bm.header = [bm.header]
            else:
                # build new header
                if bm.img_header is None:
                    zsh, ysh, xsh = label.shape
                    bm.img_header = sitk.Image(xsh, ysh, zsh, bm.header.GetPixelID())
                # copy metadata
                for key in bm.header.GetMetaDataKeys():
                    if not (re.match(r'Segment\d+_Extent$', key) or key=='Segmentation_ConversionParameters'):
                        bm.img_header.SetMetaData(key, bm.header.GetMetaData(key))
                bm.header = bm.img_header
        results['header'] = bm.header

        # save result
        if bm.path_to_image:
            save_data(bm.path_to_final, label, header=bm.header, compress=bm.compression)
            if bm.return_probs and not load_blockwise:
                imwrite(path_to_probs, probabilities)

        # remove outliers
        if bm.clean:
            cleaned_result = clean(label, bm.clean)
            results['cleaned'] = cleaned_result
            if bm.path_to_image:
                save_data(path_to_cleaned, cleaned_result, header=bm.header, compress=bm.compression)
        if bm.fill:
            filled_result = clean(label, bm.fill)
            results['filled'] = filled_result
            if bm.path_to_image:
                save_data(path_to_filled, filled_result, header=bm.header, compress=bm.compression)
        if bm.clean and bm.fill:
            cleaned_filled_result = cleaned_result + (filled_result - label)
            results['cleaned_filled'] = cleaned_filled_result
            if bm.path_to_image:
                save_data(path_to_cleaned_filled, cleaned_filled_result, header=bm.header, compress=bm.compression)

        # post-processing with active contour
        if bm.acwe:
            acwe_result = activeContour(bm.img_data, label, bm.acwe_alpha, bm.acwe_smooth, bm.acwe_steps)
            refined_result = activeContour(bm.img_data, label, simple=True)
            results['acwe'] = acwe_result
            results['refined'] = refined_result
            if bm.path_to_image:
                save_data(path_to_acwe, acwe_result, header=bm.header, compress=bm.compression)
                save_data(path_to_refined, refined_result, header=bm.header, compress=bm.compression)

        return results, bm
    else:
        return None, None

