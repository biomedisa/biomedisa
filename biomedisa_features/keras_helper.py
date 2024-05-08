##########################################################################
##                                                                      ##
##  Copyright (c) 2024 Philipp Lösel. All rights reserved.              ##
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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    from tensorflow.keras.optimizers.legacy import SGD
except:
    from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, UpSampling3D, Activation, Reshape,
    BatchNormalization, Concatenate, ReLU, Add, GlobalAveragePooling3D,
    Dense, Dropout, MaxPool3D, Flatten, Multiply)
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from biomedisa_features.DataGenerator import DataGenerator
from biomedisa_features.PredictDataGenerator import PredictDataGenerator
from biomedisa_features.biomedisa_helper import (
    img_resize, load_data, save_data, set_labels_to_zero, id_generator)
from biomedisa_features.remove_outlier import clean, fill
from biomedisa_features.active_contour import activeContour
import matplotlib.pyplot as plt
import SimpleITK as sitk
import tensorflow as tf
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
import atexit
import shutil

class InputError(Exception):
    def __init__(self, message=None, img_names=[], label_names=[]):
        self.message = message
        self.img_names = img_names
        self.label_names = label_names

def save_history(history, path_to_model, val_dice, train_dice):
    # summarize history for accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    if val_dice and train_dice:
        plt.plot(history['dice'])
        plt.plot(history['val_dice'])
        plt.legend(['Accuracy (train)', 'Accuracy (test)', 'Dice score (train)', 'Dice score (test)'], loc='upper left')
    elif train_dice:
        plt.plot(history['dice'])
        plt.legend(['Accuracy (train)', 'Accuracy (test)', 'Dice score (train)'], loc='upper left')
    elif val_dice:
        plt.plot(history['val_dice'])
        plt.legend(['Accuracy (train)', 'Accuracy (test)', 'Dice score (test)'], loc='upper left')
    else:
        plt.legend(['Accuracy (train)', 'Accuracy (test)'], loc='upper left')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.tight_layout()  # To prevent overlapping of subplots
    plt.savefig(path_to_model.replace('.h5','_acc.png'), dpi=300, bbox_inches='tight')
    plt.clf()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.tight_layout()  # To prevent overlapping of subplots
    plt.savefig(path_to_model.replace('.h5','_loss.png'), dpi=300, bbox_inches='tight')
    plt.clf()
    # save history dictonary
    np.save(path_to_model.replace('.h5','.npy'), history)

def predict_blocksize(labelData, x_puffer, y_puffer, z_puffer):
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

def get_image_dimensions(header, data):

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

def get_physical_size(header, img_header):

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

def make_conv_block(nb_filters, input_tensor, block):
    def make_stage(input_tensor, stage):
        name = 'conv_{}_{}'.format(block, stage)
        x = Conv3D(nb_filters, (3, 3, 3), activation='relu',
                   padding='same', name=name, data_format="channels_last")(input_tensor)
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

def make_conv_block_resnet(nb_filters, input_tensor, block):

    # Residual/Skip connection
    res = Conv3D(nb_filters, (1, 1, 1), padding='same', use_bias=False, name="Identity{}_1".format(block))(input_tensor)

    stage = 1
    name = 'conv_{}_{}'.format(block, stage)
    fx = Conv3D(nb_filters, (3, 3, 3), activation='relu', padding='same', name=name, data_format="channels_last")(input_tensor)
    name = 'batch_norm_{}_{}'.format(block, stage)
    try:
        fx = BatchNormalization(name=name, synchronized=True)(fx)
    except:
        fx = BatchNormalization(name=name)(fx)
    fx = Activation('relu')(fx)

    stage = 2
    name = 'conv_{}_{}'.format(block, stage)
    fx = Conv3D(nb_filters, (3, 3, 3), padding='same', name=name, data_format="channels_last")(fx)
    name = 'batch_norm_{}_{}'.format(block, stage)
    try:
        fx = BatchNormalization(name=name, synchronized=True)(fx)
    except:
        fx = BatchNormalization(name=name)(fx)

    out = Add()([res,fx])
    out = ReLU()(out)

    return out

def make_unet(input_shape, nb_labels, filters='32-64-128-256-512', resnet=False):

    nb_plans, nb_rows, nb_cols, _ = input_shape

    inputs = Input(input_shape)

    filters = filters.split('-')
    filters = np.array(filters, dtype=int)
    latent_space_size = filters[-1]
    filters = filters[:-1]
    convs = []

    i = 1
    for f in filters:
        if i==1:
            if resnet:
                conv = make_conv_block_resnet(f, inputs, i)
            else:
                conv = make_conv_block(f, inputs, i)
        else:
            if resnet:
                conv = make_conv_block_resnet(f, pool, i)
            else:
                conv = make_conv_block(f, pool, i)
        pool = MaxPooling3D(pool_size=(2, 2, 2))(conv)
        convs.append(conv)
        i += 1

    if resnet:
        conv = make_conv_block_resnet(latent_space_size, pool, i)
    else:
        conv = make_conv_block(latent_space_size, pool, i)
    i += 1

    for k, f in enumerate(filters[::-1]):
        up = Concatenate()([UpSampling3D(size=(2, 2, 2))(conv), convs[-(k+1)]])
        if resnet:
            conv = make_conv_block_resnet(f, up, i)
        else:
            conv = make_conv_block(f, up, i)
        i += 1

    conv = Conv3D(nb_labels, (1, 1, 1), name=f'conv_{i}_1')(conv)

    x = Reshape((nb_plans * nb_rows * nb_cols, nb_labels))(conv)
    x = Activation('softmax')(x)
    outputs = Reshape((nb_plans, nb_rows, nb_cols, nb_labels))(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def get_labels(arr, allLabels):
    np_unique = np.unique(arr)
    final = np.zeros_like(arr)
    for k in np_unique:
        final[arr == k] = allLabels[k]
    return final

def read_img_list(img_list, label_list):
    # read filenames
    InputError.img_names = []
    InputError.label_names = []
    img_names, label_names = [], []
    for img_name, label_name in zip(img_list, label_list):

        # check for tarball
        img_dir, img_ext = os.path.splitext(img_name)
        if img_ext == '.gz':
            img_dir, img_ext = os.path.splitext(img_dir)

        label_dir, label_ext = os.path.splitext(label_name)
        if label_ext == '.gz':
            label_dir, label_ext = os.path.splitext(label_dir)

        if (img_ext == '.tar' and label_ext == '.tar') or (os.path.isdir(img_name) and os.path.isdir(label_name)):

            # extract files
            if img_ext == '.tar':
                img_dir = BASE_DIR + '/tmp/' + id_generator(40)
                tar = tarfile.open(img_name)
                tar.extractall(path=img_dir)
                tar.close()
                img_name = img_dir
            if label_ext == '.tar':
                label_dir = BASE_DIR + '/tmp/' + id_generator(40)
                tar = tarfile.open(label_name)
                tar.extractall(path=label_dir)
                tar.close()
                label_name = label_dir

            for data_type in ['.am','.tif','.tiff','.hdr','.mhd','.mha','.nrrd','.nii','.nii.gz','.zip','.mrc']:
                img_names += [file for file in glob.glob(img_name+'/**/*'+data_type, recursive=True) if not os.path.basename(file).startswith('.')]
                label_names += [file for file in glob.glob(label_name+'/**/*'+data_type, recursive=True) if not os.path.basename(file).startswith('.')]
            img_names = sorted(img_names)
            label_names = sorted(label_names)
            if len(img_names)==0 or len(label_names)==0 or len(img_names)!=len(label_names):
                if img_ext == '.tar' and os.path.exists(img_name):
                    shutil.rmtree(img_name)
                if label_ext == '.tar' and os.path.exists(label_name):
                    shutil.rmtree(label_name)
                if len(img_names)!=len(label_names):
                    InputError.message = 'Number of image and label files must be the same'
                elif img_ext == '.tar':
                    InputError.message = 'Invalid image TAR file'
                elif label_ext == '.tar':
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

def remove_extracted_data(img_list, label_list):
    for img_name in img_list:
        if BASE_DIR + '/tmp/' in img_name:
            img_dir = img_name[:len(BASE_DIR + '/tmp/') + 40]
            if os.path.exists(img_dir):
                shutil.rmtree(img_dir)
    for label_name in label_list:
        if BASE_DIR + '/tmp/' in label_name:
            label_dir = label_name[:len(BASE_DIR + '/tmp/') + 40]
            if os.path.exists(label_dir):
                shutil.rmtree(label_dir)

def load_training_data(normalize, img_list, label_list, channels, x_scale, y_scale, z_scale, no_scaling,
        crop_data, labels_to_compute, labels_to_remove, img_in=None, label_in=None,
        normalization_parameters=None, allLabels=None, header=None, extension='.tif',
        x_puffer=25, y_puffer=25, z_puffer=25):

    # read image lists
    if any(img_list):
        img_names, label_names = read_img_list(img_list, label_list)
        InputError.img_names = img_names
        InputError.label_names = label_names

    # load first label
    if any(img_list):
        label, header, extension = load_data(label_names[0], 'first_queue', True)
        if label is None:
            InputError.message = f'Invalid label data "{os.path.basename(label_names[0])}"'
            raise InputError()
    elif type(label_in) is list:
        label = label_in[0]
    else:
        label = label_in
    label_dim = label.shape
    label = set_labels_to_zero(label, labels_to_compute, labels_to_remove)
    label_values, counts = np.unique(label, return_counts=True)
    print(f'{os.path.basename(label_names[0])}:', 'Labels:', label_values[1:], 'Sizes:', counts[1:])
    if crop_data:
        argmin_z,argmax_z,argmin_y,argmax_y,argmin_x,argmax_x = predict_blocksize(label, x_puffer, y_puffer, z_puffer)
        label = np.copy(label[argmin_z:argmax_z,argmin_y:argmax_y,argmin_x:argmax_x], order='C')
    if not no_scaling:
        label = img_resize(label, z_scale, y_scale, x_scale, labels=True)

    # if header is not single data stream Amira Mesh falling back to Multi-TIFF
    if extension != '.am':
        if extension != '.tif':
            print(f'Warning! Please use -hf or --header_file="path_to_training_label{extension}" for prediction to save your result as "{extension}"')
        extension, header = '.tif', None
    elif len(header) > 1:
        print('Warning! Multiple data streams are not supported. Falling back to TIFF.')
        extension, header = '.tif', None
    else:
        header = header[0]

    # load first img
    if any(img_list):
        img, _ = load_data(img_names[0], 'first_queue')
        if img is None:
            InputError.message = f'Invalid image data "{os.path.basename(img_names[0])}"'
            raise InputError()
    elif type(img_in) is list:
        img = img_in[0]
    else:
        img = img_in
    if label_dim != img.shape:
        InputError.message = f'Dimensions of "{os.path.basename(img_names[0])}" and "{os.path.basename(label_names[0])}" do not match'
        raise InputError()

    # ensure images have channels >=1
    if len(img.shape)==3:
        z_shape, y_shape, x_shape = img.shape
        img = img.reshape(z_shape, y_shape, x_shape, 1)
    if channels is None:
        channels = img.shape[3]
    if channels != img.shape[3]:
        InputError.message = f'Number of channels must be {channels} for "{os.path.basename(img_names[0])}"'
        raise InputError()

    # crop data
    if crop_data:
        img = np.copy(img[argmin_z:argmax_z,argmin_y:argmax_y,argmin_x:argmax_x], order='C')

    # scale/resize image data
    img = img.astype(np.float32)
    if not no_scaling:
        img = img_resize(img, z_scale, y_scale, x_scale)

    # normalize image data
    for c in range(channels):
        img[:,:,:,c] -= np.amin(img[:,:,:,c])
        img[:,:,:,c] /= np.amax(img[:,:,:,c])
        if normalization_parameters is None:
            normalization_parameters = np.zeros((2,channels))
            normalization_parameters[0,c] = np.mean(img[:,:,:,c])
            normalization_parameters[1,c] = np.std(img[:,:,:,c])
        elif normalize:
            mean, std = np.mean(img[:,:,:,c]), np.std(img[:,:,:,c])
            img[:,:,:,c] = (img[:,:,:,c] - mean) / std
            img[:,:,:,c] = img[:,:,:,c] * normalization_parameters[1,c] + normalization_parameters[0,c]

    # loop over list of images
    if any(img_list) or type(img_in) is list:
        number_of_images = len(img_names) if any(img_list) else len(img_in)

        for k in range(1, number_of_images):

            # append label
            if any(label_list):
                a, _ = load_data(label_names[k], 'first_queue')
                if a is None:
                    InputError.message = f'Invalid label data "{os.path.basename(label_names[k])}"'
                    raise InputError()
            else:
                a = label_in[k]
            label_dim = a.shape
            a = set_labels_to_zero(a, labels_to_compute, labels_to_remove)
            label_values, counts = np.unique(a, return_counts=True)
            print(f'{os.path.basename(label_names[k])}:', 'Labels:', label_values[1:], 'Sizes:', counts[1:])
            if crop_data:
                argmin_z,argmax_z,argmin_y,argmax_y,argmin_x,argmax_x = predict_blocksize(a, x_puffer, y_puffer, z_puffer)
                a = np.copy(a[argmin_z:argmax_z,argmin_y:argmax_y,argmin_x:argmax_x], order='C')
            if not no_scaling:
                a = img_resize(a, z_scale, y_scale, x_scale, labels=True)
            label = np.append(label, a, axis=0)

            # append image
            if any(img_list):
                a, _ = load_data(img_names[k], 'first_queue')
                if a is None:
                    InputError.message = f'Invalid image data "{os.path.basename(img_names[k])}"'
                    raise InputError()
            else:
                a = img_in[k]
            if label_dim != a.shape:
                InputError.message = f'Dimensions of "{os.path.basename(img_names[k])}" and "{os.path.basename(label_names[k])}" do not match'
                raise InputError()
            if len(a.shape)==3:
                z_shape, y_shape, x_shape = a.shape
                a = a.reshape(z_shape, y_shape, x_shape, 1)
            if a.shape[3] != channels:
                InputError.message = f'Number of channels must be {channels} for "{os.path.basename(img_names[k])}"'
                raise InputError()
            if crop_data:
                a = np.copy(a[argmin_z:argmax_z,argmin_y:argmax_y,argmin_x:argmax_x], order='C')
            a = a.astype(np.float32)
            if not no_scaling:
                a = img_resize(a, z_scale, y_scale, x_scale)
            for c in range(channels):
                a[:,:,:,c] -= np.amin(a[:,:,:,c])
                a[:,:,:,c] /= np.amax(a[:,:,:,c])
                if normalize:
                    mean, std = np.mean(a[:,:,:,c]), np.std(a[:,:,:,c])
                    a[:,:,:,c] = (a[:,:,:,c] - mean) / std
                    a[:,:,:,c] = a[:,:,:,c] * normalization_parameters[1,c] + normalization_parameters[0,c]
            img = np.append(img, a, axis=0)

    # remove extracted data
    if any(img_list):
        remove_extracted_data(img_names, label_names)

    # limit intensity range
    img[img<0] = 0
    img[img>1] = 1

    # get labels
    if allLabels is None:
        allLabels = np.unique(label)

    # labels must be in ascending order
    for k, l in enumerate(allLabels):
        label[label==l] = k

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
            try:
                val_accuracy = round(float(logs["val_accuracy"])*100,1)
                image.message = 'Progress {}%, {} remaining, {}% accuracy'.format(percentage,time_remaining,val_accuracy)
            except KeyError:
                image.message = 'Progress {}%, {} remaining'.format(percentage,time_remaining)
            image.save()
            print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

class MetaData(Callback):
    def __init__(self, path_to_model, configuration_data, allLabels,
        extension, header, crop_data, cropping_weights, cropping_config,
        normalization_parameters, cropping_norm):

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

    def on_epoch_end(self, epoch, logs={}):
        hf = h5py.File(self.path_to_model, 'r')
        if not '/meta' in hf:
            hf.close()
            hf = h5py.File(self.path_to_model, 'r+')
            group = hf.create_group('meta')
            group.create_dataset('configuration', data=self.configuration_data)
            group.create_dataset('normalization', data=self.normalization_parameters)
            group.create_dataset('labels', data=self.allLabels)
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
                        tmp_X = np.copy(tmp_X, order='C')
                        for c in range(self.n_channels):
                            tmp_X[:,:,:,c] -= np.mean(tmp_X[:,:,:,c])
                            tmp_X[:,:,:,c] /= max(np.std(tmp_X[:,:,:,c]), 1e-6)
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

            # calculate categorical crossentropy
            if not self.train:
                crossentropy = categorical_crossentropy(self.label, softmax(result))

            # get result
            result = np.argmax(result, axis=-1)
            result = result.astype(np.uint8)

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
                if epoch == 0 or round(dice,4) > max(self.history['val_dice']):
                    self.model.save(str(self.path_to_model))

                # add accuracy to history
                self.history['loss'].append(round(logs['loss'],4))
                self.history['accuracy'].append(round(logs['accuracy'],4))
                if self.train_dice:
                    self.history['dice'].append(round(logs['dice'],4))
                self.history['val_accuracy'].append(round(accuracy,4))
                self.history['val_dice'].append(round(dice,4))
                self.history['val_loss'].append(round(crossentropy,4))

                # tensorflow monitoring variables
                logs['val_loss'] = crossentropy
                logs['val_accuracy'] = accuracy
                logs['val_dice'] = dice
                logs['best_acc'] = max(self.history['accuracy'])
                if self.train_dice:
                    logs['best_dice'] = max(self.history['dice'])
                logs['best_val_acc'] = max(self.history['val_accuracy'])
                logs['best_val_dice'] = max(self.history['val_dice'])

                # plot history in figure and save as numpy array
                save_history(self.history, self.path_to_model, True, self.train_dice)

                # print accuracies
                print('\nValidation history:')
                print('train_acc:', self.history['accuracy'])
                if self.train_dice:
                    print('train_dice:', self.history['dice'])
                print('val_acc:', self.history['val_accuracy'])
                print('val_dice:', self.history['val_dice'])
                print('')

                # early stopping
                if self.early_stopping > 0 and max(self.history['val_dice']) not in self.history['val_dice'][-self.early_stopping:]:
                    self.model.stop_training = True

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
    zsh,ysh,xsh = true_labels.shape
    # Calculate categorical crossentropy
    loss = 0
    for z in range(zsh):
        for y in range(ysh):
            for x in range(xsh):
                l = true_labels[z,y,x]
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
        loss = -K.log(dice)
        #loss = 1 - dice
        return loss
    return loss_fn

def train_semantic_segmentation(bm,
    img_list, label_list,
    val_img_list, val_label_list,
    img=None, label=None,
    img_val=None, label_val=None,
    header=None, extension='.tif'):

    # training data
    img, label, allLabels, normalization_parameters, header, extension, bm.channels = load_training_data(bm.normalize,
                    img_list, label_list, None, bm.x_scale, bm.y_scale, bm.z_scale, bm.no_scaling, bm.crop_data,
                    bm.only, bm.ignore, img, label, None, None, header, extension)

    # configuration data
    configuration_data = np.array([bm.channels, bm.x_scale, bm.y_scale, bm.z_scale, bm.normalize, 0, 1])

    # img shape
    zsh, ysh, xsh, _ = img.shape

    # validation data
    if any(val_img_list) or img_val is not None:
        img_val, label_val, _, _, _, _, _ = load_training_data(bm.normalize,
                    val_img_list, val_label_list, bm.channels, bm.x_scale, bm.y_scale, bm.z_scale, bm.no_scaling, bm.crop_data,
                    bm.only, bm.ignore, img_val, label_val, normalization_parameters, allLabels)

    elif bm.validation_split:
        split = round(zsh * bm.validation_split)
        img_val = np.copy(img[split:], order='C')
        label_val = np.copy(label[split:], order='C')
        img = np.copy(img[:split], order='C')
        label = np.copy(label[:split], order='C')
        zsh, ysh, xsh, _ = img.shape

    # list of IDs
    list_IDs_fg, list_IDs_bg = [], []

    # get IDs of patches
    if bm.balance:
        for k in range(0, zsh-bm.z_patch+1, bm.stride_size):
            for l in range(0, ysh-bm.y_patch+1, bm.stride_size):
                for m in range(0, xsh-bm.x_patch+1, bm.stride_size):
                    if np.any(label[k:k+bm.z_patch, l:l+bm.y_patch, m:m+bm.x_patch]):
                        list_IDs_fg.append(k*ysh*xsh+l*xsh+m)
                    else:
                        list_IDs_bg.append(k*ysh*xsh+l*xsh+m)
    else:
        for k in range(0, zsh-bm.z_patch+1, bm.stride_size):
            for l in range(0, ysh-bm.y_patch+1, bm.stride_size):
                for m in range(0, xsh-bm.x_patch+1, bm.stride_size):
                    list_IDs_fg.append(k*ysh*xsh+l*xsh+m)

    if img_val is not None:

        # img_val shape
        zsh_val, ysh_val, xsh_val, _ = img_val.shape

        # list of validation IDs
        list_IDs_val_fg, list_IDs_val_bg = [], []

        # get validation IDs of patches
        if bm.balance and not bm.val_dice:
            for k in range(0, zsh_val-bm.z_patch+1, bm.validation_stride_size):
                for l in range(0, ysh_val-bm.y_patch+1, bm.validation_stride_size):
                    for m in range(0, xsh_val-bm.x_patch+1, bm.validation_stride_size):
                        if np.any(label_val[k:k+bm.z_patch, l:l+bm.y_patch, m:m+bm.x_patch]):
                            list_IDs_val_fg.append(k*ysh_val*xsh_val+l*xsh_val+m)
                        else:
                            list_IDs_val_bg.append(k*ysh_val*xsh_val+l*xsh_val+m)
        else:
            for k in range(0, zsh_val-bm.z_patch+1, bm.validation_stride_size):
                for l in range(0, ysh_val-bm.y_patch+1, bm.validation_stride_size):
                    for m in range(0, xsh_val-bm.x_patch+1, bm.validation_stride_size):
                        list_IDs_val_fg.append(k*ysh_val*xsh_val+l*xsh_val+m)

    # number of labels
    nb_labels = len(allLabels)

    # input shape
    input_shape = (bm.z_patch, bm.y_patch, bm.x_patch, bm.channels)

    # parameters
    params = {'batch_size': bm.batch_size,
              'dim': (bm.z_patch, bm.y_patch, bm.x_patch),
              'dim_img': (zsh, ysh, xsh),
              'n_classes': nb_labels,
              'n_channels': bm.channels,
              'augment': (bm.flip_x, bm.flip_y, bm.flip_z, bm.swapaxes, bm.rotate),
              'patch_normalization': bm.patch_normalization}

    # data generator
    validation_generator = None
    training_generator = DataGenerator(img, label, list_IDs_fg, list_IDs_bg, True, True, False, **params)
    if img_val is not None:
        if bm.val_dice:
            val_metrics = Metrics(bm, img_val, label_val, list_IDs_val_fg, (zsh_val, ysh_val, xsh_val), nb_labels, False)
        else:
            params['dim_img'] = (zsh_val, ysh_val, xsh_val)
            params['augment'] = (False, False, False, False, 0)
            validation_generator = DataGenerator(img_val, label_val, list_IDs_val_fg, list_IDs_val_bg, True, False, False, **params)

    # monitor dice score on training data
    if bm.train_dice:
        train_metrics = Metrics(bm, img, label, list_IDs_fg, (zsh, ysh, xsh), nb_labels, True)

    # create a MirroredStrategy
    cdo = tf.distribute.ReductionToOneDevice()
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=cdo)
    ngpus = int(strategy.num_replicas_in_sync)
    print(f'Number of devices: {ngpus}')
    if ngpus == 1 and os.name == 'nt':
        atexit.register(strategy._extended._collective_ops._pool.close)

    # compile model
    with strategy.scope():

        # build model
        model = make_unet(input_shape, nb_labels, bm.network_filters, bm.resnet)
        model.summary()

        # pretrained model
        if bm.pretrained_model:
            model_pretrained = load_model(bm.pretrained_model)
            model.set_weights(model_pretrained.get_weights())
            if not bm.fine_tune:
                nb_blocks = len(bm.network_filters.split('-'))
                for k in range(nb_blocks+1, 2*nb_blocks):
                    for l in [1,2]:
                        name = f'conv_{k}_{l}'
                        layer = model.get_layer(name)
                        layer.trainable = False
                name = f'conv_{2*nb_blocks}_1'
                layer = model.get_layer(name)
                layer.trainable = False

        # optimizer
        sgd = SGD(learning_rate=bm.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

        # comile model
        loss=dice_coef_loss(nb_labels) if bm.dice_loss else 'categorical_crossentropy'
        model.compile(loss=loss,
                      optimizer=sgd,
                      metrics=['accuracy'])

    # save meta data
    meta_data = MetaData(bm.path_to_model, configuration_data, allLabels,
        extension, header, bm.crop_data, bm.cropping_weights, bm.cropping_config,
        normalization_parameters, bm.cropping_norm)

    # model checkpoint
    if img_val is not None:
        if bm.val_dice:
            callbacks = [val_metrics, meta_data]
        else:
            model_checkpoint_callback = ModelCheckpoint(
                filepath=str(bm.path_to_model),
                save_weights_only=False,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)
            callbacks = [model_checkpoint_callback, meta_data]
            if bm.early_stopping > 0:
                callbacks.insert(0, EarlyStopping(monitor='val_accuracy', mode='max', patience=bm.early_stopping))
    else:
        callbacks = [ModelCheckpoint(filepath=str(bm.path_to_model)), meta_data]

    # monitor dice score on training data
    if bm.train_dice:
        callbacks = [train_metrics] + callbacks

    # custom callback
    if bm.django_env and not bm.remote:
        callbacks.insert(-1, CustomCallback(bm.img_id, bm.epochs))

    # train model
    history = model.fit(training_generator,
              epochs=bm.epochs,
              validation_data=validation_generator,
              callbacks=callbacks,
              workers=bm.workers)

    # save monitoring figure on train end
    if img_val is not None and not bm.val_dice:
        save_history(history.history, bm.path_to_model, False, bm.train_dice)

def load_prediction_data(path_to_img, channels, x_scale, y_scale, z_scale,
                        no_scaling, normalize, normalization_parameters, region_of_interest,
                        img, img_header):
    # read image data
    if img is None:
        img, img_header = load_data(path_to_img, 'first_queue')
        InputError.img_names = [path_to_img]
        InputError.label_names = []

    # verify validity
    if img is None:
        InputError.message = f'Invalid image data: {os.path.basename(path_to_img)}.'
        raise InputError()

    # preserve original image data
    img_data = img.copy()

    # handle all images having channels >=1
    if len(img.shape)==3:
        z_shape, y_shape, x_shape = img.shape
        img = img.reshape(z_shape, y_shape, x_shape, 1)
    if img.shape[3] != channels:
        InputError.message = f'Number of channels must be {channels}.'
        raise InputError()

    # image shape
    z_shape, y_shape, x_shape, _ = img.shape

    # automatic cropping of image to region of interest
    if np.any(region_of_interest):
        min_z, max_z, min_y, max_y, min_x, max_x = region_of_interest[:]
        img = np.copy(img[min_z:max_z,min_y:max_y,min_x:max_x], order='C')
        region_of_interest = np.array([min_z,max_z,min_y,max_y,min_x,max_x,z_shape,y_shape,x_shape])
        z_shape, y_shape, x_shape = max_z-min_z, max_y-min_y, max_x-min_x

    # scale/resize image data
    img = img.astype(np.float32)
    if not no_scaling:
        img = img_resize(img, z_scale, y_scale, x_scale)

    # normalize image data
    for c in range(channels):
        img[:,:,:,c] -= np.amin(img[:,:,:,c])
        img[:,:,:,c] /= np.amax(img[:,:,:,c])
        if normalize:
            mean, std = np.mean(img[:,:,:,c]), np.std(img[:,:,:,c])
            img[:,:,:,c] = (img[:,:,:,c] - mean) / std
            img[:,:,:,c] = img[:,:,:,c] * normalization_parameters[1,c] + normalization_parameters[0,c]

    # limit intensity range
    if normalize:
        img[img<0] = 0
        img[img>1] = 1

    return img, img_header, z_shape, y_shape, x_shape, region_of_interest, img_data

def predict_semantic_segmentation(bm, img, path_to_model,
    z_patch, y_patch, x_patch, z_shape, y_shape, x_shape, compress, header,
    img_header, stride_size, allLabels, batch_size, region_of_interest,
    no_scaling, extension, img_data):

    results = {}

    # img shape
    zsh, ysh, xsh, csh = img.shape

    # number of labels
    nb_labels = len(allLabels)

    # list of IDs
    list_IDs = []

    # get Ids of patches
    for k in range(0, zsh-z_patch+1, stride_size):
        for l in range(0, ysh-y_patch+1, stride_size):
            for m in range(0, xsh-x_patch+1, stride_size):
                list_IDs.append(k*ysh*xsh+l*xsh+m)

    # make length of list divisible by batch size
    rest = batch_size - (len(list_IDs) % batch_size)
    list_IDs = list_IDs + list_IDs[:rest]

    # number of patches
    nb_patches = len(list_IDs)

    # parameters
    params = {'dim': (z_patch, y_patch, x_patch),
              'dim_img': (zsh, ysh, xsh),
              'batch_size': batch_size,
              'n_channels': csh,
              'patch_normalization': bm.patch_normalization}

    # data generator
    predict_generator = PredictDataGenerator(img, list_IDs, **params)

    # load model
    model = load_model(str(path_to_model))

    # predict
    if nb_patches < 400:
        probabilities = model.predict(predict_generator, verbose=0, steps=None)
    else:
        X = np.empty((batch_size, z_patch, y_patch, x_patch, csh), dtype=np.float32)
        probabilities = np.zeros((nb_patches, z_patch, y_patch, x_patch, nb_labels), dtype=np.float32)

        # get image patches
        for step in range(nb_patches//batch_size):
            for i, ID in enumerate(list_IDs[step*batch_size:(step+1)*batch_size]):

                # get patch indices
                k = ID // (ysh*xsh)
                rest = ID % (ysh*xsh)
                l = rest // xsh
                m = rest % xsh

                # get patch
                tmp_X = img[k:k+z_patch,l:l+y_patch,m:m+x_patch]
                if bm.patch_normalization:
                    tmp_X = np.copy(tmp_X, order='C')
                    for c in range(csh):
                        tmp_X[:,:,:,c] -= np.mean(tmp_X[:,:,:,c])
                        tmp_X[:,:,:,c] /= max(np.std(tmp_X[:,:,:,c]), 1e-6)
                X[i] = tmp_X

            probabilities[step*batch_size:(step+1)*batch_size] = model.predict(X, verbose=0, steps=None, batch_size=batch_size)

    # create final
    final = np.zeros((zsh, ysh, xsh, nb_labels), dtype=np.float32)
    if bm.return_probs:
        counter = np.zeros((zsh, ysh, xsh, nb_labels), dtype=np.float32)
    nb = 0
    for k in range(0, zsh-z_patch+1, stride_size):
        for l in range(0, ysh-y_patch+1, stride_size):
            for m in range(0, xsh-x_patch+1, stride_size):
                final[k:k+z_patch, l:l+y_patch, m:m+x_patch] += probabilities[nb]
                if bm.return_probs:
                    counter[k:k+z_patch, l:l+y_patch, m:m+x_patch] += 1
                nb += 1

    # return probabilities
    if bm.return_probs:
        counter[counter==0] = 1
        probabilities = final / counter
        if not no_scaling:
            probabilities = img_resize(probabilities, z_shape, y_shape, x_shape)
        if np.any(region_of_interest):
            min_z,max_z,min_y,max_y,min_x,max_x,original_zsh,original_ysh,original_xsh = region_of_interest[:]
            tmp = np.zeros((original_zsh, original_ysh, original_xsh, nb_labels), dtype=np.float32)
            tmp[min_z:max_z,min_y:max_y,min_x:max_x] = probabilities
            probabilities = np.copy(tmp, order='C')
        results['probs'] = probabilities

    # get final
    label = np.argmax(final, axis=3)
    label = label.astype(np.uint8)

    # rescale final to input size
    if not no_scaling:
        label = img_resize(label, z_shape, y_shape, x_shape, labels=True)

    # revert automatic cropping
    if np.any(region_of_interest):
        min_z,max_z,min_y,max_y,min_x,max_x,original_zsh,original_ysh,original_xsh = region_of_interest[:]
        tmp = np.zeros((original_zsh, original_ysh, original_xsh), dtype=np.uint8)
        tmp[min_z:max_z,min_y:max_y,min_x:max_x] = label
        label = np.copy(tmp, order='C')

    # get result
    label = get_labels(label, allLabels)
    results['regular'] = label

    # load header from file
    if bm.header_file and os.path.exists(bm.header_file):
        _, header = load_data(bm.header_file)
        # update file extension
        if header is not None and bm.path_to_image:
            extension = os.path.splitext(bm.header_file)[1]
            if extension == '.gz':
                extension = '.nii.gz'
            bm.path_to_final = os.path.splitext(bm.path_to_final)[0] + extension
            if bm.django_env and not bm.remote and not bm.tmp_dir:
                from biomedisa_app.views import unique_file_path
                bm.path_to_final = unique_file_path(bm.path_to_final)

    # handle amira header
    if header is not None:
        if extension == '.am':
            header = get_image_dimensions(header[0], label)
            if img_header is not None:
                try:
                    header = get_physical_size(header, img_header[0])
                except:
                    pass
            header = [header]
        else:
            # build new header
            if img_header is None:
                zsh, ysh, xsh = label.shape
                img_header = sitk.Image(xsh, ysh, zsh, header.GetPixelID())
            # copy metadata
            for key in header.GetMetaDataKeys():
                if not (re.match(r'Segment\d+_Extent$', key) or key=='Segmentation_ConversionParameters'):
                    img_header.SetMetaData(key, header.GetMetaData(key))
            header = img_header
    results['header'] = header

    # save result
    if bm.path_to_image:
        save_data(bm.path_to_final, label, header=header, compress=compress)

        # paths to optional results
        filename, extension = os.path.splitext(bm.path_to_final)
        if extension == '.gz':
            extension = '.nii.gz'
            filename = filename[:-4]
        path_to_cleaned = filename + '.cleaned' + extension
        path_to_filled = filename + '.filled' + extension
        path_to_cleaned_filled = filename + '.cleaned.filled' + extension
        path_to_refined = filename + '.refined' + extension
        path_to_acwe = filename + '.acwe' + extension

    # remove outliers
    if bm.clean:
        cleaned_result = clean(label, bm.clean)
        results['cleaned'] = cleaned_result
        if bm.path_to_image:
            save_data(path_to_cleaned, cleaned_result, header=header, compress=compress)
    if bm.fill:
        filled_result = clean(label, bm.fill)
        results['filled'] = filled_result
        if bm.path_to_image:
            save_data(path_to_filled, filled_result, header=header, compress=compress)
    if bm.clean and bm.fill:
        cleaned_filled_result = cleaned_result + (filled_result - label)
        results['cleaned_filled'] = cleaned_filled_result
        if bm.path_to_image:
            save_data(path_to_cleaned_filled, cleaned_filled_result, header=header, compress=compress)

    # post-processing with active contour
    if bm.acwe:
        acwe_result = activeContour(img_data, label, bm.acwe_alpha, bm.acwe_smooth, bm.acwe_steps)
        refined_result = activeContour(img_data, label, simple=True)
        results['acwe'] = acwe_result
        results['refined'] = refined_result
        if bm.path_to_image:
            save_data(path_to_acwe, acwe_result, header=header, compress=compress)
            save_data(path_to_refined, refined_result, header=header, compress=compress)

    return results, bm

