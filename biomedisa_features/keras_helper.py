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

import django
django.setup()
from biomedisa_app.models import Upload
from biomedisa_features.biomedisa_helper import img_resize, load_data, save_data, set_labels_to_zero
try:
    from tensorflow.keras.optimizers.legacy import SGD
except:
    from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, UpSampling3D, Activation, Reshape,
    BatchNormalization, Concatenate, ReLU, Add)
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from biomedisa_features.DataGenerator import DataGenerator
from biomedisa_features.PredictDataGenerator import PredictDataGenerator
import tensorflow as tf
import numpy as np
import cv2
import tarfile
from random import shuffle
from glob import glob
import random
import numba
import re
import os
import time
import h5py

class InputError(Exception):
    def __init__(self, message=None):
        self.message = message

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
    lattice = re.search('BoundingBox (.*),\n', s)
    lattice = lattice.group(1)
    i0, i1, i2, i3, i4, i5 = lattice.split(' ')
    bounding_box_i = re.search('&BoundingBox (.*),\n', s)
    bounding_box_i = bounding_box_i.group(1)

    # read header as string
    b = header.tobytes()
    try:
        s = b.decode("utf-8")
    except:
        s = b.decode("latin1")

    # get physical size from header
    lattice = re.search('BoundingBox (.*),\n', s)
    lattice = lattice.group(1)
    l0, l1, l2, l3, l4, l5 = lattice.split(' ')
    bounding_box_l = re.search('&BoundingBox (.*),\n', s)
    bounding_box_l = bounding_box_l.group(1)

    # change physical size in header
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

def make_axis_divisible_by_patch_size(a, patch_size):
    zsh, ysh, xsh = a.shape
    a = np.append(a, np.zeros((patch_size-(zsh % patch_size), ysh, xsh), a.dtype), axis=0)
    zsh, ysh, xsh = a.shape
    a = np.append(a, np.zeros((zsh, patch_size-(ysh % patch_size), xsh), a.dtype), axis=1)
    zsh, ysh, xsh = a.shape
    a = np.append(a, np.zeros((zsh, ysh, patch_size-(xsh % patch_size)), a.dtype), axis=2)
    a = np.copy(a, order='C')
    return a

def make_conv_block(nb_filters, input_tensor, block):
    def make_stage(input_tensor, stage):
        name = 'conv_{}_{}'.format(block, stage)
        x = Conv3D(nb_filters, (3, 3, 3), activation='relu',
                   padding='same', name=name, data_format="channels_last")(input_tensor)
        name = 'batch_norm_{}_{}'.format(block, stage)
        x = SyncBatchNormalization(name=name)(x)
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
    fx = SyncBatchNormalization(name=name)(fx)
    fx = Activation('relu')(fx)

    stage = 2
    name = 'conv_{}_{}'.format(block, stage)
    fx = Conv3D(nb_filters, (3, 3, 3), padding='same', name=name, data_format="channels_last")(fx)
    name = 'batch_norm_{}_{}'.format(block, stage)
    fx = SyncBatchNormalization(name=name)(fx)

    out = Add()([res,fx])
    out = ReLU()(out)

    return out

def make_unet(input_shape, nb_labels, filters='32-64-128-256-512-1024', resnet=False):

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

#=====================
# regular
#=====================

def load_training_data(normalize, img_list, label_list, channels, x_scale, y_scale, z_scale,
        crop_data, labels_to_compute, labels_to_remove, x_puffer=25, y_puffer=25, z_puffer=25):

    # get filenames
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

            # extract files if necessary
            if img_ext == '.tar':
                if not os.path.exists(img_dir):
                    tar = tarfile.open(img_name)
                    tar.extractall(path=img_dir)
                    tar.close()
                img_name = img_dir
            if label_ext == '.tar':
                if not os.path.exists(label_dir):
                    tar = tarfile.open(label_name)
                    tar.extractall(path=label_dir)
                    tar.close()
                label_name = label_dir

            for data_type in ['.am','.tif','.tiff','.hdr','.mhd','.mha','.nrrd','.nii','.nii.gz']:
                tmp_img_names = glob(img_name+'/**/*'+data_type, recursive=True)
                tmp_label_names = glob(label_name+'/**/*'+data_type, recursive=True)
                tmp_img_names = sorted(tmp_img_names)
                tmp_label_names = sorted(tmp_label_names)
                img_names.extend(tmp_img_names)
                label_names.extend(tmp_label_names)
            if len(img_names)==0:
                InputError.message = "Invalid image TAR file."
                raise InputError()
            if len(label_names)==0:
                InputError.message = "Invalid label TAR file."
                raise InputError()
        else:
            img_names.append(img_name)
            label_names.append(label_name)

    # load first label
    a, header, extension = load_data(label_names[0], 'first_queue', True)

    # if header is not single data stream Amira Mesh falling back to Multi-TIFF
    if extension != '.am':
        print(f'Warning! {extension} not supported. Falling back to TIFF.')
        extension, header = '.tif', None
    elif len(header) > 1:
        print('Warning! Multiple data streams are not supported. Falling back to TIFF.')
        extension, header = '.tif', None
    else:
        header = header[0]

    if a is None:
        InputError.message = "Invalid label data %s." %(os.path.basename(label_names[0]))
        raise InputError()
    a = a.astype(np.uint8)
    a = set_labels_to_zero(a, labels_to_compute, labels_to_remove)
    if crop_data:
        argmin_z,argmax_z,argmin_y,argmax_y,argmin_x,argmax_x = predict_blocksize(a, x_puffer, y_puffer, z_puffer)
        a = np.copy(a[argmin_z:argmax_z,argmin_y:argmax_y,argmin_x:argmax_x], order='C')
    np_unique = np.unique(a)
    label = np.zeros((z_scale, y_scale, x_scale), dtype=a.dtype)
    for k in np_unique:
        tmp = np.zeros_like(a)
        tmp[a==k] = 1
        tmp = img_resize(tmp, z_scale, y_scale, x_scale)
        label[tmp==1] = k

    # load first img
    img, _ = load_data(img_names[0], 'first_queue')
    if img is None:
        InputError.message = "Invalid image data %s." %(os.path.basename(img_names[0]))
        raise InputError()
    if crop_data:
        img = np.copy(img[argmin_z:argmax_z,argmin_y:argmax_y,argmin_x:argmax_x], order='C')
    img = img.astype(np.float32)
    img = img_resize(img, z_scale, y_scale, x_scale)
    img -= np.amin(img)
    img /= np.amax(img)
    mu, sig = np.mean(img), np.std(img)

    for img_name, label_name in zip(img_names[1:], label_names[1:]):

        # append label
        a, _ = load_data(label_name, 'first_queue')
        if a is None:
            InputError.message = "Invalid label data %s." %(os.path.basename(name))
            raise InputError()
        a = a.astype(np.uint8)
        a = set_labels_to_zero(a, labels_to_compute, labels_to_remove)
        if crop_data:
            argmin_z,argmax_z,argmin_y,argmax_y,argmin_x,argmax_x = predict_blocksize(a, x_puffer, y_puffer, z_puffer)
            a = np.copy(a[argmin_z:argmax_z,argmin_y:argmax_y,argmin_x:argmax_x], order='C')
        np_unique = np.unique(a)
        next_label = np.zeros((z_scale, y_scale, x_scale), dtype=a.dtype)
        for k in np_unique:
            tmp = np.zeros_like(a)
            tmp[a==k] = 1
            tmp = img_resize(tmp, z_scale, y_scale, x_scale)
            next_label[tmp==1] = k
        label = np.append(label, next_label, axis=0)

        # append image
        a, _ = load_data(img_name, 'first_queue')
        if a is None:
            InputError.message = "Invalid image data %s." %(os.path.basename(name))
            raise InputError()
        if crop_data:
            a = np.copy(a[argmin_z:argmax_z,argmin_y:argmax_y,argmin_x:argmax_x], order='C')
        a = a.astype(np.float32)
        a = img_resize(a, z_scale, y_scale, x_scale)
        a -= np.amin(a)
        a /= np.amax(a)
        if normalize:
            mu_tmp, sig_tmp = np.mean(a), np.std(a)
            a = (a - mu_tmp) / sig_tmp
            a = a * sig + mu
        img = np.append(img, a, axis=0)

    # scale image data to [0,1]
    img[img<0] = 0
    img[img>1] = 1

    # compute position data
    position = None
    if channels == 2:
        position = np.empty((z_scale, y_scale, x_scale), dtype=np.float32)
        position = compute_position(position, z_scale, y_scale, x_scale)
        position = np.sqrt(position)
        position /= np.amax(position)
        for k in range(len(img_names[1:])):
            a = np.copy(position)
            position = np.append(position, a, axis=0)

    # get labels
    allLabels = np.unique(label)

    # labels must be in ascending order
    for k, l in enumerate(allLabels):
        label[label==l] = k

    # configuration data
    configuration_data = np.array([channels, x_scale, y_scale, z_scale, normalize, mu, sig])

    return img, label, position, allLabels, configuration_data, header, extension, len(img_names)

class CustomCallback(Callback):
    def __init__(self, id, epochs):
        self.epochs = epochs
        self.id = id

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        image = Upload.objects.get(pk=self.id)
        if image.status == 3:
            self.model.stop_training = True
        else:
            keys = list(logs.keys())
            percentage = round((int(epoch)+1)*100/float(self.epochs),1)
            t = round(time.time() - self.epoch_time_start) * (self.epochs-int(epoch)-1)
            if t < 3600:
                time_remaining = str(t // 60) + ' min'
            else:
                time_remaining = str(t // 3600) + ' h ' + str((t % 3600) // 60) + ' min'
            try:
                val_accuracy = round(float(logs["val_accuracy"])*100,2)
                image.message = 'Progress {}%, {} remaining, {}% accuracy'.format(percentage,time_remaining,val_accuracy)
            except KeyError:
                image.message = 'Progress {}%, {} remaining'.format(percentage,time_remaining)
            image.save()
            print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

class MetaData(Callback):
    def __init__(self, path_to_model, configuration_data, allLabels, extension, header, crop_data, cropping_weights, cropping_config):
        self.path_to_model = path_to_model
        self.configuration_data = configuration_data
        self.allLabels = allLabels
        self.extension = extension
        self.header = header
        self.crop_data = crop_data
        self.cropping_weights = cropping_weights
        self.cropping_config = cropping_config

    def on_epoch_end(self, epoch, logs={}):
        hf = h5py.File(self.path_to_model, 'r')
        if not '/meta' in hf:
            hf.close()
            hf = h5py.File(self.path_to_model, 'r+')
            group = hf.create_group('meta')
            group.create_dataset('configuration', data=self.configuration_data)
            group.create_dataset('labels', data=self.allLabels)
            if self.extension == '.am':
                group.create_dataset('extension', data=self.extension)
                group.create_dataset('header', data=self.header)
            if self.crop_data:
                cm_group = hf.create_group('cropping_meta')
                cm_group.create_dataset('configuration', data=self.cropping_config)
                cw_group = hf.create_group('cropping_weights')
                for iterator, arr in enumerate(self.cropping_weights):
                    cw_group.create_dataset(str(iterator), data=arr)
        hf.close()

class Metrics(Callback):
    def __init__(self, img, label, list_IDs, dim_patch, dim_img, batch_size, path_to_model, early_stopping, validation_freq, n_classes, number_of_images):
        self.dim_patch = dim_patch
        self.dim_img = dim_img
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.label = label
        self.img = img
        self.path_to_model = path_to_model
        self.early_stopping = early_stopping
        self.validation_freq = validation_freq
        self.n_classes = n_classes
        self.number_of_images = number_of_images

    def on_train_begin(self, logs={}):
        self.history = {}
        self.history['val_accuracy'] = []
        self.history['accuracy'] = []
        self.history['loss'] = []

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.validation_freq == 0:

            result = np.zeros((*self.dim_img, self.n_classes), dtype=np.float32)

            len_IDs = len(self.list_IDs)
            n_batches = int(np.floor(len_IDs / self.batch_size))
            n_batches = min((512 * self.number_of_images) // self.batch_size, n_batches)
            np.random.shuffle(self.list_IDs)

            for batch in range(n_batches):
                # Generate indexes of the batch
                list_IDs_batch = self.list_IDs[batch*self.batch_size:(batch+1)*self.batch_size]

                # Initialization
                X_val = np.empty((self.batch_size, *self.dim_patch, 1), dtype=np.float32)

                # Generate data
                for i, ID in enumerate(list_IDs_batch):

                    # get patch indices
                    k = ID // (self.dim_img[1]*self.dim_img[2])
                    rest = ID % (self.dim_img[1]*self.dim_img[2])
                    l = rest // self.dim_img[2]
                    m = rest % self.dim_img[2]
                    X_val[i,:,:,:,0] = self.img[k:k+self.dim_patch[0],l:l+self.dim_patch[1],m:m+self.dim_patch[2]]

                # Prediction segmentation
                y_predict = np.asarray(self.model.predict(X_val, verbose=0, steps=None, batch_size=self.batch_size))

                for i, ID in enumerate(list_IDs_batch):

                    # get patch indices
                    k = ID // (self.dim_img[1]*self.dim_img[2])
                    rest = ID % (self.dim_img[1]*self.dim_img[2])
                    l = rest // self.dim_img[2]
                    m = rest % self.dim_img[2]
                    result[k:k+self.dim_patch[0],l:l+self.dim_patch[1],m:m+self.dim_patch[2]] += y_predict[i]

            # Compute dice score
            result = np.argmax(result, axis=-1)
            result = result.astype(np.uint8)
            dice = 2 * np.logical_and(self.label==result, (self.label+result)>0).sum() / \
                   float((self.label>0).sum() + (result>0).sum())

            # save best model only
            if epoch == 0:
                self.model.save(str(self.path_to_model))
            elif round(dice,5) > max(self.history['val_accuracy']):
                self.model.save(str(self.path_to_model))

            # add accuracy to history
            self.history['val_accuracy'].append(round(dice,5))
            self.history['accuracy'].append(round(logs["accuracy"],5))
            self.history['loss'].append(round(logs["loss"],5))
            logs["val_accuracy"] = max(self.history['val_accuracy'])

            # print accuracies
            print()
            print('val_acc (Dice):', self.history['val_accuracy'])
            print('train_acc:', self.history['accuracy'])
            print()

            # early stopping
            if self.early_stopping and max(self.history['val_accuracy']) not in self.history['val_accuracy'][-10:]:
                self.model.stop_training = True

def train_semantic_segmentation(normalize, img_list, label_list, x_scale, y_scale,
            z_scale, crop_data, path_to_model, z_patch, y_patch, x_patch, epochs,
            batch_size, channels, validation_split, stride_size, balance,
            flip_x, flip_y, flip_z, rotate, image, early_stopping, val_tf,
            validation_freq, cropping_weights, cropping_config, labels_to_compute,
            labels_to_remove, filters, resnet):

    # training data
    img, label, position, allLabels, configuration_data, header, extension, number_of_images = load_training_data(normalize,
                    img_list, label_list, channels, x_scale, y_scale, z_scale, crop_data, labels_to_compute, labels_to_remove)

    # img shape
    zsh, ysh, xsh = img.shape

    # validation data
    if validation_split:
        number_of_images = zsh // z_scale
        number_of_val_images = number_of_images - round(number_of_images * validation_split)
        number_of_images = round(number_of_images * validation_split)
        img_val = np.copy(img[number_of_images*z_scale:])
        label_val = np.copy(label[number_of_images*z_scale:])
        img = np.copy(img[:number_of_images*z_scale])
        label = np.copy(label[:number_of_images*z_scale])
        zsh, ysh, xsh = img.shape
        if channels == 2:
            position_val = np.copy(position[number_of_images*z_scale:])
            position = np.copy(position[:number_of_images*z_scale])
        else:
            position_val = None

        # img_val shape
        zsh_val, ysh_val, xsh_val = img_val.shape

        # list of validation IDs
        list_IDs_val = []

        # get validation IDs of patches
        for k in range(0, zsh_val-z_patch+1, stride_size):
            for l in range(0, ysh_val-y_patch+1, stride_size):
                for m in range(0, xsh_val-x_patch+1, stride_size):
                    list_IDs_val.append(k*ysh_val*xsh_val+l*xsh_val+m)

    # list of IDs
    list_IDs = []

    # get IDs of patches
    for k in range(0, zsh-z_patch+1, stride_size):
        for l in range(0, ysh-y_patch+1, stride_size):
            for m in range(0, xsh-x_patch+1, stride_size):
                list_IDs.append(k*ysh*xsh+l*xsh+m)

    # number of labels
    nb_labels = len(allLabels)

    # input shape
    input_shape = (z_patch, y_patch, x_patch, channels)

    # parameters
    params = {'batch_size': batch_size,
              'dim': (z_patch, y_patch, x_patch),
              'dim_img': (zsh, ysh, xsh),
              'n_classes': nb_labels,
              'n_channels': channels,
              'augment': (flip_x, flip_y, flip_z, False, rotate)}

    # create a strategy
    strategy = tf.distribute.experimental.CentralStorageStrategy()
    ngpus = int(strategy.num_replicas_in_sync)
    print(f'Number of devices: {ngpus}')

    # data generator
    validation_generator = None
    training_generator = DataGenerator(img, label, position, list_IDs, [], True, number_of_images, True, False, **params)
    if validation_split:
        if val_tf:
            params['batch_size'] = batch_size * ngpus
            params['dim_img'] = (zsh_val, ysh_val, xsh_val)
            params['augment'] = (False, False, False, False, 0)
            validation_generator = DataGenerator(img_val, label_val, position_val, list_IDs_val, [], True, number_of_val_images, False, False, **params)
        else:
            metrics = Metrics(img_val, label_val, list_IDs_val, (z_patch, y_patch, x_patch), (zsh_val, ysh_val, xsh_val), batch_size * ngpus,
                              path_to_model, early_stopping, validation_freq, nb_labels, number_of_val_images)

    # optimizer
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # compile model
    with strategy.scope():
        model = make_unet(input_shape, nb_labels, filters, resnet)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

    # save meta data
    meta_data = MetaData(path_to_model, configuration_data, allLabels, extension, header, crop_data, cropping_weights, cropping_config)

    # model checkpoint
    if validation_split:
        if val_tf:
            model_checkpoint_callback = ModelCheckpoint(
                filepath=str(path_to_model),
                save_weights_only=False,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)
            callbacks = [model_checkpoint_callback, CustomCallback(image.id,epochs), meta_data]
            if early_stopping:
                callbacks.insert(0, EarlyStopping(monitor='val_accuracy', mode='max', patience=10))
        else:
            callbacks = [metrics, CustomCallback(image.id,epochs), meta_data]
    else:
        callbacks = [ModelCheckpoint(filepath=str(path_to_model)),
                CustomCallback(image.id,epochs), meta_data]

    # train model
    model.fit(training_generator,
              epochs=epochs,
              validation_data=validation_generator,
              callbacks=callbacks)

def load_prediction_data(path_to_img, channels, x_scale, y_scale, z_scale,
                        normalize, mu, sig, region_of_interest):

    # read image data
    img, img_header, img_ext = load_data(path_to_img, 'first_queue', return_extension=True)
    if img is None:
        InputError.message = "Invalid image data %s." %(os.path.basename(path_to_img))
        raise InputError()
    if img_ext != '.am':
        img_header = None
    else:
        img_header = img_header[0]
    z_shape, y_shape, x_shape = img.shape

    # automatic cropping of image to region of interest
    if np.any(region_of_interest):
        min_z, max_z, min_y, max_y, min_x, max_x = region_of_interest[:]
        img = np.copy(img[min_z:max_z,min_y:max_y,min_x:max_x], order='C')
        region_of_interest = np.array([min_z,max_z,min_y,max_y,min_x,max_x,z_shape,y_shape,x_shape])
        z_shape, y_shape, x_shape = max_z-min_z, max_y-min_y, max_x-min_x

    # scale image data
    img = img.astype(np.float32)
    img = img_resize(img, z_scale, y_scale, x_scale)
    img -= np.amin(img)
    img /= np.amax(img)
    if normalize:
        mu_tmp, sig_tmp = np.mean(img), np.std(img)
        img = (img - mu_tmp) / sig_tmp
        img = img * sig + mu
        img[img<0] = 0
        img[img>1] = 1

    # compute position data
    position = None
    if channels == 2:
        position = np.empty((z_scale, y_scale, x_scale), dtype=np.float32)
        position = compute_position(position, z_scale, y_scale, x_scale)
        position = np.sqrt(position)
        position /= np.amax(position)

    return img, img_header, position, z_shape, y_shape, x_shape, region_of_interest

def predict_semantic_segmentation(img, position, path_to_model, path_to_final,
    z_patch, y_patch, x_patch, z_shape, y_shape, x_shape, compress, header,
    img_header, channels, stride_size, allLabels, batch_size, region_of_interest):

    # img shape
    zsh, ysh, xsh = img.shape

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
              'n_channels': channels}

    # data generator
    predict_generator = PredictDataGenerator(img, position, list_IDs, **params)

    # load model
    model = load_model(str(path_to_model))

    # predict
    if nb_patches < 400:
        probabilities = model.predict(predict_generator, verbose=0, steps=None)
    else:
        X = np.empty((batch_size, z_patch, y_patch, x_patch, channels), dtype=np.float32)
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
                X[i,:,:,:,0] = img[k:k+z_patch,l:l+y_patch,m:m+x_patch]
                if channels == 2:
                    X[i,:,:,:,1] = position[k:k+z_patch,l:l+y_patch,m:m+x_patch]

            probabilities[step*batch_size:(step+1)*batch_size] = model.predict(X, verbose=0, steps=None, batch_size=batch_size)

    # create final
    final = np.zeros((zsh, ysh, xsh, nb_labels), dtype=np.float32)
    nb = 0
    for k in range(0, zsh-z_patch+1, stride_size):
        for l in range(0, ysh-y_patch+1, stride_size):
            for m in range(0, xsh-x_patch+1, stride_size):
                final[k:k+z_patch, l:l+y_patch, m:m+x_patch] += probabilities[nb]
                nb += 1

    # get final
    out = np.argmax(final, axis=3)
    out = out.astype(np.uint8)

    # rescale final to input size
    np_unique = np.unique(out)
    label = np.zeros((z_shape, y_shape, x_shape), dtype=out.dtype)
    for k in np_unique:
        tmp = np.zeros_like(out)
        tmp[out==k] = 1
        tmp = img_resize(tmp, z_shape, y_shape, x_shape)
        label[tmp==1] = k

    # revert automatic cropping
    if np.any(region_of_interest):
        min_z,max_z,min_y,max_y,min_x,max_x,z_shape,y_shape,x_shape = region_of_interest[:]
        tmp = np.zeros((z_shape, y_shape, x_shape), dtype=out.dtype)
        tmp[min_z:max_z,min_y:max_y,min_x:max_x] = label
        label = np.copy(tmp)

    # save final
    label = label.astype(np.uint8)
    label = get_labels(label, allLabels)
    if header is not None:
        header = get_image_dimensions(header, label)
        if img_header is not None:
            try:
                header = get_physical_size(header, img_header)
            except:
                pass
        header = [header]
    save_data(path_to_final, label, header=header, compress=compress)

def predict_pre_final(img, path_to_model, x_scale, y_scale, z_scale, z_patch, y_patch, x_patch, \
                      normalize, mu, sig, channels, stride_size, batch_size):

    # img shape
    z_shape, y_shape, x_shape = img.shape

    # load position data
    if channels == 2:
        position = np.empty((z_scale, y_scale, x_scale), dtype=np.float32)
        position = compute_position(position, z_scale, y_scale, x_scale)
        position = np.sqrt(position)
        position /= np.amax(position)

    # resize img data
    img = img.astype(np.float32)
    img = img_resize(img, z_scale, y_scale, x_scale)
    img -= np.amin(img)
    img /= np.amax(img)
    if normalize:
        mu_tmp, sig_tmp = np.mean(img), np.std(img)
        img = (img - mu_tmp) / sig_tmp
        img = img * sig + mu
        img[img<0] = 0
        img[img>1] = 1

    # img shape
    zsh, ysh, xsh = img.shape

    # get number of 3D-patches
    nb = 0
    for k in range(0, zsh-z_patch+1, stride_size):
        for l in range(0, ysh-y_patch+1, stride_size):
            for m in range(0, xsh-x_patch+1, stride_size):
                nb += 1

    # allocate memory
    x_test = np.empty((nb, z_patch, y_patch, x_patch, channels), dtype=img.dtype)

    # create testing set
    nb = 0
    for k in range(0, zsh-z_patch+1, stride_size):
        for l in range(0, ysh-y_patch+1, stride_size):
            for m in range(0, xsh-x_patch+1, stride_size):
                x_test[nb,:,:,:,0] = img[k:k+z_patch, l:l+y_patch, m:m+x_patch]
                if channels == 2:
                    x_test[nb,:,:,:,1] = position[k:k+z_patch, l:l+y_patch, m:m+x_patch]
                nb += 1

    # reshape testing set
    x_test = x_test.reshape(nb, z_patch, y_patch, x_patch, channels)

    # create a MirroredStrategy
    if os.name == 'nt':
        cdo = tf.distribute.HierarchicalCopyAllReduce()
    else:
        cdo = tf.distribute.NcclAllReduce()
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=cdo)

    # load model
    with strategy.scope():
        model = load_model(str(path_to_model))

    # predict
    tmp = model.predict(x_test, batch_size=batch_size, verbose=0, steps=None)

    # create final
    final = np.zeros((zsh, ysh, xsh, tmp.shape[4]), dtype=np.float32)
    nb = 0
    for k in range(0, zsh-z_patch+1, stride_size):
        for l in range(0, ysh-y_patch+1, stride_size):
            for m in range(0, xsh-x_patch+1, stride_size):
                final[k:k+z_patch, l:l+y_patch, m:m+x_patch] += tmp[nb]
                nb += 1

    # get final
    out = np.argmax(final, axis=3)
    out = out.astype(np.uint8)

    # rescale final to input size
    np_unique = np.unique(out)
    label = np.zeros((z_shape, y_shape, x_shape), dtype=out.dtype)
    for k in np_unique:
        tmp = np.zeros_like(out)
        tmp[out==k] = 1
        tmp = img_resize(tmp, z_shape, y_shape, x_shape)
        label[tmp==1] = k

    return label

#=====================
# refine
#=====================

def load_training_data_refine(path_to_model, x_scale, y_scale, z_scale, patch_size, z_patch, y_patch, x_patch, normalize, \
                    img_list, label_list, channels, stride_size, allLabels, mu, sig, batch_size):

    # get filenames
    img_names, label_names = [], []
    for img_name, label_name in zip(img_list, label_list):

        img_dir, img_ext = os.path.splitext(img_name)
        if img_ext == '.gz':
            img_dir, img_ext = os.path.splitext(img_dir)

        label_dir, label_ext = os.path.splitext(label_name)
        if label_ext == '.gz':
            label_dir, label_ext = os.path.splitext(label_dir)

        if img_ext == '.tar' and label_ext == '.tar':
            for data_type in ['.am','.tif','.tiff','.hdr','.mhd','.mha','.nrrd','.nii','.nii.gz']:
                tmp_img_names = glob(img_dir+'/**/*'+data_type, recursive=True)
                tmp_label_names = glob(label_dir+'/**/*'+data_type, recursive=True)
                tmp_img_names = sorted(tmp_img_names)
                tmp_label_names = sorted(tmp_label_names)
                img_names.extend(tmp_img_names)
                label_names.extend(tmp_label_names)
        else:
            img_names.append(img_name)
            label_names.append(label_name)

    # predict pre-final
    final = []
    for name in img_names:
        a, _ = load_data(name, 'first_queue')
        if a is None:
            InputError.message = "Invalid image data %s." %(os.path.basename(name))
            raise InputError()
        a = predict_pre_final(a, path_to_model, x_scale, y_scale, z_scale, z_patch, y_patch, x_patch, \
                              normalize, mu, sig, channels, stride_size, batch_size)
        a = a.astype(np.float32)
        a /= len(allLabels) - 1
        #a = make_axis_divisible_by_patch_size(a, patch_size)
        final.append(a)

    # load img data
    img = []
    for name in img_names:
        a, _ = load_data(name, 'first_queue')
        a = a.astype(np.float32)
        a -= np.amin(a)
        a /= np.amax(a)
        if normalize:
            mu_tmp, sig_tmp = np.mean(a), np.std(a)
            a = (a - mu_tmp) / sig_tmp
            a = a * sig + mu
            a[a<0] = 0
            a[a>1] = 1
        #a = make_axis_divisible_by_patch_size(a, patch_size)
        img.append(a)

    # load label data
    label = []
    for name in label_names:
        a, _ = load_data(name, 'first_queue')
        if a is None:
            InputError.message = "Invalid label data %s." %(os.path.basename(name))
            raise InputError()
        #a = make_axis_divisible_by_patch_size(a, patch_size)
        label.append(a)

    # labels must be in ascending order
    for i in range(len(label)):
        for k, l in enumerate(allLabels):
            label[i][label[i]==l] = k

    return img, label, final

def config_training_data_refine(img, label, final, patch_size, stride_size):

    # get number of patches
    nb = 0
    for i in range(len(img)):
        zsh, ysh, xsh = img[i].shape
        for k in range(0, zsh-patch_size+1, stride_size):
            for l in range(0, ysh-patch_size+1, stride_size):
                for m in range(0, xsh-patch_size+1, stride_size):
                    tmp = np.copy(final[i][k:k+patch_size, l:l+patch_size, m:m+patch_size])
                    #if 0.1 * patch_size**3 < np.sum(tmp > 0) < 0.9 * patch_size**3:
                    if np.any(tmp[1:]!=tmp[0,0,0]):
                        nb += 1

    # create training data
    x_train = np.empty((nb, patch_size, patch_size, patch_size, 2), dtype=img[0].dtype)
    y_train = np.empty((nb, patch_size, patch_size, patch_size), dtype=label[0].dtype)

    nb = 0
    for i in range(len(img)):
        zsh, ysh, xsh = img[i].shape
        for k in range(0, zsh-patch_size+1, stride_size):
            for l in range(0, ysh-patch_size+1, stride_size):
                for m in range(0, xsh-patch_size+1, stride_size):
                    tmp = np.copy(final[i][k:k+patch_size, l:l+patch_size, m:m+patch_size])
                    #if 0.1 * patch_size**3 < np.sum(tmp > 0) < 0.9 * patch_size**3:
                    if np.any(tmp[1:]!=tmp[0,0,0]):
                        x_train[nb,:,:,:,0] = img[i][k:k+patch_size, l:l+patch_size, m:m+patch_size]
                        x_train[nb,:,:,:,1] = tmp
                        y_train[nb] = label[i][k:k+patch_size, l:l+patch_size, m:m+patch_size]
                        nb += 1

    return x_train, y_train

def train_semantic_segmentation_refine(img, label, final, path_to_model, patch_size, \
                    epochs, batch_size, allLabels, validation_split, stride_size):

    # number of labels
    nb_labels = len(allLabels)

    # load training
    x_train, y_train = config_training_data_refine(img, label, final, patch_size, stride_size)
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.int32)

    # make arrays divisible by batch_size
    rest = x_train.shape[0] % batch_size
    rest = x_train.shape[0] - rest
    x_train = x_train[:rest]
    y_train = y_train[:rest]

    # reshape arrays
    nsh, zsh, ysh, xsh, _ = x_train.shape
    x_train = x_train.reshape(nsh, zsh, ysh, xsh, 2)
    y_train = y_train.reshape(nsh, zsh, ysh, xsh, 1)

    # create one-hot vector
    y_train = to_categorical(y_train, num_classes=nb_labels)

    # input shape
    input_shape = (patch_size, patch_size, patch_size, 2)

    # optimizer
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # create a MirroredStrategy
    if os.name == 'nt':
        cdo = tf.distribute.HierarchicalCopyAllReduce()
    else:
        cdo = tf.distribute.NcclAllReduce()
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=cdo)
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # compile model
    with strategy.scope():
        model = make_unet(input_shape, nb_labels)
        model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    # fit model
    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=validation_split)

    # save model
    model.save(str(path_to_model))

def load_refine_data(path_to_img, path_to_final, patch_size, normalize, allLabels, mu, sig):

    # read image data
    img, _ = load_data(path_to_img, 'first_queue')
    if img is None:
        InputError.message = "Invalid image data %s." %(os.path.basename(path_to_img))
        raise InputError()
    z_shape, y_shape, x_shape = img.shape
    img = img.astype(np.float32)
    img -= np.amin(img)
    img /= np.amax(img)
    if normalize:
        mu_tmp, sig_tmp = np.mean(img), np.std(img)
        img = (img - mu_tmp) / sig_tmp
        img = img * sig + mu
        img[img<0] = 0
        img[img>1] = 1
    #img = make_axis_divisible_by_patch_size(img, patch_size)

    # load label data
    label, _ = load_data(path_to_final, 'first_queue')
    if label is None:
        InputError.message = "Invalid label data %s." %(os.path.basename(path_to_final))
        raise InputError()
    #label = make_axis_divisible_by_patch_size(label, patch_size)

    # labels must be in ascending order
    for k, l in enumerate(allLabels):
        label[label==l] = k

    # load final data and scale to [0,1]
    final = np.copy(label)
    final = final.astype(np.float32)
    final /= len(allLabels) - 1

    return img, label, final, z_shape, y_shape, x_shape

def refine_semantic_segmentation(path_to_img, path_to_final, path_to_model, patch_size,
                                 compress, header, img_header, normalize, stride_size, allLabels,
                                 mu, sig, batch_size):

    # load refine data
    img, label, final, z_shape, y_shape, x_shape = load_refine_data(path_to_img, path_to_final,
                                             patch_size, normalize, allLabels, mu, sig)

    # get number of 3D-patches
    nb = 0
    zsh, ysh, xsh = img.shape
    for k in range(0, zsh-patch_size+1, stride_size):
        for l in range(0, ysh-patch_size+1, stride_size):
            for m in range(0, xsh-patch_size+1, stride_size):
                tmp = label[k:k+patch_size, l:l+patch_size, m:m+patch_size]
                #if 0.1 * patch_size**3 < np.sum(tmp > 0) < 0.9 * patch_size**3:
                if np.any(tmp[1:]!=tmp[0,0,0]):
                    nb += 1

    # create prediction set
    x_test = np.empty((nb, patch_size, patch_size, patch_size, 2), dtype=img.dtype)
    nb = 0
    zsh, ysh, xsh = img.shape
    for k in range(0, zsh-patch_size+1, stride_size):
        for l in range(0, ysh-patch_size+1, stride_size):
            for m in range(0, xsh-patch_size+1, stride_size):
                tmp = label[k:k+patch_size, l:l+patch_size, m:m+patch_size]
                #if 0.1 * patch_size**3 < np.sum(tmp > 0) < 0.9 * patch_size**3:
                if np.any(tmp[1:]!=tmp[0,0,0]):
                    x_test[nb,:,:,:,0] = img[k:k+patch_size, l:l+patch_size, m:m+patch_size]
                    x_test[nb,:,:,:,1] = final[k:k+patch_size, l:l+patch_size, m:m+patch_size]
                    nb += 1

    # reshape prediction data
    x_test = x_test.reshape(nb, patch_size, patch_size, patch_size, 2)

    # create a MirroredStrategy
    if os.name == 'nt':
        cdo = tf.distribute.HierarchicalCopyAllReduce()
    else:
        cdo = tf.distribute.NcclAllReduce()
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=cdo)

    # load model
    with strategy.scope():
        model = load_model(str(path_to_model))

    # predict
    prob = model.predict(x_test, batch_size=batch_size, verbose=0, steps=None)

    # create final
    nb = 0
    zsh, ysh, xsh = img.shape
    final = np.zeros((zsh, ysh, xsh, prob.shape[4]), dtype=np.float32)
    for k in range(0, zsh-patch_size+1, stride_size):
        for l in range(0, ysh-patch_size+1, stride_size):
            for m in range(0, xsh-patch_size+1, stride_size):
                tmp = label[k:k+patch_size, l:l+patch_size, m:m+patch_size]
                #if 0.1 * patch_size**3 < np.sum(tmp > 0) < 0.9 * patch_size**3:
                if np.any(tmp[1:]!=tmp[0,0,0]):
                    final[k:k+patch_size, l:l+patch_size, m:m+patch_size] += prob[nb]
                    nb += 1

    final = np.argmax(final, axis=3)
    final = final.astype(np.uint8)

    out = np.copy(label)
    for k in range(0, zsh-patch_size+1, stride_size):
        for l in range(0, ysh-patch_size+1, stride_size):
            for m in range(0, xsh-patch_size+1, stride_size):
                tmp = label[k:k+patch_size, l:l+patch_size, m:m+patch_size]
                #if 0.1 * patch_size**3 < np.sum(tmp > 0) < 0.9 * patch_size**3:
                if np.any(tmp[1:]!=tmp[0,0,0]):
                    out[k:k+patch_size, l:l+patch_size, m:m+patch_size] = final[k:k+patch_size, l:l+patch_size, m:m+patch_size]

    # save final
    out = out.astype(np.uint8)
    out = get_labels(out, allLabels)
    out = out[:z_shape, :y_shape, :x_shape]
    if header is not None:
        header = get_image_dimensions(header, out)
        if img_header is not None:
            try:
                header = get_physical_size(header, img_header)
            except:
                pass
        header = [header]
    save_data(path_to_final, out, header=header, compress=compress)

