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

try:
    import django
    django.setup()
    from biomedisa_app.models import Upload
except:
    pass

from biomedisa_features.biomedisa_helper import img_resize, load_data, save_data, set_labels_to_zero
try:
    from tensorflow.keras.optimizers.legacy import SGD
except:
    from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, UpSampling3D, Activation, Reshape,
    BatchNormalization, Concatenate, ReLU, Add, GlobalAveragePooling3D,
    Dense, Dropout, MaxPool3D, Flatten)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from tensorflow import keras
from biomedisa_features.DataGenerator import DataGenerator
from biomedisa_features.PredictDataGenerator import PredictDataGenerator
import matplotlib.pyplot as plt
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
import atexit

class InputError(Exception):
    def __init__(self, message=None):
        self.message = message

def save_history(history, path_to_model):
    # summarize history for accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    if 'val_loss' in history:
        plt.legend(['train', 'test'], loc='upper left')
    else:
        plt.legend(['train', 'test (Dice)'], loc='upper left')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.tight_layout()  # To prevent overlapping of subplots
    plt.savefig(path_to_model.replace('.h5','_acc.png'), dpi=300, bbox_inches='tight')
    plt.clf()
    # summarize history for loss
    plt.plot(history['loss'])
    if 'val_loss' in history:
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

def make_classification_model(input_shape, nb_labels):
    """Build a 3D convolutional neural network model."""
    inputs = Input(input_shape)

    x = Conv3D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = MaxPool3D(pool_size=2)(x)
    #try:
    #    x = BatchNormalization(name=name, synchronized=True)(x)
    #except:
    #    x = BatchNormalization(name=name)(x)
    #x = Dropout(0.25)(x)

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    try:
        x = BatchNormalization(name=name, synchronized=True)(x)
    except:
        x = BatchNormalization(name=name)(x)
    #x = Dropout(0.25)(x)

    x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    try:
        x = BatchNormalization(name=name, synchronized=True)(x)
    except:
        x = BatchNormalization(name=name)(x)
    #x = Dropout(0.25)(x)

    #x = Flatten()(x)
    x = GlobalAveragePooling3D()(x)
    #x = Dense(units=1024, activation="relu")(x)
    #x = Dropout(0.5)(x)

    x = Dense(units=512, activation="relu")(x)
    x = Dropout(0.25)(x)

    outputs = Dense(units=nb_labels, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def get_labels(arr, allLabels):
    np_unique = np.unique(arr)
    final = np.zeros_like(arr)
    for k in np_unique:
        final[arr == k] = allLabels[k]
    return final

def load_training_data(normalize, img_list, label_list, channels, x_scale, y_scale, z_scale, no_scaling,
        crop_data, labels_to_compute, labels_to_remove, img_in=None, label_in=None,
        configuration_data=None, allLabels=None, header=None, extension='.tif',
        x_puffer=25, y_puffer=25, z_puffer=25):

    if any(img_list):

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
                    InputError.message = "Invalid image data."
                    raise InputError()
                if len(label_names)==0:
                    InputError.message = "Invalid label data."
                    raise InputError()
            else:
                img_names.append(img_name)
                label_names.append(label_name)

    # load first label
    if any(img_list):
        label, header, extension = load_data(label_names[0], 'first_queue', True)
        if label is None:
            InputError.message = "Invalid label data %s." %(os.path.basename(label_names[0]))
            raise InputError()
    elif type(label_in) is list:
        label = label_in[0]
    else:
        label = label_in
    label = label.astype(np.uint8)
    label = set_labels_to_zero(label, labels_to_compute, labels_to_remove)
    if crop_data:
        argmin_z,argmax_z,argmin_y,argmax_y,argmin_x,argmax_x = predict_blocksize(label, x_puffer, y_puffer, z_puffer)
        label = np.copy(label[argmin_z:argmax_z,argmin_y:argmax_y,argmin_x:argmax_x], order='C')
    if not no_scaling:
        label = img_resize(label, z_scale, y_scale, x_scale, labels=True)

    # if header is not single data stream Amira Mesh falling back to Multi-TIFF
    if extension != '.am':
        if extension != '.tif':
            print(f'Warning! {extension} not supported. Falling back to TIFF.')
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
            InputError.message = "Invalid image data %s." %(os.path.basename(img_names[0]))
            raise InputError()
    elif type(img_in) is list:
        img = img_in[0]
    else:
        img = img_in
    if crop_data:
        img = np.copy(img[argmin_z:argmax_z,argmin_y:argmax_y,argmin_x:argmax_x], order='C')
    img = img.astype(np.float32)
    if not no_scaling:
        img = img_resize(img, z_scale, y_scale, x_scale)
    img -= np.amin(img)
    img /= np.amax(img)
    if configuration_data is not None and normalize:
        mu, sig = configuration_data[5], configuration_data[6]
        mu_tmp, sig_tmp = np.mean(img), np.std(img)
        img = (img - mu_tmp) / sig_tmp
        img = img * sig + mu
    else:
        mu, sig = np.mean(img), np.std(img)

    # loop over list of images
    if any(img_list) or type(img_in) is list:
        number_of_images = len(img_names) if any(img_list) else len(img_in)

        for k in range(1, number_of_images):

            # append label
            if any(label_list):
                a, _ = load_data(label_names[k], 'first_queue')
                if a is None:
                    InputError.message = "Invalid label data %s." %(os.path.basename(label_names[k]))
                    raise InputError()
            else:
                a = label_in[k]
            a = a.astype(np.uint8)
            a = set_labels_to_zero(a, labels_to_compute, labels_to_remove)
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
                    InputError.message = "Invalid image data %s." %(os.path.basename(img_names[k]))
                    raise InputError()
            else:
                a = img_in[k]
            if crop_data:
                a = np.copy(a[argmin_z:argmax_z,argmin_y:argmax_y,argmin_x:argmax_x], order='C')
            a = a.astype(np.float32)
            if not no_scaling:
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

    # get labels
    if allLabels is None:
        allLabels = np.unique(label)

    # labels must be in ascending order
    for k, l in enumerate(allLabels):
        label[label==l] = k

    # configuration data
    configuration_data = np.array([channels, x_scale, y_scale, z_scale, normalize, mu, sig])

    return img, label, allLabels, configuration_data, header, extension

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
    def __init__(self, img, label, list_IDs, dim_patch, dim_img, batch_size, path_to_model, early_stopping, validation_freq, n_classes, n_channels, average_dice, django_env):
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
        self.n_channels = n_channels
        self.average_dice = average_dice
        self.django_env = django_env

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

            # get result
            result = np.argmax(result, axis=-1)
            result = result.astype(np.uint8)

            # Compute dice score
            if self.average_dice:
                dice = 0
                for l in range(1, self.n_classes):
                    dice += 2 * np.logical_and(self.label==l, result==l).sum() / float((self.label==l).sum() + (result==l).sum())
                dice /= float(self.n_classes-1)
            else:
                dice = 2 * np.logical_and(self.label==result, (self.label+result)>0).sum() / \
                       float((self.label>0).sum() + (result>0).sum())

            # save best model only
            if epoch == 0 or round(dice,5) > max(self.history['val_accuracy']):
                self.model.save(str(self.path_to_model))

            # add accuracy to history
            self.history['val_accuracy'].append(round(dice,5))
            self.history['accuracy'].append(round(logs["accuracy"],5))
            self.history['loss'].append(round(logs["loss"],5))
            logs["val_accuracy"] = max(self.history['val_accuracy'])
            if not self.django_env:
                save_history(self.history, self.path_to_model)

            # print accuracies
            print()
            print('val_acc (Dice):', self.history['val_accuracy'])
            print('train_acc:', self.history['accuracy'])
            print()

            # early stopping
            if self.early_stopping > 0 and max(self.history['val_accuracy']) not in self.history['val_accuracy'][-self.early_stopping:]:
                self.model.stop_training = True

def train_semantic_segmentation(bm,
    img_list, label_list,
    val_img_list, val_label_list,
    img=None, label=None,
    img_val=None, label_val=None,
    header=None, extension='.tif'):

    # training data
    img, label, allLabels, configuration_data, header, extension = load_training_data(bm.normalize,
                    img_list, label_list, bm.channels, bm.x_scale, bm.y_scale, bm.z_scale, bm.no_scaling, bm.crop_data,
                    bm.only, bm.ignore, img, label, None, None, header, extension)

    # img shape
    zsh, ysh, xsh = img.shape

    # validation data
    if any(val_img_list) or img_val is not None:
        img_val, label_val, _, _, _, _ = load_training_data(bm.normalize,
                        val_img_list, val_label_list, bm.channels, bm.x_scale, bm.y_scale, bm.z_scale, bm.no_scaling, bm.crop_data,
                        bm.only, bm.ignore, img_val, label_val, configuration_data, allLabels)

    elif bm.validation_split:
        split = round(zsh * bm.validation_split)
        img_val = np.copy(img[split:])
        label_val = np.copy(label[split:])
        img = np.copy(img[:split])
        label = np.copy(label[:split])
        zsh, ysh, xsh = img.shape

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
        zsh_val, ysh_val, xsh_val = img_val.shape

        # list of validation IDs
        list_IDs_val_fg, list_IDs_val_bg = [], []

        # get validation IDs of patches
        if bm.balance and bm.val_tf:
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
              'augment': (bm.flip_x, bm.flip_y, bm.flip_z, bm.swapaxes, bm.rotate)}

    # data generator
    validation_generator = None
    training_generator = DataGenerator(img, label, list_IDs_fg, list_IDs_bg, True, True, False, **params)
    if img_val is not None:
        if bm.val_tf:
            params['dim_img'] = (zsh_val, ysh_val, xsh_val)
            params['augment'] = (False, False, False, False, 0)
            validation_generator = DataGenerator(img_val, label_val, list_IDs_val_fg, list_IDs_val_bg, True, False, False, **params)
        else:
            metrics = Metrics(img_val, label_val, list_IDs_val_fg, (bm.z_patch, bm.y_patch, bm.x_patch), (zsh_val, ysh_val, xsh_val), bm.batch_size,
                              bm.path_to_model, bm.early_stopping, bm.validation_freq, nb_labels, bm.channels, bm.average_dice, bm.django_env)

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
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

    # save meta data
    meta_data = MetaData(bm.path_to_model, configuration_data, allLabels, extension, header, bm.crop_data, bm.cropping_weights, bm.cropping_config)

    # model checkpoint
    if img_val is not None:
        if bm.val_tf:
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
            callbacks = [metrics, meta_data]
    else:
        callbacks = [ModelCheckpoint(filepath=str(bm.path_to_model)), meta_data]

    # custom callback
    if bm.django_env:
        callbacks.insert(-1, CustomCallback(bm.image.id, bm.epochs))

    # train model
    model.fit(training_generator,
              epochs=bm.epochs,
              validation_data=validation_generator,
              callbacks=callbacks,
              workers=bm.workers)

    # save monitoring figure on train end
    if img_val is not None and bm.val_tf:
        save_history(history.history, bm.path_to_model)

def load_prediction_data(path_to_img, channels, x_scale, y_scale, z_scale,
                        no_scaling, normalize, mu, sig, region_of_interest,
                        img, img_header, img_extension):

    # read image data
    if img is None:
        img, img_header, img_extension = load_data(path_to_img, 'first_queue', return_extension=True)
    if img is None:
        InputError.message = "Invalid image data %s." %(os.path.basename(path_to_img))
        raise InputError()
    if img_extension != '.am':
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
    if not no_scaling:
        img = img_resize(img, z_scale, y_scale, x_scale)
    img -= np.amin(img)
    img /= np.amax(img)
    if normalize:
        mu_tmp, sig_tmp = np.mean(img), np.std(img)
        img = (img - mu_tmp) / sig_tmp
        img = img * sig + mu
        img[img<0] = 0
        img[img>1] = 1

    return img, img_header, z_shape, y_shape, x_shape, region_of_interest

def predict_semantic_segmentation(bm, img, path_to_model,
    z_patch, y_patch, x_patch, z_shape, y_shape, x_shape, compress, header,
    img_header, channels, stride_size, allLabels, batch_size, region_of_interest,
    no_scaling):

    results = {}

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
    predict_generator = PredictDataGenerator(img, list_IDs, **params)

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
            probs = np.zeros((z_shape, y_shape, x_shape, nb_labels), dtype=np.float32)
            for k in range(nb_labels):
                probs[:,:,:,k] = img_resize(probabilities[:,:,:,k], z_shape, y_shape, x_shape)
            probabilities = probs
        if np.any(region_of_interest):
            min_z,max_z,min_y,max_y,min_x,max_x,original_zsh,original_ysh,original_xsh = region_of_interest[:]
            tmp = np.zeros((original_zsh, original_ysh, original_xsh, nb_labels), dtype=np.float32)
            tmp[min_z:max_z,min_y:max_y,min_x:max_x] = probabilities
            probabilities = np.copy(tmp)
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
        label = np.copy(tmp)

    # get result
    label = get_labels(label, allLabels)
    results['regular'] = label

    # handle header
    if header is not None:
        header = get_image_dimensions(header, label)
        if img_header is not None:
            try:
                header = get_physical_size(header, img_header)
            except:
                pass
        header = [header]
        results['header'] = header

    # save result
    if bm.path_to_images:
        save_data(bm.path_to_final, label, header=header, compress=compress)

    return results

