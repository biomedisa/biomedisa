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

import django
django.setup()
from biomedisa_app.models import Upload
from biomedisa_features.biomedisa_helper import img_resize, load_data, save_data
from keras.optimizers import SGD
from keras.models import Model, load_model
from keras.layers import (
    Input, Conv3D, MaxPooling3D, UpSampling3D, Activation, Reshape,
    BatchNormalization, Concatenate)
from keras.utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint
from DataGenerator import DataGenerator
from PredictDataGenerator import PredictDataGenerator
import tensorflow as tf
import numpy as np
import cv2
from random import shuffle
from glob import glob
import random
import numba
import re
import os
import time

class InputError(Exception):
    def __init__(self, message=None):
        self.message = message

def predict_blocksize(labelData):
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
    argmin_x = argmin_x - 64 if argmin_x - 64 > 0 else 0
    argmax_x = argmax_x + 64 if argmax_x + 64 < xsh else xsh
    argmin_y = argmin_y - 64 if argmin_y - 64 > 0 else 0
    argmax_y = argmax_y + 64 if argmax_y + 64 < ysh else ysh
    argmin_z = argmin_z - 64 if argmin_z - 64 > 0 else 0
    argmax_z = argmax_z + 64 if argmax_z + 64 < zsh else zsh
    return argmin_z,argmax_z,argmin_y,argmax_y,argmin_x,argmax_x

def get_image_dimensions(header, data):

    # read header as string
    b = header.tobytes()
    s = b.decode("utf-8")

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
    s = b.decode("utf-8")

    # get physical size from image header
    lattice = re.search('BoundingBox (.*),\n', s)
    lattice = lattice.group(1)
    i0, i1, i2, i3, i4, i5 = lattice.split(' ')
    bounding_box_i = re.search('&BoundingBox (.*),\n', s)
    bounding_box_i = bounding_box_i.group(1)

    # read header as string
    b = header.tobytes()
    s = b.decode("utf-8")

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
        x = BatchNormalization(name=name)(x)
        x = Activation('relu')(x)
        return x

    x = make_stage(input_tensor, 1)
    x = make_stage(x, 2)
    return x

def make_unet(input_shape, nb_labels):

    nb_plans, nb_rows, nb_cols, _ = input_shape

    inputs = Input(input_shape)
    conv1 = make_conv_block(32, inputs, 1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = make_conv_block(64, pool1, 2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = make_conv_block(128, pool2, 3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = make_conv_block(256, pool3, 4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = make_conv_block(512, pool4, 5)
    pool5 = MaxPooling3D(pool_size=(2, 2, 2))(conv5)

    conv6 = make_conv_block(1024, pool5, 6)

    up7 = Concatenate()([UpSampling3D(size=(2, 2, 2))(conv6), conv5])
    conv7 = make_conv_block(512, up7, 7)

    up8 = Concatenate()([UpSampling3D(size=(2, 2, 2))(conv7), conv4])
    conv8 = make_conv_block(256, up8, 8)

    up9 = Concatenate()([UpSampling3D(size=(2, 2, 2))(conv8), conv3])
    conv9 = make_conv_block(128, up9, 9)

    up10 = Concatenate()([UpSampling3D(size=(2, 2, 2))(conv9), conv2])
    conv10 = make_conv_block(64, up10, 10)

    up11 = Concatenate()([UpSampling3D(size=(2, 2, 2))(conv10), conv1])
    conv11 = make_conv_block(32, up11, 11)

    conv12 = Conv3D(nb_labels, (1, 1, 1), name='conv_12_1')(conv11)

    x = Reshape((nb_plans * nb_rows * nb_cols, nb_labels))(conv12)
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

def load_training_data(normalize, img_list, label_list, channels,
                      x_scale, y_scale, z_scale, crop_data):

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
                tmp_img_names = glob(img_dir+'/*/*'+data_type)+glob(img_dir+'/*'+data_type)
                tmp_label_names = glob(label_dir+'/*/*'+data_type)+glob(label_dir+'/*'+data_type)
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
    region_of_interest = None
    a, header, extension = load_data(label_names[0], 'first_queue', True)
    if a is None:
        InputError.message = "Invalid label data %s." %(os.path.basename(label_names[0]))
        raise InputError()
    if crop_data:
        region_of_interest = np.zeros(6)
        argmin_z,argmax_z,argmin_y,argmax_y,argmin_x,argmax_x=predict_blocksize(a)
        a = np.copy(a[argmin_z:argmax_z,argmin_y:argmax_y,argmin_x:argmax_x], order='C')
        region_of_interest += [argmin_z,argmax_z,argmin_y,argmax_y,argmin_x,argmax_x]
    a = a.astype(np.uint8)
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
        if crop_data:
            argmin_z,argmax_z,argmin_y,argmax_y,argmin_x,argmax_x=predict_blocksize(a)
            a = np.copy(a[argmin_z:argmax_z,argmin_y:argmax_y,argmin_x:argmax_x], order='C')
            region_of_interest += [argmin_z,argmax_z,argmin_y,argmax_y,argmin_x,argmax_x]
        a = a.astype(np.uint8)
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

    # automatic cropping
    if crop_data:
        region_of_interest /= float(len(img_names))
        region_of_interest = np.round(region_of_interest)
        region_of_interest[region_of_interest<0] = 0
        region_of_interest = region_of_interest.astype(int)

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

    # labels must be in ascending order
    allLabels = np.unique(label)
    for k, l in enumerate(allLabels):
        label[label==l] = k

    return img, label, position, allLabels, mu, sig, header, extension, region_of_interest

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

def train_semantic_segmentaion(img, label, path_to_model, z_patch, y_patch, x_patch, allLabels, epochs,
                        batch_size, channels, validation_split, stride_size, balance, position,
                        flip_x, flip_y, flip_z, rotate, image):

    # img shape
    zsh, ysh, xsh = img.shape

    # list of IDs
    list_IDs_fg, list_IDs_bg = [], []

    # get IDs of patches
    if balance:
        for k in range(0, zsh-z_patch+1, stride_size):
            for l in range(0, ysh-y_patch+1, stride_size):
                for m in range(0, xsh-x_patch+1, stride_size):
                    if np.any(label[k:k+z_patch, l:l+y_patch, m:m+x_patch]):
                        list_IDs_fg.append(k*ysh*xsh+l*xsh+m)
                    else:
                        list_IDs_bg.append(k*ysh*xsh+l*xsh+m)
    else:
        for k in range(0, zsh-z_patch+1, stride_size):
            for l in range(0, ysh-y_patch+1, stride_size):
                for m in range(0, xsh-x_patch+1, stride_size):
                    list_IDs_fg.append(k*ysh*xsh+l*xsh+m)

    # if balance, batch_size must be even
    if balance and batch_size % 2 > 0:
        batch_size -= 1

    # number of labels
    nb_labels = len(allLabels)

    # input shape
    input_shape = (z_patch, y_patch, x_patch, channels)

    # parameters
    params = {'dim': (z_patch, y_patch, x_patch),
              'dim_img': (zsh, ysh, xsh),
              'batch_size': batch_size,
              'n_classes': nb_labels,
              'n_channels': channels,
              'shuffle': True,
              'augment': (flip_x, flip_y, flip_z, rotate)}

    # data generator
    if validation_split:
        np.random.shuffle(list_IDs_fg)
        np.random.shuffle(list_IDs_bg)
        split_fg = int(len(list_IDs_fg) * validation_split)
        split_bg = int(len(list_IDs_bg) * validation_split)
        list_IDs_fg = list_IDs_fg.copy()
        list_IDs_bg = list_IDs_bg.copy()
        training_generator = DataGenerator(img, label, position, list_IDs_fg[:split_fg], list_IDs_bg[:split_bg], **params)
        validation_generator = DataGenerator(img, label, position, list_IDs_fg[split_fg:], list_IDs_bg[split_bg:], **params)
    else:
        training_generator = DataGenerator(img, label, position, list_IDs_fg, list_IDs_bg, **params)
        validation_generator = None

    # optimizer
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # compile model
    with strategy.scope():
        model = make_unet(input_shape, nb_labels)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

    # callbacks
    my_callbacks = [CustomCallback(image.id,epochs),
        ModelCheckpoint(filepath=str(path_to_model))]

    # train model
    model.fit(training_generator,
              epochs=epochs,
              validation_data=validation_generator,
              callbacks=[my_callbacks])
              #use_multiprocessing=True,
              #workers=6)

    # save model
    #model.save(str(path_to_model))

def load_prediction_data(path_to_img, channels, x_scale, y_scale, z_scale,
                        normalize, mu, sig, region_of_interest):

    # read image data
    img, img_header, img_ext = load_data(path_to_img, 'first_queue', return_extension=True)
    if img is None:
        InputError.message = "Invalid image data %s." %(os.path.basename(path_to_img))
        raise InputError()
    if img_ext != '.am':
        img_header = None
    z_shape, y_shape, x_shape = img.shape

    # automatic cropping of image to region of interest
    if np.any(region_of_interest):
        min_z, max_z, min_y, max_y, min_x, max_x = region_of_interest[:]
        min_z = min(min_z, z_shape)
        min_y = min(min_y, y_shape)
        min_x = min(min_x, x_shape)
        max_z = min(max_z, z_shape)
        max_y = min(max_y, y_shape)
        max_x = min(max_x, x_shape)
        if max_z-min_z < z_shape:
            min_z, max_z = 0, z_shape
        if max_y-min_y < y_shape:
            min_y, max_y = 0, y_shape
        if max_x-min_x < x_shape:
            min_x, max_x = 0, x_shape
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

    # list of IDs
    list_IDs = []

    # get nIds of patches
    for k in range(0, zsh-z_patch+1, stride_size):
        for l in range(0, ysh-y_patch+1, stride_size):
            for m in range(0, xsh-x_patch+1, stride_size):
                list_IDs.append(k*ysh*xsh+l*xsh+m)

    # make length of list divisible by batch size
    rest = batch_size - (len(list_IDs) % batch_size)
    list_IDs = list_IDs + list_IDs[:rest]

    # parameters
    params = {'dim': (z_patch, y_patch, x_patch),
              'dim_img': (zsh, ysh, xsh),
              'batch_size': batch_size,
              'n_channels': channels}

    # data generator
    predict_generator = PredictDataGenerator(img, position, list_IDs, **params)

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()

    # load model
    with strategy.scope():
        model = load_model(str(path_to_model))

    # predict
    probabilities = model.predict(predict_generator, verbose=0, steps=None)

    # create final
    final = np.zeros((zsh, ysh, xsh, probabilities.shape[4]), dtype=np.float32)
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
            header = get_physical_size(header, img_header)
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

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()

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
                tmp_img_names = glob(img_dir+'/*/*'+data_type)+glob(img_dir+'/*'+data_type)
                tmp_label_names = glob(label_dir+'/*/*'+data_type)+glob(label_dir+'/*'+data_type)
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

def train_semantic_segmentaion_refine(img, label, final, path_to_model, patch_size, \
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
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
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

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()

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
            header = get_physical_size(header, img_header)
    save_data(path_to_final, out, header=header, compress=compress)
