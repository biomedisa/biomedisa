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

import os
from biomedisa_features.biomedisa_helper import img_resize, load_data, save_data, set_labels_to_zero
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
from tensorflow.keras.applications import DenseNet121, densenet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from biomedisa_features.DataGeneratorCrop import DataGeneratorCrop
from biomedisa_features.PredictDataGeneratorCrop import PredictDataGeneratorCrop
import tensorflow as tf
import numpy as np
from glob import glob
import h5py
import tarfile
import matplotlib.pyplot as plt

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
    plt.savefig(path_to_model.replace(".h5","_acc.png"), dpi=300, bbox_inches='tight')
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
    plt.savefig(path_to_model.replace(".h5","_loss.png"), dpi=300, bbox_inches='tight')
    plt.clf()

def make_densenet(inputshape):
    base_model = DenseNet121(
        input_tensor=Input(inputshape),
        include_top=False,)
    
    base_model.trainable= False
    
    inputs = Input(inputshape)
    x = densenet.preprocess_input(inputs)

    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model

def load_cropping_training_data(normalize, img_list, label_list, x_scale, y_scale, z_scale,
    labels_to_compute, labels_to_remove, mu=None, sig=None):

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
    if a is None:
        InputError.message = "Invalid label data %s." %(os.path.basename(label_names[0]))
        raise InputError()
    a = a.astype(np.uint8)
    a = set_labels_to_zero(a, labels_to_compute, labels_to_remove)
    label_z = np.any(a,axis=(1,2))
    label_y = np.any(a,axis=(0,2))
    label_x = np.any(a,axis=(0,1))
    label = np.append(label_z,label_y,axis=0)
    label = np.append(label,label_x,axis=0)

    # load first img
    img, _ = load_data(img_names[0], 'first_queue')
    if img is None:
        InputError.message = "Invalid image data %s." %(os.path.basename(img_names[0]))
        raise InputError()
    img = img.astype(np.float32)
    img_z = img_resize(img, a.shape[0], y_scale, x_scale)
    img_y = np.swapaxes(img_resize(img, z_scale, a.shape[1], x_scale),0,1)
    img_x = np.swapaxes(img_resize(img, z_scale, y_scale, a.shape[2]),0,2)
    img = np.append(img_z,img_y,axis=0)
    img = np.append(img,img_x,axis=0)
    img -= np.amin(img)
    img /= np.amax(img)
    if mu is not None and normalize:
        mu_tmp, sig_tmp = np.mean(img), np.std(img)
        img = (img - mu_tmp) / sig_tmp
        img = img * sig + mu
        img[img<0] = 0
        img[img>1] = 1
    else:
        mu, sig = np.mean(img), np.std(img)
    img = np.uint8(img*255)

    for img_name, label_name in zip(img_names[1:], label_names[1:]):

        # append label
        a, _ = load_data(label_name, 'first_queue')
        if a is None:
            InputError.message = "Invalid label data %s." %(os.path.basename(name))
            raise InputError()
        a = a.astype(np.uint8)
        a = set_labels_to_zero(a, labels_to_compute, labels_to_remove)
        next_label_z = np.any(a,axis=(1,2))
        next_label_y = np.any(a,axis=(0,2))
        next_label_x = np.any(a,axis=(0,1))
        label = np.append(label,next_label_z,axis=0)
        label = np.append(label,next_label_y,axis=0)
        label = np.append(label,next_label_x,axis=0)

        # append image
        a, _ = load_data(img_name, 'first_queue')
        if a is None:
            InputError.message = "Invalid image data %s." %(os.path.basename(name))
            raise InputError()
        a = a.astype(np.float32)
        img_z = img_resize(a, a.shape[0], y_scale, x_scale)
        img_y = np.swapaxes(img_resize(a, z_scale, a.shape[1], x_scale),0,1)
        img_x = np.swapaxes(img_resize(a, z_scale, y_scale, a.shape[2]),0,2)
        next_img = np.append(img_z,img_y,axis=0)
        next_img = np.append(next_img,img_x,axis=0)
        next_img -= np.amin(next_img)
        next_img /= np.amax(next_img)
        if normalize:
            mu_tmp, sig_tmp = np.mean(next_img), np.std(next_img)
            next_img = (next_img - mu_tmp) / sig_tmp
            next_img = next_img * sig + mu
            next_img[next_img<0] = 0
            next_img[next_img>1] = 1
        next_img = np.uint8(next_img*255)
        img = np.append(img, next_img, axis=0)

    img_rgb = np.empty((img.shape + (3,)), dtype=np.uint8)
    for i in range(3):
        img_rgb[...,i] = img

    # compute position data
    position = None

    return img_rgb, label, position, mu, sig, header, extension, len(img_names)

def train_cropping(img, label, path_to_model, epochs, batch_size,
                    validation_split, position, flip_x, flip_y, flip_z, rotate,
                    img_val, label_val, position_val):

    # img shape
    zsh, ysh, xsh, channels = img.shape

    # list of IDs
    list_IDs_fg = list(np.where(label)[0])
    list_IDs_bg = list(np.where(label==False)[0])

    # validation data
    if np.any(img_val):
        list_IDs_val_fg = list(np.where(label_val)[0])
        list_IDs_val_bg = list(np.where(label_val==False)[0])
    elif validation_split:
        split_fg = int(len(list_IDs_fg) * validation_split)
        split_bg = int(len(list_IDs_bg) * validation_split)
        list_IDs_val_fg = list_IDs_fg[split_fg:]
        list_IDs_val_bg = list_IDs_bg[split_bg:]
        list_IDs_fg = list_IDs_fg[:split_fg]
        list_IDs_bg = list_IDs_bg[:split_bg]

    # upsample IDs
    max_IDs = max(len(list_IDs_fg), len(list_IDs_bg))
    tmp_fg = []
    while len(tmp_fg) < max_IDs:
        tmp_fg.extend(list_IDs_fg)
        tmp_fg = tmp_fg[:max_IDs]
    list_IDs_fg = tmp_fg[:]

    tmp_bg = []
    while len(tmp_bg) < max_IDs:
        tmp_bg.extend(list_IDs_bg)
        tmp_bg = tmp_bg[:max_IDs]
    list_IDs_bg = tmp_bg[:]

    # validation data
    if np.any(img_val) or validation_split:
        max_IDs = max(len(list_IDs_val_fg), len(list_IDs_val_bg))
        tmp_fg = []
        while len(tmp_fg) < max_IDs:
            tmp_fg.extend(list_IDs_val_fg)
            tmp_fg = tmp_fg[:max_IDs]
        list_IDs_val_fg = tmp_fg[:]
        tmp_bg = []

        while len(tmp_bg) < max_IDs:
            tmp_bg.extend(list_IDs_val_bg)
            tmp_bg = tmp_bg[:max_IDs]
        list_IDs_val_bg = tmp_bg[:]

    # input shape
    input_shape = (ysh, xsh, channels)

    # parameters
    params = {'dim': (ysh, xsh),
              'batch_size': batch_size,
              'n_classes': 2,
              'n_channels': channels,
              'shuffle': True}

    # validation parameters
    params_val = {'dim': (ysh, xsh),
                  'batch_size': batch_size,
                  'n_classes': 2,
                  'n_channels': channels,
                  'shuffle': False}

    # data generator
    training_generator = DataGeneratorCrop(img, label, position, list_IDs_fg, list_IDs_bg, **params)
    if np.any(img_val):
        validation_generator = DataGeneratorCrop(img_val, label_val, position_val, list_IDs_val_fg, list_IDs_val_bg, **params_val)
    elif validation_split:
        validation_generator = DataGeneratorCrop(img, label, position, list_IDs_val_fg, list_IDs_val_bg, **params_val)
    else:
        validation_generator = None

    # create a MirroredStrategy
    if os.name == 'nt':
        cdo = tf.distribute.HierarchicalCopyAllReduce()
    else:
        cdo = tf.distribute.NcclAllReduce()
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=cdo)
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # create callback
    if np.any(img_val) or validation_split:
        save_best_only = True
    else:
        save_best_only = False
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(str(path_to_model), save_best_only=save_best_only)

    # compile model
    with strategy.scope():
        model = make_densenet(input_shape)
        model.compile(loss= tf.keras.losses.BinaryCrossentropy(),
                      optimizer= Adam(learning_rate=0.001),
                      metrics=['accuracy'])
    # train model
    history = model.fit(training_generator,
              validation_data=validation_generator,
              epochs=max(1,epochs),
              callbacks=checkpoint_cb)

    # save results in figure on train end
    if np.any(img_val) or validation_split:
        save_history(history.history, path_to_model.replace(".h5","_cropping.h5"))

    # compile model for finetunning
    with strategy.scope():
        model = load_model(str(path_to_model))
        model.trainable = True
        model.compile(loss= tf.keras.losses.BinaryCrossentropy(),
                      optimizer= Adam(learning_rate=1e-5),
                      metrics=['accuracy'])

    # finetune model
    history = model.fit(training_generator,
              validation_data=validation_generator,
              epochs=max(1,epochs),
              callbacks=checkpoint_cb)

    # save results in figure on train end
    if np.any(img_val) or validation_split:
        save_history(history.history, path_to_model.replace(".h5","_cropfine.h5"))

def load_data_to_crop(path_to_img, x_scale, y_scale, z_scale,
                        normalize, mu, sig, img):
    # read image data
    if img is None:
        img, _, _ = load_data(path_to_img, 'first_queue', return_extension=True)
    img_data = np.copy(img)
    if img is None:
        InputError.message = "Invalid image data %s." %(os.path.basename(path_to_img))
        raise InputError()
    z_shape, y_shape, x_shape = img.shape
    img = img.astype(np.float32)
    img_z = img_resize(img, z_shape, y_scale, x_scale)
    img_y = np.swapaxes(img_resize(img,z_scale,y_shape,x_scale),0,1)
    img_x = np.swapaxes(img_resize(img,z_scale,y_scale,x_shape),0,2)
    img = np.append(img_z,img_y,axis=0)
    img = np.append(img,img_x,axis=0)
    img -= np.amin(img)
    img /= np.amax(img)
    if normalize:
        mu_tmp, sig_tmp = np.mean(img), np.std(img)
        img = (img - mu_tmp) / sig_tmp
        img = img * sig + mu
        img[img<0] = 0
        img[img>1] = 1
    img = np.uint8(img*255)

    img_rgb = np.empty((img.shape + (3,)), dtype=np.uint8)
    for i in range(3):
        img_rgb[...,i] = img
    return img_rgb, z_shape, y_shape, x_shape, img_data

def crop_volume(img, path_to_model, path_to_final, z_shape, y_shape, x_shape, batch_size,
        debug_cropping, save_cropped, img_data=None, args=None, x_puffer=25, y_puffer=25, z_puffer=25):

    # img shape
    zsh, ysh, xsh, channels = img.shape

    # list of IDs
    list_IDs = [x for x in range(zsh)]

    # make length of list divisible by batch size
    rest = batch_size - (len(list_IDs) % batch_size)
    list_IDs = list_IDs + list_IDs[:rest]

    # parameters
    params = {'dim': (ysh,xsh),
              'dim_img': (zsh, ysh, xsh),
              'batch_size': batch_size,
              'n_channels': channels}

    # data generator
    predict_generator = PredictDataGeneratorCrop(img, list_IDs, **params)

    # create a MirroredStrategy
    if os.name == 'nt':
        cdo = tf.distribute.HierarchicalCopyAllReduce()
    else:
        cdo = tf.distribute.NcclAllReduce()
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=cdo)

    # input shape
    input_shape = (ysh, xsh, channels)

    # load model
    with strategy.scope():
        model = make_densenet(input_shape)

    # load weights
    hf = h5py.File(path_to_model, 'r')
    cropping_weights = hf.get('cropping_weights')
    iterator = 0
    for layer in model.layers:
        if layer.get_weights() != []:
            new_weights = []
            for arr in layer.get_weights():
                new_weights.append(cropping_weights.get(str(iterator)))
                iterator += 1
            layer.set_weights(new_weights)
    hf.close()

    # predict
    probabilities = model.predict(predict_generator, verbose=0, steps=None)
    probabilities = probabilities[:zsh]
    probabilities = np.ravel(probabilities)

    # plot prediction
    if debug_cropping and path_to_final:
        import matplotlib.pyplot as plt
        import matplotlib
        x = range(len(probabilities))
        y = list(probabilities)
        plt.plot(x, y)

    # create mask
    probabilities[probabilities > 0.5] = 1
    probabilities[probabilities <= 0.5] = 0

    # remove outliers
    for k in range(4,zsh-4):
        if np.all(probabilities[k-1:k+2] == np.array([0,1,0])):
            probabilities[k-1:k+2] = 0
        elif np.all(probabilities[k-2:k+2] == np.array([0,1,1,0])):
            probabilities[k-2:k+2] = 0
        elif np.all(probabilities[k-2:k+3] == np.array([0,1,1,1,0])):
            probabilities[k-2:k+3] = 0
        elif np.all(probabilities[k-3:k+3] == np.array([0,1,1,1,1,0])):
            probabilities[k-3:k+3] = 0
        elif np.all(probabilities[k-3:k+4] == np.array([0,1,1,1,1,1,0])):
            probabilities[k-3:k+4] = 0
        elif np.all(probabilities[k-4:k+4] == np.array([0,1,1,1,1,1,1,0])):
            probabilities[k-4:k+4] = 0
        elif np.all(probabilities[k-4:k+5] == np.array([0,1,1,1,1,1,1,1,0])):
            probabilities[k-4:k+5] = 0

    # create final
    if args.z_range is not None:
        z_lower, z_upper = args.z_range
    else:
        z_lower = max(0,np.argmax(probabilities[:z_shape]) - z_puffer)
        z_upper = min(z_shape,z_shape - np.argmax(np.flip(probabilities[:z_shape])) + z_puffer +1)

    if args.y_range is not None:
        y_lower, y_upper = args.y_range
    else:
        y_lower = max(0,np.argmax(probabilities[z_shape:z_shape+y_shape]) - y_puffer)
        y_upper = min(y_shape,y_shape - np.argmax(np.flip(probabilities[z_shape:z_shape+y_shape])) + y_puffer +1)

    if args.x_range is not None:
        x_lower, x_upper = args.x_range
    else:
        x_lower = max(0,np.argmax(probabilities[z_shape+y_shape:]) - x_puffer)
        x_upper = min(x_shape,x_shape - np.argmax(np.flip(probabilities[z_shape+y_shape:])) + x_puffer +1)

    # plot result
    if debug_cropping and path_to_final:
        y = np.zeros_like(probabilities)
        y[z_lower:z_upper] = 1
        y[z_shape+y_lower:z_shape+y_upper] = 1
        y[z_shape+y_shape+x_lower:z_shape+y_shape+x_upper] = 1
        plt.plot(x, y, '--')
        plt.tight_layout()  # To prevent overlapping of subplots
        #matplotlib.use("GTK3Agg")
        plt.savefig(path_to_final.replace('.tif','.png'), dpi=300)

    # crop image data
    cropped_volume = img_data[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper]
    if save_cropped and path_to_final:
        save_data(path_to_final, cropped_volume, compress=False)

    return z_lower, z_upper, y_lower, y_upper, x_lower, x_upper, cropped_volume

#=====================
# main functions
#=====================

def load_and_train(normalize,path_to_img,path_to_labels,path_to_model,
                epochs,batch_size,validation_split,x_scale,y_scale,z_scale,
                flip_x,flip_y,flip_z,rotate,labels_to_compute,labels_to_remove,
                path_val_img=[None],path_val_labels=[None],demo=False):

    # load training data
    img_val, labelData_val, position_val = None, None, None
    img, labelData, position, mu, sig, header, extension, number_of_images = load_cropping_training_data(normalize,
                        path_to_img, path_to_labels, x_scale, y_scale, z_scale, labels_to_compute, labels_to_remove)

    # load validation data
    if any(path_val_img) and any(path_val_labels):
        img_val, labelData_val, position_val, _, _, _, _, _ = load_cropping_training_data(normalize,
                            path_val_img, path_val_labels, x_scale, y_scale, z_scale, labels_to_compute, labels_to_remove, mu, sig)

    # force validation_split for large number of training images
    if number_of_images > 20 and not demo:
        if validation_split == 0:
            validation_split = 0.8

    # train cropping
    train_cropping(img, labelData, path_to_model, epochs,
                    batch_size, validation_split, position,
                    flip_x, flip_y, flip_z, rotate,
                    img_val, labelData_val, position_val)

    # load weights
    model = load_model(str(path_to_model))
    cropping_weights = []
    for layer in model.layers:
        if layer.get_weights() != []:
            for arr in layer.get_weights():
                cropping_weights.append(arr)

    # configuration data
    cropping_config = np.array([3, x_scale, y_scale, z_scale, normalize, mu, sig])

    return cropping_weights, cropping_config

def crop_data(path_to_data, path_to_model, path_to_cropped_image, batch_size,
    debug_cropping=False, save_cropped=True, img_data=None, args=None):

    # get meta data
    hf = h5py.File(path_to_model, 'r')
    meta = hf.get('cropping_meta')
    configuration = meta.get('configuration')
    channels, x_scale, y_scale, z_scale, normalize, mu, sig = np.array(configuration)[:]
    channels, x_scale, y_scale, z_scale, normalize, mu, sig = int(channels), int(x_scale), \
                            int(y_scale), int(z_scale), int(normalize), float(mu), float(sig)
    hf.close()

    # load data
    img, z_shape, y_shape, x_shape, img_data = load_data_to_crop(path_to_data,
                    x_scale, y_scale, z_scale, normalize, mu, sig, img_data)

    # make prediction
    z_lower, z_upper, y_lower, y_upper, x_lower, x_upper, cropped_volume = crop_volume(img, path_to_model,
        path_to_cropped_image, z_shape, y_shape, x_shape, batch_size, debug_cropping, save_cropped, img_data, args)

    # region of interest
    region_of_interest = np.array([z_lower, z_upper, y_lower, y_upper, x_lower, x_upper])

    return region_of_interest, cropped_volume

