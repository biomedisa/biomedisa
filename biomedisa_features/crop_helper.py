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
from biomedisa_helper import img_resize, load_data, save_data
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
from tensorflow.keras.applications import DenseNet121, densenet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.applications import DenseNet121, densenet
from DataGeneratorCrop import DataGeneratorCrop
from PredictDataGeneratorCrop import PredictDataGeneratorCrop
import tensorflow as tf
import numpy as np
from glob import glob
import h5py

class InputError(Exception):
    def __init__(self, message=None):
        self.message = message

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

def load_cropping_training_data(normalize, img_list, label_list, x_scale, y_scale, z_scale, mu=None, sig=None):

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
                    validation_split, position, flip_x, flip_y, flip_z, rotate):

    # img shape
    zsh, ysh, xsh, channels = img.shape

    # list of IDs
    list_IDs_fg = list(np.where(label)[0])
    list_IDs_bg = list(np.where(label==False)[0])

    # input shape
    input_shape = (ysh, xsh, channels)

    # parameters
    params = {'dim': (ysh, xsh),
              'dim_img': (zsh, ysh, xsh),
              'batch_size': batch_size,
              'n_classes': 2,
              'n_channels': channels,
              'shuffle': True,
              'augment': (flip_x, flip_y, flip_z, rotate)}

    # data generator
    if validation_split:
        split_IDs = int(zsh * validation_split)
        list_IDs_fg = list(np.where(label[:split_IDs])[0])
        list_IDs_bg = list(np.where(label[:split_IDs]==False)[0])
        list_IDs_val_fg = list(np.where(label[split_IDs:])[0] + split_IDs)
        list_IDs_val_bg = list(np.where(label[split_IDs:]==False)[0] + split_IDs)
        training_generator = DataGeneratorCrop(img, label, position, list_IDs_fg, list_IDs_bg, **params)
        validation_generator = DataGeneratorCrop(img, label, position, list_IDs_val_fg, list_IDs_val_bg, **params)
    else:
        training_generator = DataGeneratorCrop(img, label, position, list_IDs_fg, list_IDs_bg, **params)
        validation_generator = None

    # create a MirroredStrategy
    if os.name == 'nt':
        cdo = tf.distribute.HierarchicalCopyAllReduce()
    else:
        cdo = tf.distribute.NcclAllReduce()
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=cdo)
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # create callback
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(str(path_to_model), save_best_only=True)

    # compile model
    with strategy.scope():
        model = make_densenet(input_shape)
        model.compile(loss= tf.keras.losses.BinaryCrossentropy(),
                      optimizer= Adam(learning_rate=0.001),
                      metrics=['accuracy'])
    # train model
    model.fit(training_generator,
              validation_data=validation_generator,
              epochs=epochs//4,
              callbacks=checkpoint_cb)

    # compile model for finetunning
    with strategy.scope():
        model = load_model(str(path_to_model))
        model.trainable = True
        model.compile(loss= tf.keras.losses.BinaryCrossentropy(),
                      optimizer= Adam(learning_rate=1e-5),
                      metrics=['accuracy'])

    # finetune model
    model.fit(training_generator,
              validation_data=validation_generator,
              epochs=epochs//4,
              callbacks=checkpoint_cb)

def evaluate_crop(img,labelData,position,batch_size,flip_x,flip_y,flip_z,rotate,path_to_model):
    # img shape
    zsh, ysh, xsh, channels = img.shape
    # list of IDs
    list_IDs = [x for x in range(zsh)]

    # parameters
    params = {'dim': (ysh, xsh),
              'dim_img': (zsh, ysh, xsh),
              'batch_size': batch_size,
              'n_classes': 2,
              'n_channels': channels,
              'shuffle': True,
              'augment': (flip_x, flip_y, flip_z, rotate)}
    # generate data
    training_generator = DataGeneratorCrop(img, labelData, position, list_IDs, [], **params)
    # load and evaluate model
    model = tf.keras.models.load_model(path_to_model)
    score = model.evaluate(training_generator,batch_size=batch_size,verbose=1)
    print(score)

def load_data_to_crop(path_to_img, x_scale, y_scale, z_scale,
                        normalize, mu, sig):
    # read image data
    img, _, img_ext = load_data(path_to_img, 'first_queue', return_extension=True)
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
    return img_rgb, z_shape, y_shape, x_shape

def crop_volume(img, path_to_volume, path_to_model, z_shape, y_shape, x_shape, batch_size, debug_cropping,
        x_puffer=25,y_puffer=25,z_puffer=25):

    # path to cropped image
    filename = os.path.basename(path_to_volume)
    filename = os.path.splitext(filename)[0]
    if filename[-4:] in ['.nii']:
        filename = filename[:-4]
    filename = filename + '_cropped.tif'
    path_to_final = path_to_volume.replace(os.path.basename(path_to_volume), filename)

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
    if debug_cropping:
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

    # plot cleaned result
    if debug_cropping:
        y = list(probabilities)
        plt.plot(x, y, '--')
        plt.tight_layout()  # To prevent overlapping of subplots
        matplotlib.use("GTK3Agg")
        plt.savefig(path_to_final.replace('.tif','.png'), dpi=300)

    # create final
    z_upper = max(0,np.argmax(probabilities[:z_shape]) - z_puffer)
    z_lower = min(z_shape,z_shape - np.argmax(np.flip(probabilities[:z_shape])) + z_puffer +1)
    y_upper = max(0,np.argmax(probabilities[z_shape:z_shape+y_shape]) - y_puffer)
    y_lower = min(y_shape,y_shape - np.argmax(np.flip(probabilities[z_shape:z_shape+y_shape])) + y_puffer +1)
    x_upper = max(0,np.argmax(probabilities[z_shape+y_shape:]) - x_puffer)
    x_lower = min(x_shape,x_shape - np.argmax(np.flip(probabilities[z_shape+y_shape:])) + x_puffer +1)

    # crop image data
    if debug_cropping:
        volume, _ = load_data(path_to_volume)
        final = volume[z_upper:z_lower,y_upper:y_lower,x_upper:x_lower]
        save_data(path_to_final, final, compress=False)

    return z_upper, z_lower, y_upper, y_lower, x_upper, x_lower

#=====================
# main functions
#=====================

def load_and_train(normalize,path_to_img,path_to_labels,path_to_model,
                epochs,batch_size,validation_split,x_scale,y_scale,z_scale,
                flip_x,flip_y,flip_z,rotate):

    # load training data
    img, labelData, position, mu, sig, header, extension, number_of_images = load_cropping_training_data(normalize,
                        path_to_img, path_to_labels, x_scale, y_scale, z_scale)

    # force validation_split for large number of training images
    if number_of_images > 20:
        if validation_split == 0:
            validation_split = 0.8
        #early_stopping = True

    # train cropping
    train_cropping(img, labelData, path_to_model, epochs,
                    batch_size, validation_split, position,
                    flip_x, flip_y, flip_z, rotate)

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

def crop_data(path_to_data, path_to_model, batch_size, debug_cropping):

    # get meta data
    hf = h5py.File(path_to_model, 'r')
    meta = hf.get('cropping_meta')
    configuration = meta.get('configuration')
    channels, x_scale, y_scale, z_scale, normalize, mu, sig = np.array(configuration)[:]
    channels, x_scale, y_scale, z_scale, normalize, mu, sig = int(channels), int(x_scale), \
                            int(y_scale), int(z_scale), int(normalize), float(mu), float(sig)
    hf.close()

    # load data
    img, z_shape, y_shape, x_shape = load_data_to_crop(path_to_data,
                    x_scale, y_scale, z_scale, normalize, mu, sig)

    # make prediction
    z_upper, z_lower, y_upper, y_lower, x_upper, x_lower = crop_volume(img, path_to_data, path_to_model,
        z_shape, y_shape, x_shape, batch_size, debug_cropping)

    # region of interest
    region_of_interest = np.array([z_upper, z_lower, y_upper, y_lower, x_upper, x_lower])

    return region_of_interest

def evaluate_network(normalize,path_to_img,path_to_labels,path_to_model,
                batch_size,x_scale,y_scale,z_scale,
                flip_x,flip_y,flip_z,rotate):

    img, labelData, position, mu, sig, header, extension, number_of_images = load_cropping_training_data(normalize,
                path_to_img, path_to_labels, x_scale, y_scale, z_scale)

    evaluate_crop(img,labelData,position,batch_size,flip_x,flip_y,flip_z,rotate,path_to_model)

