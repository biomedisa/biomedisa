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

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, rotate
from tf_keras.utils import Sequence
from tf_keras.applications.densenet import preprocess_input
import random

def elastic_transform(image, alpha=100, sigma=20):
    ysh, xsh, csh = image.shape
    dx = gaussian_filter((np.random.rand(ysh, xsh) * 2 - 1) * alpha, sigma)
    dy = gaussian_filter((np.random.rand(ysh, xsh) * 2 - 1) * alpha, sigma)
    y, x = np.meshgrid(np.arange(ysh), np.arange(xsh), indexing='ij')
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    for k in range(csh):
        image[:,:,k] = map_coordinates(image[:,:,k], indices, order=0, mode='reflect').reshape(ysh, xsh)
    return image

class DataGeneratorCrop(Sequence):
    'Generates data for Keras'
    def __init__(self, img, label, list_IDs_fg, list_IDs_bg, batch_size=32,
            dim=(32,32,32), n_channels=3, n_classes=2, shuffle=True,
            augment=(False,False,False,0), train=True, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.list_IDs_fg = list_IDs_fg
        self.list_IDs_bg = list_IDs_bg
        self.batch_size = batch_size
        self.label = label
        self.img = img
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.train = train
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        if len(self.list_IDs_bg) > 0:
            len_IDs = 2 * min(len(self.list_IDs_fg), len(self.list_IDs_bg))
        else:
            len_IDs = len(self.list_IDs_fg)
        return int(np.floor(len_IDs / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        if len(self.list_IDs_bg) > 0:

            # len IDs
            len_IDs = min(len(self.list_IDs_fg), len(self.list_IDs_bg))

            # sample lists of indexes to the same size
            tmp_indexes_fg = self.indexes_fg[:len_IDs]
            tmp_indexes_bg = self.indexes_bg[:len_IDs]

            # Generate indexes of the batch
            tmp_batch_size = int(self.batch_size / 2)
            indexes_fg = tmp_indexes_fg[index*tmp_batch_size:(index+1)*tmp_batch_size]
            indexes_bg = tmp_indexes_bg[index*tmp_batch_size:(index+1)*tmp_batch_size]

            # Find list of IDs
            list_IDs_temp = [self.list_IDs_fg[k] for k in indexes_fg] + [self.list_IDs_bg[k] for k in indexes_bg]

        else:

            # Generate indexes of the batch
            indexes_fg = self.indexes_fg[index*self.batch_size:(index+1)*self.batch_size]

            # Find list of IDs
            list_IDs_temp = [self.list_IDs_fg[k] for k in indexes_fg]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        X = preprocess_input(X)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes_fg = np.arange(len(self.list_IDs_fg))
        self.indexes_bg = np.arange(len(self.list_IDs_bg))
        if self.shuffle == True:
            np.random.shuffle(self.indexes_fg)
            np.random.shuffle(self.indexes_bg)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # get augmentation parameter
        flip_x, flip_y, flip_z, rotation = self.augment
        elastic = False

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.uint8)
        y = np.empty((self.batch_size,), dtype=np.int32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            tmp_X = self.img[ID,...].copy()

            # augmentation
            if self.train and (any(self.augment) or elastic):
                if flip_x and np.random.randint(2) and abs(self.label[ID])!=3:
                    tmp_X = np.flip(tmp_X, 1)
                if flip_y and np.random.randint(2):
                    if abs(self.label[ID])==1:
                        tmp_X = np.flip(tmp_X, 0)
                    elif abs(self.label[ID])==3:
                        tmp_X = np.flip(tmp_X, 1)
                if flip_z and np.random.randint(2) and abs(self.label[ID])!=1:
                    tmp_X = np.flip(tmp_X, 0)
                if rotation:
                    angle = random.uniform(-rotation, rotation)
                    tmp_X = rotate(tmp_X, angle, order=0, mode='reflect', reshape=False)
                if elastic:
                    tmp_X = elastic_transform(tmp_X)

            X[i,...] = tmp_X
            y[i] = 0 if self.label[ID] < 0 else 1

        return X, y

