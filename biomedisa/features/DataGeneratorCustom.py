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
import tensorflow as tf
import numba
import random
import scipy

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, img, label, list_IDs_fg, list_IDs_bg, shuffle, train, classification, batch_size=32, dim=(32,32,32),
                 dim_img=(32,32,32), n_classes=10, n_channels=1, augment=(False,False,False,False,0,False), patch_normalization=False, separation=False):
        'Initialization'
        self.dim = dim
        self.dim_img = dim_img
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
        self.classification = classification
        self.on_epoch_end()
        self.patch_normalization = patch_normalization
        self.separation = separation

    def __len__(self):
        'Denotes the number of batches per epoch'
        if len(self.list_IDs_bg) > 0:
            len_IDs = max(len(self.list_IDs_fg), len(self.list_IDs_bg))
            n_batches = len_IDs // (self.batch_size // 2)
        else:
            len_IDs = len(self.list_IDs_fg)
            n_batches = len_IDs // self.batch_size
        return n_batches

    def __getitem__(self, index):
        'Generate one batch of data'
        if len(self.list_IDs_bg) > 0:
            # Generate indexes of the batch
            half_batch_size = self.batch_size // 2
            indexes_fg = self.indexes_fg[index*half_batch_size:(index+1)*half_batch_size]
            indexes_bg = self.indexes_bg[index*half_batch_size:(index+1)*half_batch_size]

            # Find list of IDs
            list_IDs_temp = [self.list_IDs_fg[k] for k in indexes_fg] + [self.list_IDs_bg[k] for k in indexes_bg]

        else:
            # Generate indexes of the batch
            indexes_fg = self.indexes_fg[index*self.batch_size:(index+1)*self.batch_size]

            # Find list of IDs
            list_IDs_temp = [self.list_IDs_fg[k] for k in indexes_fg]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if len(self.list_IDs_bg) > 0:
            # upsample lists of indexes
            indexes_fg = np.arange(len(self.list_IDs_fg))
            indexes_bg = np.arange(len(self.list_IDs_bg))
            len_IDs = max(len(self.list_IDs_fg), len(self.list_IDs_bg))
            repetitions = len_IDs // len(self.list_IDs_fg) + 1
            self.indexes_fg = np.tile(indexes_fg, repetitions)
            repetitions = len_IDs // len(self.list_IDs_bg) + 1
            self.indexes_bg = np.tile(indexes_bg, repetitions)
        else:
            self.indexes_fg = np.arange(len(self.list_IDs_fg))
        # shuffle indexes
        if self.shuffle == True:
            np.random.shuffle(self.indexes_fg)
            if len(self.list_IDs_bg) > 0:
                np.random.shuffle(self.indexes_bg)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, *self.dim, 2), dtype=np.int32)

        # get augmentation parameter
        flip_x, flip_y, flip_z, swapaxes, rotate, rotate3d = self.augment
        n_aug = np.sum([flip_z, flip_y, flip_x])
        flips =  np.where([flip_z, flip_y, flip_x])[0]

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # get patch indices
            k = ID // (self.dim_img[1]*self.dim_img[2])
            rest = ID % (self.dim_img[1]*self.dim_img[2])
            l = rest // self.dim_img[2]
            m = rest % self.dim_img[2]

            # get patch
            tmp_X = self.img[k:k+self.dim[0],l:l+self.dim[1],m:m+self.dim[2]]
            tmp_y = self.label[k:k+self.dim[0],l:l+self.dim[1],m:m+self.dim[2]]

            # center label gets value 1
            if self.separation:
                centerLabel = tmp_y[self.dim[0]//2,self.dim[1]//2,self.dim[2]//2]
                tmp_y = tmp_y.copy()
                tmp_y[tmp_y!=centerLabel]=0
                tmp_y[tmp_y==centerLabel]=1

            # augmentation
            if self.train:

                # flip patch along axes
                v = np.random.randint(n_aug+1)
                if np.any([flip_x, flip_y, flip_z]) and v>0:
                    flip = flips[v-1]
                    tmp_X = np.flip(tmp_X, flip)
                    tmp_y = np.flip(tmp_y, flip)

                # swap axes
                if swapaxes:
                    v = np.random.randint(4)
                    if v==1:
                        tmp_X = np.swapaxes(tmp_X,0,1)
                        tmp_y = np.swapaxes(tmp_y,0,1)
                    elif v==2:
                        tmp_X = np.swapaxes(tmp_X,0,2)
                        tmp_y = np.swapaxes(tmp_y,0,2)
                    elif v==3:
                        tmp_X = np.swapaxes(tmp_X,1,2)
                        tmp_y = np.swapaxes(tmp_y,1,2)

            # patch normalization
            if self.patch_normalization:
                tmp_X = tmp_X.copy().astype(np.float32)
                for c in range(self.n_channels):
                    tmp_X[:,:,:,c] -= np.mean(tmp_X[:,:,:,c])
                    tmp_X[:,:,:,c] /= max(np.std(tmp_X[:,:,:,c]), 1e-6)

            # assign to batch
            X[i] = tmp_X
            y[i] = tmp_y

        return X, y

