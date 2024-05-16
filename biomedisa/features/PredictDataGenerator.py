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

import numpy as np
import tensorflow as tf

class PredictDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img, list_IDs, batch_size=32, dim=(32,32,32),
                 dim_img=(32,32,32), n_channels=1, patch_normalization=False):
        'Initialization'
        self.dim = dim
        self.dim_img = dim_img
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.img = img
        self.n_channels = n_channels
        self.indexes = np.arange(len(self.list_IDs))
        self.patch_normalization = patch_normalization

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # get patch indices
            k = ID // (self.dim_img[1]*self.dim_img[2])
            rest = ID % (self.dim_img[1]*self.dim_img[2])
            l = rest // self.dim_img[2]
            m = rest % self.dim_img[2]

            # get patch
            tmp_X = self.img[k:k+self.dim[0],l:l+self.dim[1],m:m+self.dim[2]]
            if self.patch_normalization:
                tmp_X = np.copy(tmp_X, order='C')
                for c in range(self.n_channels):
                    tmp_X[:,:,:,c] -= np.mean(tmp_X[:,:,:,c])
                    tmp_X[:,:,:,c] /= max(np.std(tmp_X[:,:,:,c]), 1e-6)
            X[i] = tmp_X

        return X

