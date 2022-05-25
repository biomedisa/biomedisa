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

import numpy as np
import tensorflow as tf
import numba

@numba.jit(nopython=True)#parallel=True
def rotate_patch(src,trg,k,l,m,cos_a,sin_a,z_patch,y_patch,x_patch,imageHeight,imageWidth):
    for y in range(l,l+y_patch):
        yA = y - imageHeight/2
        for x in range(m,m+x_patch):
            xA = x - imageWidth/2
            xR = xA * cos_a - yA * sin_a
            yR = xA * sin_a + yA * cos_a
            src_x = xR + imageWidth/2
            src_y = yR + imageHeight/2
            # bilinear interpolation
            src_x0 = float(int(src_x))
            src_x1 = src_x0 + 1
            src_y0 = float(int(src_y))
            src_y1 = src_y0 + 1
            sx = src_x - src_x0
            sy = src_y - src_y0
            idx_src_x0 = int(min(max(0,src_x0),imageWidth-1))
            idx_src_x1 = int(min(max(0,src_x1),imageWidth-1))
            idx_src_y0 = int(min(max(0,src_y0),imageHeight-1))
            idx_src_y1 = int(min(max(0,src_y1),imageHeight-1))
            for z in range(k,k+z_patch):
                val  = (1-sy) * (1-sx) * float(src[z,idx_src_y0,idx_src_x0])
                val += (sy) * (1-sx) * float(src[z,idx_src_y1,idx_src_x0])
                val += (1-sy) * (sx) * float(src[z,idx_src_y0,idx_src_x1])
                val += (sy) * (sx) * float(src[z,idx_src_y1,idx_src_x1])
                trg[z-k,y-l,x-m] = val
    return trg

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, img, label, position, list_IDs, counts, shuffle, batch_size=32, dim=(32,32,32),
                 dim_img=(32,32,32), n_classes=10, n_channels=1, class_weights=False, augment=(False,False,False,0)):
        'Initialization'
        self.dim = dim
        self.dim_img = dim_img
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.label = label
        self.img = img
        self.position = position
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.counts = counts
        self.class_weights = class_weights
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        len_IDs = len(self.list_IDs)
        return int(np.floor(len_IDs / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y, sample_weights = self.__data_generation(list_IDs_temp)

        return X, y, sample_weights

    def on_epoch_end(self, logs={}):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, *self.dim, 1), dtype=np.int32)
        tmp_X = np.empty(self.dim, dtype=np.float32)
        tmp_y = np.empty(self.dim, dtype=np.int32)

        # get augmentation parameter
        flip_x, flip_y, flip_z, rotate = self.augment
        n_aug = np.sum([flip_z, flip_y, flip_x])
        flips =  np.where([flip_z, flip_y, flip_x])[0]

        # create random angles
        if rotate:
            angle = np.random.uniform(-1,1,self.batch_size) * 3.1416/180*rotate
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # get patch indices
            k = ID // (self.dim_img[1]*self.dim_img[2])
            rest = ID % (self.dim_img[1]*self.dim_img[2])
            l = rest // self.dim_img[2]
            m = rest % self.dim_img[2]

            # rotate patch
            if rotate:
                cos_a = cos_angle[i]
                sin_a = sin_angle[i]
                tmp_X = rotate_patch(self.img,tmp_X,k,l,m,cos_a,sin_a,
                    self.dim[0],self.dim[1],self.dim[2],
                    self.dim_img[1],self.dim_img[2])
                tmp_y = rotate_patch(self.label,tmp_y,k,l,m,cos_a,sin_a,
                    self.dim[0],self.dim[1],self.dim[2],
                    self.dim_img[1],self.dim_img[2])
            else:
                tmp_X = self.img[k:k+self.dim[0],l:l+self.dim[1],m:m+self.dim[2]]
                tmp_y = self.label[k:k+self.dim[0],l:l+self.dim[1],m:m+self.dim[2]]

            # flip patch
            v = np.random.randint(n_aug+1)
            if np.any([flip_x, flip_y, flip_z]) and v>0:
                flip = flips[v-1]
                X[i,:,:,:,0] = np.flip(tmp_X,flip)
                y[i,:,:,:,0] = np.flip(tmp_y,flip)
            else:
                X[i,:,:,:,0] = tmp_X
                y[i,:,:,:,0] = tmp_y

            if self.n_channels == 2:
                X[i,:,:,:,1] = self.position[k:k+self.dim[0],l:l+self.dim[1],m:m+self.dim[2]]

        # sample weights
        sample_weights = np.ones(y.shape, dtype=np.float32)
        if self.class_weights:
            counts_max = np.amax(self.counts)
            for i in range(self.n_classes):
                sample_weights[y==i] = counts_max / self.counts[i]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes), sample_weights

