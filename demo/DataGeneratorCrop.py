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

class DataGeneratorCrop(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, img, label, position, list_IDs_fg, list_IDs_bg, batch_size=32, dim=(32,32,32),
                 dim_img=(32,32,32), n_channels=3, n_classes=2, shuffle=True, augment=(False,False,False,0)):
        'Initialization'
        self.dim = dim
        self.dim_img = dim_img
        self.list_IDs_fg = list_IDs_fg
        self.list_IDs_bg = list_IDs_bg
        self.batch_size = batch_size
        self.label = label
        self.img = img
        self.position = position
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
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

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.uint8)
        y = np.empty((self.batch_size,), dtype=np.int32)
        tmp_X = np.empty((*self.dim, self.n_channels), dtype=np.uint8)
        tmp_y = np.empty(1, dtype=np.int32)

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
            k = ID
            l = self.dim_img[1]
            m = self.dim_img[2]

            # rotate patch
            if rotate:
                cos_a = cos_angle[i]
                sin_a = sin_angle[i]
                tmp_X = rotate_patch(self.img,tmp_X,k,l,m,cos_a,sin_a,
                    self.dim[0],self.dim[1],
                    self.dim_img[1],self.dim_img[2])
                tmp_y = rotate_patch(self.label,tmp_y,k,l,m,cos_a,sin_a,
                    self.dim[0],self.dim[1],
                    self.dim_img[1],self.dim_img[2])
            else:
                tmp_X = self.img[ID,...]
                tmp_y = self.label[ID]

            # flip patch
            v = np.random.randint(n_aug+1)
            if np.any([flip_x, flip_y, flip_z]) and v>0:
                flip = flips[v-1]
                X[i,...] = np.flip(tmp_X,flip)
                y[i] = np.flip(tmp_y,flip)
            else:
                X[i,...] = tmp_X
                y[i] = tmp_y

        return X, y

