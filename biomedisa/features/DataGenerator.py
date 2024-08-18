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
import numba
import random
import scipy

@numba.jit(nopython=True)#parallel=True
def rotate_img_patch(src,trg,k,l,m,cos_a,sin_a,z_patch,y_patch,x_patch,imageHeight,imageWidth):
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

@numba.jit(nopython=True)#parallel=True
def rotate_label_patch(src,trg,k,l,m,cos_a,sin_a,z_patch,y_patch,x_patch,imageHeight,imageWidth):
    for y in range(l,l+y_patch):
        yA = y - imageHeight/2
        for x in range(m,m+x_patch):
            xA = x - imageWidth/2
            xR = xA * cos_a - yA * sin_a
            yR = xA * sin_a + yA * cos_a
            src_x = xR + imageWidth/2
            src_y = yR + imageHeight/2
            # nearest neighbour
            src_x = round(src_x)
            src_y = round(src_y)
            idx_src_x = int(min(max(0,src_x),imageWidth-1))
            idx_src_y = int(min(max(0,src_y),imageHeight-1))
            for z in range(k,k+z_patch):
                trg[z-k,y-l,x-m] = src[z,idx_src_y,idx_src_x]
    return trg

def random_rotation_3d(image, max_angle=180):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).

    Arguments:
    max_angle: `float`. The maximum rotation angle.

    Returns:
    batch of rotated 3D images
    """

    # rotate along x-axis
    angle = random.uniform(-max_angle, max_angle)
    image2 = scipy.ndimage.rotate(image, angle, mode='nearest', axes=(0, 1), reshape=False)

    # rotate along y-axis
    angle = random.uniform(-max_angle, max_angle)
    image3 = scipy.ndimage.rotate(image2, angle, mode='nearest', axes=(0, 2), reshape=False)

    # rotate along z-axis
    angle = random.uniform(-max_angle, max_angle)
    image_rot = scipy.ndimage.rotate(image3, angle, mode='nearest', axes=(1, 2), reshape=False)

    return image_rot

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, img, label, list_IDs_fg, list_IDs_bg, shuffle, train, classification, batch_size=32, dim=(32,32,32),
                 dim_img=(32,32,32), n_classes=10, n_channels=1, augment=(False,False,False,False,0), patch_normalization=False, separation=False):
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
            len_IDs = 2 * max(len(self.list_IDs_fg), len(self.list_IDs_bg))
        else:
            len_IDs = len(self.list_IDs_fg)
        n_batches = int(np.floor(len_IDs / self.batch_size))
        return n_batches

    def __getitem__(self, index):
        'Generate one batch of data'

        if len(self.list_IDs_bg) > 0:

            # len IDs
            len_IDs = max(len(self.list_IDs_fg), len(self.list_IDs_bg))

            # upsample lists of indexes to the same size
            repetitions = int(np.floor(len_IDs / len(self.list_IDs_fg))) + 1
            upsampled_indexes_fg = np.tile(self.indexes_fg, repetitions)
            upsampled_indexes_fg = upsampled_indexes_fg[:len_IDs]

            repetitions = int(np.floor(len_IDs / len(self.list_IDs_bg))) + 1
            upsampled_indexes_bg = np.tile(self.indexes_bg, repetitions)
            upsampled_indexes_bg = upsampled_indexes_bg[:len_IDs]

            # Generate indexes of the batch
            tmp_batch_size = int(self.batch_size / 2)
            indexes_fg = upsampled_indexes_fg[index*tmp_batch_size:(index+1)*tmp_batch_size]
            indexes_bg = upsampled_indexes_bg[index*tmp_batch_size:(index+1)*tmp_batch_size]

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
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        if self.classification:
            y = np.empty((self.batch_size, 1), dtype=np.int32)
        else:
            y = np.empty((self.batch_size, *self.dim, 1), dtype=np.int32)

        # get augmentation parameter
        flip_x, flip_y, flip_z, swapaxes, rotate = self.augment
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

            # get patch
            if self.classification:
                tmp_X = self.img[k:k+self.dim[0],l:l+self.dim[1],m:m+self.dim[2]]
                tmp_y = self.label[k,l,m]

                # augmentation
                if self.train:

                    # rotate in 3D
                    if rotate:
                        tmp_X = random_rotation_3d(tmp_X, max_angle=rotate)

                    # flip patch along axes
                    v = np.random.randint(n_aug+1)
                    if np.any([flip_x, flip_y, flip_z]) and v>0:
                        flip = flips[v-1]
                        tmp_X = np.flip(tmp_X, flip)

                    # swap axes
                    if swapaxes:
                        v = np.random.randint(4)
                        if v==1:
                            tmp_X = np.swapaxes(tmp_X,0,1)
                        elif v==2:
                            tmp_X = np.swapaxes(tmp_X,0,2)
                        elif v==3:
                            tmp_X = np.swapaxes(tmp_X,1,2)

                # assign to batch
                X[i,:,:,:,0] = tmp_X
                y[i,0] = tmp_y

            else:
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

                    # rotate in xy plane
                    if rotate:
                        tmp_X = np.empty((*self.dim, self.n_channels), dtype=np.float32)
                        tmp_y = np.empty(self.dim, dtype=np.int32)
                        cos_a = cos_angle[i]
                        sin_a = sin_angle[i]
                        for c in range(self.n_channels):
                            tmp_X[:,:,:,c] = rotate_img_patch(self.img[:,:,:,c],tmp_X[:,:,:,c],k,l,m,cos_a,sin_a,
                                self.dim[0],self.dim[1],self.dim[2],
                                self.dim_img[1],self.dim_img[2])
                        tmp_y = rotate_label_patch(self.label,tmp_y,k,l,m,cos_a,sin_a,
                            self.dim[0],self.dim[1],self.dim[2],
                            self.dim_img[1],self.dim_img[2])

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
                    tmp_X = tmp_X.copy()
                    for c in range(self.n_channels):
                        tmp_X[:,:,:,c] -= np.mean(tmp_X[:,:,:,c])
                        tmp_X[:,:,:,c] /= max(np.std(tmp_X[:,:,:,c]), 1e-6)

                # assign to batch
                X[i] = tmp_X
                y[i,:,:,:,0] = tmp_y

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

