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

from biomedisa.features.biomedisa_helper import welford_mean_std
import numpy as np
import keras
import numba
import random

def get_random_rotation_matrix(batch_size):
    angle_xy = np.random.uniform(0, 2. * np.pi, batch_size)
    Householder_1 = np.random.uniform(0, 2. * np.pi, batch_size)
    Householder_2 = np.random.uniform(0, 1., batch_size)
    # Matrix for xy rotation
    RR = np.zeros([batch_size, 3, 3])
    RR[:, 0, 0] = np.cos(angle_xy[:])
    RR[:, 0, 1] = np.sin(angle_xy[:])
    RR[:, 1, 0] = np.cos(angle_xy[:])
    RR[:, 1, 1] = -np.sin(angle_xy[:])
    RR[:, 2, 2] = 1.
    # Householder matrix
    vv = np.zeros([batch_size, 3, 1])
    vv[:, 0, 0] = np.cos(Householder_1[:]) * np.sqrt(Householder_2[:])
    vv[:, 1, 0] = np.sin(Householder_1[:]) * np.sqrt(Householder_2[:])
    vv[:, 2, 0] = np.sqrt(1. - Householder_2[:])
    HH = np.eye(3)[np.newaxis, :, :] - 2. * np.matmul(vv, vv.transpose(0, 2, 1))
    return -np.matmul(HH, RR)

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
def rotate_img_patch_3d(src,trg,k,l,m,rm_xx,rm_xy,rm_xz,rm_yx,rm_yy,rm_yz,rm_zx,rm_zy,rm_zz,z_patch,y_patch,x_patch,imageVertStride,imageDepth,imageHeight,imageWidth):
    #return rotate_label_patch_3d(src,trg,k,l,m,rm_xx,rm_xy,rm_xz,rm_yx,rm_yy,rm_yz,rm_zx,rm_zy,rm_zz,z_patch,y_patch,x_patch,imageVertStride,imageDepth,imageHeight,imageWidth)
    for z in range(k,k+z_patch):
        zCentreRotation = imageVertStride * (z // imageVertStride) + imageVertStride/2
        zA = z - zCentreRotation
        for y in range(l,l+y_patch):
            yA = y - imageHeight/2
            for x in range(m,m+x_patch):
                xA = x - imageWidth/2
                xR = xA * rm_xx + yA * rm_xy + zA * rm_xz
                yR = xA * rm_yx + yA * rm_yy + zA * rm_yz
                zR = xA * rm_zx + yA * rm_zy + zA * rm_zz
                src_x = xR + imageWidth/2
                src_y = yR + imageHeight/2
                src_z = zR + zCentreRotation
                # bilinear interpolation
                src_x0 = float(int(src_x))
                src_x1 = src_x0 + 1
                src_y0 = float(int(src_y))
                src_y1 = src_y0 + 1
                src_z0 = float(int(src_z))
                src_z1 = src_z0 + 1
                sx = src_x - src_x0
                sy = src_y - src_y0
                sz = src_z - src_z0
                idx_src_x0 = int(min(max(0,src_x0),imageWidth-1))
                idx_src_x1 = int(min(max(0,src_x1),imageWidth-1))
                idx_src_y0 = int(min(max(0,src_y0),imageHeight-1))
                idx_src_y1 = int(min(max(0,src_y1),imageHeight-1))
                idx_src_z0 = int(min(max(0,src_z0),imageDepth-1))
                idx_src_z1 = int(min(max(0,src_z1),imageDepth-1))
                
                val  = (1-sy) * (1-sx) * (1-sz) * float(src[idx_src_z0,idx_src_y0,idx_src_x0])
                val += (sy) * (1-sx) * (1-sz) * float(src[idx_src_z0,idx_src_y1,idx_src_x0])
                val += (1-sy) * (sx) * (1-sz) * float(src[idx_src_z0,idx_src_y0,idx_src_x1])
                val += (sy) * (sx) * (1-sz) * float(src[idx_src_z0,idx_src_y1,idx_src_x1])
                val += (1-sy) * (1-sx) * (sz) * float(src[idx_src_z1,idx_src_y0,idx_src_x0])
                val += (sy) * (1-sx) * (sz) * float(src[idx_src_z1,idx_src_y1,idx_src_x0])
                val += (1-sy) * (sx) * (sz) * float(src[idx_src_z1,idx_src_y0,idx_src_x1])
                val += (sy) * (sx) * (sz) * float(src[idx_src_z1,idx_src_y1,idx_src_x1])
                trg[z-k,y-l,x-m] = val
    return trg

# This exists so I could test it. It's not called because sometimes numba is funny about nested
# nopython functions.
@numba.jit(nopython=True)#parallel=True
def centre_of_z_rotation(z, imageVertStride):
    return imageVertStride * (z // imageVertStride) + imageVertStride/2

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

@numba.jit(nopython=True)#parallel=True
def rotate_label_patch_3d(src,trg,k,l,m,rm_xx,rm_xy,rm_xz,rm_yx,rm_yy,rm_yz,rm_zx,rm_zy,rm_zz,z_patch,y_patch,x_patch,imageVertStride,imageDepth,imageHeight,imageWidth):
    for z in range(k,k+z_patch):
        zCentreRotation = imageVertStride * (z // imageVertStride) + imageVertStride/2
        zA = z - zCentreRotation
        for y in range(l,l+y_patch):
            yA = y - imageHeight/2
            for x in range(m,m+x_patch):
                xA = x - imageWidth/2
                xR = xA * rm_xx + yA * rm_xy + zA * rm_xz
                yR = xA * rm_yx + yA * rm_yy + zA * rm_yz
                zR = xA * rm_zx + yA * rm_zy + zA * rm_zz
                src_x = xR + imageWidth/2
                src_y = yR + imageHeight/2
                src_z = zR + zCentreRotation
                # nearest neighbour
                src_x = round(src_x)
                src_y = round(src_y)
                src_z = round(src_z)
                idx_src_x = int(min(max(0,src_x),imageWidth-1))
                idx_src_y = int(min(max(0,src_y),imageHeight-1))
                idx_src_z = int(min(max(0,src_z),imageDepth-1))
                trg[z-k,y-l,x-m] = src[idx_src_z,idx_src_y,idx_src_x]
    return trg

class DataGenerator(keras.utils.PyDataset):
    'Generates data for Keras'
    def __init__(self, img, label, list_IDs_fg, list_IDs_bg, train, shuffle=True, batch_size=32, dim=(32,32,32),
                 dim_img=(32,32,32), n_classes=10, n_channels=1, augment=(False,False,False,False,0,False),
                 patch_normalization=False, separation=False, ignore_mask=False, downsample=False, **kwargs):
        super().__init__(**kwargs)
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
        self.patch_normalization = patch_normalization
        self.separation = separation
        self.ignore_mask = ignore_mask
        self.downsample = downsample
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        if len(self.list_IDs_bg) > 0:
            if self.downsample:
                len_IDs = min(len(self.list_IDs_fg), len(self.list_IDs_bg))
            else:
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
            fg_ids = self.indexes_fg[index*half_batch_size:(index+1)*half_batch_size]
            bg_ids = self.indexes_bg[index*half_batch_size:(index+1)*half_batch_size]

            # concatenate foreground and background IDs
            list_IDs_temp = np.concatenate((fg_ids, bg_ids))

        else:
            # batch of IDs
            list_IDs_temp = self.list_IDs_fg[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if len(self.list_IDs_bg) > 0:
            # downsample lists of indexes
            if self.downsample:
                len_IDs = min(len(self.list_IDs_fg), len(self.list_IDs_bg))
                self.indexes_fg = self.list_IDs_fg[:len_IDs]
                self.indexes_bg = self.list_IDs_bg[:len_IDs]
            else:
                len_IDs = max(len(self.list_IDs_fg), len(self.list_IDs_bg))
                repetitions = len_IDs // len(self.list_IDs_fg) + 1
                self.indexes_fg = np.tile(self.list_IDs_fg, repetitions)
                repetitions = len_IDs // len(self.list_IDs_bg) + 1
                self.indexes_bg = np.tile(self.list_IDs_bg, repetitions)

        # shuffle indexes
        if self.shuffle == True:
            np.random.shuffle(self.list_IDs_fg)
            if len(self.list_IDs_bg) > 0:
                np.random.shuffle(self.list_IDs_bg)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # number of label channels
        label_channels = 2 if self.ignore_mask else 1

        # allocate memory
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, *self.dim, label_channels), dtype=np.int32)

        # get augmentation parameter
        flip_x, flip_y, flip_z, swapaxes, rotate, rotate3d = self.augment
        n_aug = np.sum([flip_z, flip_y, flip_x])
        flips =  np.where([flip_z, flip_y, flip_x])[0]

        # create random angles
        if rotate:
            angle = np.random.uniform(-1,1,self.batch_size) * 3.1416/180*rotate
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
        if rotate3d:
            rot_mtx = get_random_rotation_matrix(self.batch_size)

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

                # rotate in xy plane
                if rotate:
                    tmp_X = np.empty((*self.dim, self.n_channels), dtype=np.float32)
                    tmp_y = np.empty((*self.dim, label_channels), dtype=np.int32)
                    cos_a = cos_angle[i]
                    sin_a = sin_angle[i]
                    for ch in range(self.n_channels):
                        tmp_X[...,ch] = rotate_img_patch(self.img[...,ch],tmp_X[...,ch],k,l,m,cos_a,sin_a,
                            self.dim[0],self.dim[1],self.dim[2],
                            self.dim_img[1],self.dim_img[2])
                    for ch in range(label_channels):
                        tmp_y[...,ch] = rotate_label_patch(self.label[...,ch],tmp_y[...,ch],k,l,m,cos_a,sin_a,
                            self.dim[0],self.dim[1],self.dim[2],
                            self.dim_img[1],self.dim_img[2])

                # rotate through a random 3d angle, uniformly distributed on a sphere.
                if rotate3d:
                    tmp_X = np.empty((*self.dim, self.n_channels), dtype=np.float32)
                    tmp_y = np.empty((*self.dim, label_channels), dtype=np.int32)
                    rm_xx = rot_mtx[i, 0, 0]
                    rm_xy = rot_mtx[i, 0, 1]
                    rm_xz = rot_mtx[i, 0, 2]
                    rm_yx = rot_mtx[i, 1, 0]
                    rm_yy = rot_mtx[i, 1, 1]
                    rm_yz = rot_mtx[i, 1, 2]
                    rm_zx = rot_mtx[i, 2, 0]
                    rm_zy = rot_mtx[i, 2, 1]
                    rm_zz = rot_mtx[i, 2, 2]
                    for ch in range(self.n_channels):
                        tmp_X[...,ch] = rotate_img_patch_3d(self.img[...,ch],tmp_X[...,ch],k,l,m,
                            rm_xx,rm_xy,rm_xz,rm_yx,rm_yy,rm_yz,rm_zx,rm_zy,rm_zz,
                            self.dim[0],self.dim[1],self.dim[2],
                            256, self.dim_img[0],self.dim_img[1],self.dim_img[2])
                    for ch in range(label_channels):
                        tmp_y[...,ch] = rotate_label_patch_3d(self.label[...,ch],tmp_y[...,ch],k,l,m,
                            rm_xx,rm_xy,rm_xz,rm_yx,rm_yy,rm_yz,rm_zx,rm_zy,rm_zz,
                            self.dim[0],self.dim[1],self.dim[2],
                            256, self.dim_img[0],self.dim_img[1],self.dim_img[2])

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
                for ch in range(self.n_channels):
                    mean, std = welford_mean_std(tmp_X[...,ch])
                    tmp_X[...,ch] -= mean
                    tmp_X[...,ch] /= max(std, 1e-6)

            # assign to batch
            X[i] = tmp_X
            y[i] = tmp_y

        if self.ignore_mask:
            return X, y
        else:
            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

