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

from biomedisa_features.biomedisa_helper import load_data, color_to_gray, img_to_uint8, img_resize
from PIL import Image
import numpy as np
import os, sys
import cv2
from glob import glob

def percentile(a, limit):
    # this implementation is slower than numpy's but uses less memory
    nop = float(a.shape[0] * a.shape[1] * a.shape[2])
    unique, noc = np.unique(a[a>0], return_counts=True)
    limit = nop / 100.0 * limit
    k, tmp = 0, 0
    while tmp < limit:
        tmp += noc[k]
        k += 1
    low = unique[k-1]
    limit = nop - limit
    k, tmp = -1, nop
    while tmp > limit:
        tmp -= noc[k]
        k -= 1
    high = unique[k+1]
    return low, high

def contrast(a):
    a = np.copy(a)
    try:
        bottom_quantile = np.percentile(a[a>0], 0.5)
        top_quantile = np.percentile(a[a>0], 99.5)
    except MemoryError:
        bottom_quantile, top_quantile = percentile(a, 0.5)
    a[a<bottom_quantile] = bottom_quantile
    a[a>top_quantile] = top_quantile
    a = a.astype(np.float32)
    mu, std = np.mean(a), np.std(a)
    a = (a - mu) / std
    a -= np.amin(a)
    a /= np.amax(a)
    a *= 255.0
    a = a.astype(np.uint8)
    return a

def create_slices(path_to_data, path_to_label, on_site=False):

    try:

        if on_site:
            if path_to_data:
                path_to_slices = os.path.dirname(path_to_data) + '/' + os.path.splitext(os.path.basename(path_to_data))[0]
            if path_to_label:
                path_to_label_slices = os.path.dirname(path_to_label) + '/' + os.path.splitext(os.path.basename(path_to_label))[0]
        else:
            if path_to_data:
                path_to_slices = path_to_data.replace('images', 'sliceviewer', 1)
            if path_to_label:
                path_to_label_slices = path_to_label.replace('images', 'sliceviewer', 1)

        if path_to_data:

            # load data
            path_to_dir, extension = os.path.splitext(path_to_data)
            if extension == '.gz':
                path_to_dir, extension = os.path.splitext(path_to_dir)
            if extension == '.tar':
                img_names = []
                for data_type in ['.tif','.tiff','.am','.hdr','.mhd','.mha','.nrrd','.nii','.nii.gz']:
                    tmp_img_names = glob(path_to_dir+'/**/*'+data_type, recursive=True)
                    tmp_img_names = sorted(tmp_img_names)
                    img_names.extend(tmp_img_names)
                raw, _ = load_data(img_names[0], 'create_slices')
                zsh, ysh, xsh = raw.shape
                m = max(ysh, xsh)
                m = max(zsh, m)
                scale = float(256) / float(m)
                z_scale = int(zsh * scale)
                y_scale = int(ysh * scale)
                x_scale = int(xsh * scale)
                raw = img_resize(raw, z_scale, y_scale, x_scale)
                for name in img_names[1:]:
                    tmp, _ = load_data(name, 'create_slices')
                    tmp = img_resize(tmp, z_scale, y_scale, x_scale)
                    raw = np.append(raw, tmp, axis=0)
            else:
                raw, _ = load_data(path_to_data, 'create_slices')

            # increase contrast
            raw = img_to_uint8(raw)
            raw = contrast(raw)
            zsh, ysh, xsh = raw.shape

            # create slices for slice viewer
            if not path_to_label:

                # make directory
                if not os.path.isdir(path_to_slices):
                    os.makedirs(path_to_slices)
                    os.chmod(path_to_slices, 0o770)

                # clean directory
                if any(glob(path_to_slices + "/*")):
                    os.system(f"rm {path_to_slices}" + "/*")

                # reduce image size
                m = min(ysh, xsh)
                if m > 400:
                    scale = float(400) / float(m)
                    y_shape = int(ysh * scale)
                    x_shape = int(xsh * scale)
                    for k in range(zsh):
                        tmp = cv2.resize(raw[k], (x_shape, y_shape), interpolation=cv2.INTER_AREA)
                        tmp = tmp.astype(np.uint8)
                        im = Image.fromarray(tmp)
                        im.save(path_to_slices + '/%s.png' %(k))
                else:
                    for k in range(zsh):
                        im = Image.fromarray(raw[k])
                        im.save(path_to_slices + '/%s.png' %(k))

        if path_to_label:

            # load data
            path_to_dir, extension = os.path.splitext(path_to_label)
            if extension == '.gz':
                path_to_dir, extension = os.path.splitext(path_to_dir)
            if extension == '.tar':
                img_names = []
                for data_type in ['.tif','.tiff','.am','.hdr','.mhd','.mha','.nrrd','.nii','.nii.gz']:
                    tmp_img_names = glob(path_to_dir+'/**/*'+data_type, recursive=True)
                    tmp_img_names = sorted(tmp_img_names)
                    img_names.extend(tmp_img_names)
                # load and scale label data corresponding to img data
                mask = np.zeros((0, y_scale, x_scale), dtype=np.uint8)
                for name in img_names:
                    arr, _ = load_data(name, 'create_slices')
                    arr = arr.astype(np.uint8)
                    np_unique = np.unique(arr)[1:]
                    next_mask = np.zeros((z_scale, y_scale, x_scale), dtype=arr.dtype)
                    for k in np_unique:
                        tmp = np.zeros_like(arr)
                        tmp[arr==k] = 1
                        tmp = img_resize(tmp, z_scale, y_scale, x_scale)
                        next_mask[tmp==1] = k
                    mask = np.append(mask, next_mask, axis=0)
            else:
                mask, _ = load_data(path_to_label, 'create_slices')

            # img to uint8
            mask = color_to_gray(mask)
            mask = img_to_uint8(mask)

            # create slices for slice viewer
            if path_to_data and mask.shape == raw.shape:

                # make directory
                if not os.path.isdir(path_to_label_slices):
                    os.makedirs(path_to_label_slices)
                    os.chmod(path_to_label_slices, 0o770)

                # clean directory
                if any(glob(path_to_label_slices + "/*")):
                    os.system(f"rm {path_to_label_slices}" + "/*")

                # define colors
                Color = [(255,0,0),(255,255,0),(0,0,255),(0,100,0),(0,255,0),(255,165,0),(139,0,0),(255,20,147),(255,105,180),(255,0,0),(139,0,139),(255,0,255),(160,32,240),(184,134,11),(255,185,15),(255,215,0),(0,191,255),(16,78,139),(104,131,139),(255,64,64),(165,42,42),(255,127,36),(139,90,43),(110,139,61),(0,255,127),(255,127,80),(139,10,80),(219,112,147),(178,34,34),(255,48,48),(205,79,57),(160,32,240),(255,100,0)] * 8
                labels = np.unique(mask)[1:]
                Color = Color[:len(labels)]
                labels = labels[:len(labels)]
                Color = np.array(Color, dtype=np.uint8)

                # reduce image size
                m = min(ysh, xsh)
                if m > 400:
                    scale = float(400) / float(m)
                    ysh = int(ysh * scale)
                    xsh = int(xsh * scale)

                # allocate memory
                out = np.empty((ysh, xsh, 3), dtype=np.uint8)
                gradient = np.empty((ysh, xsh), dtype=np.uint8)

                for k in range(zsh):

                    # resize slice
                    if m > 400:
                        raw_tmp = cv2.resize(raw[k], (xsh, ysh), interpolation=cv2.INTER_AREA)
                        mask_tmp = np.zeros((ysh, xsh), dtype=mask.dtype)
                        for l in labels:
                            tmp = np.zeros_like(mask[k])
                            tmp[mask[k]==l] = 1
                            tmp = cv2.resize(tmp, (xsh, ysh), interpolation=cv2.INTER_AREA)
                            mask_tmp[tmp==1] = l
                        raw_tmp = raw_tmp.astype(np.uint8)
                        mask_tmp = mask_tmp.astype(np.uint8)
                    else:
                        raw_tmp = raw[k]
                        mask_tmp = mask[k]

                    # compute gradient
                    gradient.fill(0)
                    tmp = np.abs(mask_tmp[:-1] - mask_tmp[1:])
                    tmp[tmp>0] = 1
                    gradient[:-1] += tmp
                    gradient[1:] += tmp
                    tmp = np.abs(mask_tmp[:,:-1] - mask_tmp[:,1:])
                    tmp[tmp>0] = 1
                    gradient[:,:-1] += tmp
                    gradient[:,1:] += tmp

                    # create output slice
                    for l in range(3):
                        out[:,:,l] = raw_tmp

                    # colorize
                    for j, label in enumerate(labels):
                        C = Color[j]
                        tmp = np.logical_and(gradient>0, mask_tmp==label)
                        out[:,:][tmp] = C

                    # save slice
                    im = Image.fromarray(out)
                    im.save(path_to_label_slices + '/%s.png' %(k))

    except Exception as e:
        print(e)

