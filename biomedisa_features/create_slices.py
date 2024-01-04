#!/usr/bin/python3
##########################################################################
##                                                                      ##
##  Copyright (c) 2024 Philipp Lösel. All rights reserved.              ##
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

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from biomedisa_features.biomedisa_helper import load_data, color_to_gray, img_to_uint8, img_resize
from PIL import Image
import numpy as np
from glob import glob
import numba
import shutil
import cv2

def unique(arr):
    arr = arr.astype(np.uint8)
    counts = np.zeros(256, dtype=int)
    zsh, ysh, xsh = arr.shape
    @numba.jit(nopython=True)
    def __unique__(arr, zsh, ysh, xsh, counts):
        for k in range(zsh):
            for l in range(ysh):
                for m in range(xsh):
                    index = arr[k,l,m]
                    counts[index] += 1
        return counts
    counts = __unique__(arr, zsh, ysh, xsh, counts)
    labels = np.where(counts)[0]
    counts = counts[labels]
    return labels, counts

def percentile(a, limit):
    values, counts = unique(a)
    total_positive = np.sum(counts[1:])
    limit = total_positive / 100.0 * limit
    k, tmp = 1, 0
    while tmp < limit:
        tmp += counts[k]
        k += 1
    low = values[k-1]
    limit = total_positive - limit
    k, tmp = -1, total_positive
    while tmp > limit:
        tmp -= counts[k]
        k -= 1
    high = values[k+1]
    return low, high

def contrast(arr):
    bottom_quantile, top_quantile = percentile(arr, 0.5)
    arr[arr<bottom_quantile] = bottom_quantile
    arr[arr>top_quantile] = top_quantile
    values, counts = unique(arr)
    mean = np.sum(values * counts) / np.sum(counts)
    std = np.sqrt(np.sum(counts * (values - mean)**2) / (np.sum(counts) - 1))
    arr = arr.astype(np.float16)
    arr = (arr - mean) / std
    amin = (bottom_quantile - mean) / std
    arr -= amin
    arr /= (top_quantile - mean) / std - amin
    arr *= 255.0
    arr = arr.astype(np.uint8)
    return arr

def create_slices(path_to_data, path_to_label, on_site=False):

    try:
        if on_site:
            path_to_slices = os.path.dirname(path_to_data) + '/' + os.path.splitext(os.path.basename(path_to_data))[0]
            if os.path.isdir(path_to_slices):
                shutil.rmtree(path_to_slices)
            if path_to_label:
                path_to_label_slices = os.path.dirname(path_to_label) + '/' + os.path.splitext(os.path.basename(path_to_label))[0]
                if os.path.isdir(path_to_label_slices):
                    shutil.rmtree(path_to_label_slices)

        else:
            path_to_slices = path_to_data.replace('images', 'sliceviewer', 1)
            if path_to_label:
                path_to_label_slices = path_to_label.replace('images', 'sliceviewer', 1)

        if not os.path.isdir(path_to_slices) or (path_to_label and not os.path.isdir(path_to_label_slices)):

            # load data and reduce data size
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
                scale = float(256) / float(max(zsh, ysh, xsh))
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
                zsh, ysh, xsh = raw.shape
                if min(ysh, xsh) > 400 and not on_site:
                    scale = float(400) / float(min(ysh, xsh))
                    ysh = int(ysh * scale)
                    xsh = int(xsh * scale)
                    tmp = np.empty((zsh, ysh, xsh), dtype=raw.dtype)
                    for k in range(zsh):
                        tmp[k] = cv2.resize(raw[k], (xsh, ysh), interpolation=cv2.INTER_AREA)
                    raw = tmp

            # increase contrast
            raw = img_to_uint8(raw)
            raw = contrast(raw)

            # create slices for slice viewer
            if not os.path.isdir(path_to_slices):

                # make directory
                os.makedirs(path_to_slices)
                os.chmod(path_to_slices, 0o770)

                # save slices
                for k in range(zsh):
                    im = Image.fromarray(raw[k])
                    im.save(path_to_slices + f'/{k}.png')

            if path_to_label and not os.path.isdir(path_to_label_slices):

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
                    # load and scale label data to corresponding img data
                    mask = np.zeros((0, y_scale, x_scale), dtype=np.uint8)
                    for name in img_names:
                        arr, _ = load_data(name, 'create_slices')
                        arr = color_to_gray(arr)
                        arr = arr.astype(np.uint8)
                        arr = img_resize(arr, z_scale, y_scale, x_scale, labels=True)
                        mask = np.append(mask, arr, axis=0)
                else:
                    mask, _ = load_data(path_to_label, 'create_slices')
                    mask = color_to_gray(mask)
                    mask = mask.astype(np.uint8)
                    zsh, ysh, xsh = mask.shape
                    if min(ysh, xsh) > 400 and not on_site:
                        scale = float(400) / float(min(ysh, xsh))
                        ysh = int(ysh * scale)
                        xsh = int(xsh * scale)
                        tmp = np.zeros((zsh, ysh, xsh), dtype=mask.dtype)
                        for k in range(zsh):
                            if np.any(mask[k]):
                                tmp[k] = cv2.resize(mask[k], (xsh, ysh), interpolation=cv2.INTER_NEAREST)
                        mask = tmp

                # create slices for slice viewer
                if mask.shape==raw.shape:

                    # make directory
                    os.makedirs(path_to_label_slices)
                    os.chmod(path_to_label_slices, 0o770)

                    # define colors
                    Color = [(255,0,0),(255,255,0),(0,0,255),(0,100,0),(0,255,0),(255,165,0),(139,0,0),(255,20,147),(255,105,180),(255,0,0),(139,0,139),(255,0,255),(160,32,240),(184,134,11),(255,185,15),(255,215,0),(0,191,255),(16,78,139),(104,131,139),(255,64,64),(165,42,42),(255,127,36),(139,90,43),(110,139,61),(0,255,127),(255,127,80),(139,10,80),(219,112,147),(178,34,34),(255,48,48),(205,79,57),(160,32,240),(255,100,0)] * 8
                    labels, _ = unique(mask)
                    labels = labels[1:]
                    Color = Color[:len(labels)]
                    Color = np.array(Color, dtype=np.uint8)

                    # allocate memory
                    zsh, ysh, xsh = raw.shape
                    out = np.empty((ysh, xsh, 3), dtype=np.uint8)
                    gradient = np.empty((ysh, xsh), dtype=np.uint8)

                    # save preview slice by slice
                    for k in range(zsh):
                        raw_tmp = raw[k]
                        mask_tmp = mask[k]

                        # create output slice
                        for l in range(3):
                            out[:,:,l] = raw_tmp

                        if np.any(mask_tmp):

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

                            # colorize
                            for j, label in enumerate(labels):
                                C = Color[j]
                                tmp = np.logical_and(gradient>0, mask_tmp==label)
                                out[:,:][tmp] = C

                        # save slice
                        im = Image.fromarray(out)
                        im.save(path_to_label_slices + f'/{k}.png')

    except Exception as e:
        print(e)

if __name__ == "__main__":
    if len(sys.argv)==2:
        path_to_labels = None
    else:
        path_to_labels = sys.argv[2]
    create_slices(sys.argv[1], path_to_labels, True)

