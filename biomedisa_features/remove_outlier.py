##########################################################################
##                                                                      ##
##  Copyright (c) 2020 Philipp Lösel. All rights reserved.              ##
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
import django
django.setup()
from django.shortcuts import get_object_or_404
from biomedisa_app.config import config
from biomedisa_app.models import Upload
from biomedisa_features.create_slices import create_slices
from biomedisa_features.biomedisa_helper import load_data, save_data
from multiprocessing import Process

import numpy as np
from scipy import ndimage

if config['OS'] == 'linux':
    from redis import Redis
    from rq import Queue

def reduce_blocksize(data):
    zsh, ysh, xsh = data.shape
    argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = zsh, 0, ysh, 0, xsh, 0
    for k in range(zsh):
        y, x = np.nonzero(data[k])
        if x.any():
            argmin_x = min(argmin_x, np.amin(x))
            argmax_x = max(argmax_x, np.amax(x))
            argmin_y = min(argmin_y, np.amin(y))
            argmax_y = max(argmax_y, np.amax(y))
            argmin_z = min(argmin_z, k)
            argmax_z = max(argmax_z, k)
    argmin_x = argmin_x - 10 if argmin_x - 10 > 0 else 0
    argmax_x = argmax_x + 10 if argmax_x + 10 < xsh else xsh
    argmin_y = argmin_y - 10 if argmin_y - 10 > 0 else 0
    argmax_y = argmax_y + 10 if argmax_y + 10 < ysh else ysh
    argmin_z = argmin_z - 10 if argmin_z - 10 > 0 else 0
    argmax_z = argmax_z + 10 if argmax_z + 10 < zsh else zsh
    data = data[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x]
    return data, argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x

def clean(image_i, threshold):
    image = np.copy(image_i)
    allLabels = np.unique(image)[1:]
    tmp = np.empty_like(image)
    s = [[[0,0,0], [0,1,0], [0,0,0]], [[0,1,0], [1,1,1], [0,1,0]], [[0,0,0], [0,1,0], [0,0,0]]]
    for k in allLabels:
        label = image==k
        #label_size = np.sum(label)
        tmp.fill(0)
        tmp[label] = 1
        labeled_array, _ = ndimage.label(tmp, structure=s)
        size = np.bincount(labeled_array.ravel())
        biggest_label = np.argmax(size[1:]) + 1
        label_size = size[biggest_label]
        image[label] = 0
        for l, m in enumerate(size[1:]):
            if m > threshold * label_size:
                image[labeled_array==l+1] = k
    return image

def fill(image_i, threshold):
    image = np.copy(image_i)
    allLabels = np.unique(image)[1:]
    tmp = np.empty_like(image)
    foreground = np.empty_like(tmp)
    s = [[[0,0,0], [0,1,0], [0,0,0]], [[0,1,0], [1,1,1], [0,1,0]], [[0,0,0], [0,1,0], [0,0,0]]]
    for k in allLabels:
        label = image==k
        tmp.fill(1)
        tmp[label] = 0    # with holes
        label_size = np.sum(label)
        labeled_array, _ = ndimage.label(tmp, structure=s)
        size = np.bincount(labeled_array.ravel())
        biggest_label = np.argmax(size)
        foreground.fill(1)
        foreground[labeled_array==biggest_label] = 0    # without holes
        holes = np.copy(foreground - 1 + tmp)           # holes only == 1
        labeled_array, _ = ndimage.label(holes, structure=s)
        size = np.bincount(labeled_array.ravel())
        image_copy = np.copy(image)
        image[foreground==1] = k
        for l, m in enumerate(size[1:]):
            if m > threshold * label_size:
                image[labeled_array==l+1] = image_copy[labeled_array==l+1]
    return image

def remove_outlier(path_to_data, path_to_final, friend_id, labelObject_id, fill_holes=True):

    # get label object
    labelObject = get_object_or_404(Upload, pk=labelObject_id)

    # final filenames
    filename, extension = os.path.splitext(path_to_final)
    if extension == '.gz':
        extension = '.nii.gz'
        filename = filename[:-4]
    path_to_cleaned = filename + '.cleaned' + extension
    path_to_filled = filename + '.filled' + extension
    path_to_cleaned_filled = filename + '.cleaned.filled' + extension

    # load data
    final, header = load_data(path_to_final, 'cleanup')
    if extension not in ['.tif','.am']:
        _, header = load_data(labelObject.pic.path, 'cleanup')

    # reduce block size
    zsh, ysh, xsh = final.shape
    final, argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = reduce_blocksize(final)

    # remove outlier and fill holes
    final_cleaned_tmp = clean(final, labelObject.delete_outliers)
    if fill_holes:
        final_filled_tmp = fill(final, labelObject.fill_holes)
        final_cleaned_filled_tmp = fill(final_cleaned_tmp, labelObject.fill_holes)

    # retrieve full size
    final_cleaned = np.zeros((zsh, ysh, xsh), dtype=final.dtype)
    final_cleaned[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x] = final_cleaned_tmp
    if fill_holes:
        final_filled = np.zeros((zsh, ysh, xsh), dtype=final.dtype)
        final_filled[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x] = final_filled_tmp
        final_cleaned_filled = np.zeros((zsh, ysh, xsh), dtype=final.dtype)
        final_cleaned_filled[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x] = final_cleaned_filled_tmp

    # save results
    save_data(path_to_cleaned, final_cleaned, header, extension, labelObject.compression)
    if fill_holes:
        save_data(path_to_filled, final_filled, header, extension, labelObject.compression)
        save_data(path_to_cleaned_filled, final_cleaned_filled, header, extension, labelObject.compression)

    # create slices for sliceviewer
    if config['OS'] == 'linux':
        q_slices = Queue('slices', connection=Redis())
        job = q_slices.enqueue_call(create_slices, args=(path_to_data, path_to_cleaned,), timeout=-1)
        if fill_holes:
            job = q_slices.enqueue_call(create_slices, args=(path_to_data, path_to_filled,), timeout=-1)
            job = q_slices.enqueue_call(create_slices, args=(path_to_data, path_to_cleaned_filled,), timeout=-1)
    elif config['OS'] == 'windows':
        p = Process(target=create_slices, args=(path_to_data, path_to_cleaned))
        p.start()
        p.join()
        if fill_holes:
            p = Process(target=create_slices, args=(path_to_data, path_to_filled))
            p.start()
            p.join()
            p = Process(target=create_slices, args=(path_to_data, path_to_cleaned_filled))
            p.start()
            p.join()

    # create django objects
    tmp = get_object_or_404(Upload, pk=friend_id)
    shortfilename = os.path.basename(path_to_cleaned)
    filename = 'images/' + str(tmp.user) + '/' + shortfilename
    if not fill_holes:
        Upload.objects.create(pic=filename, user=tmp.user, project=tmp.project, final=6, imageType=3, shortfilename=shortfilename, friend=friend_id)
    else:
        Upload.objects.create(pic=filename, user=tmp.user, project=tmp.project, final=2, imageType=3, shortfilename=shortfilename, friend=friend_id)
        shortfilename = os.path.basename(path_to_cleaned_filled)
        filename = 'images/' + str(tmp.user) + '/' + shortfilename
        Upload.objects.create(pic=filename, user=tmp.user, project=tmp.project, final=8, imageType=3, shortfilename=shortfilename, friend=friend_id)
        shortfilename = os.path.basename(path_to_filled)
        filename = 'images/' + str(tmp.user) + '/' + shortfilename
        Upload.objects.create(pic=filename, user=tmp.user, project=tmp.project, final=7, imageType=3, shortfilename=shortfilename, friend=friend_id)
