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
from biomedisa_features.biomedisa_helper import load_data, save_data, unique_file_path
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
    argmin_x = max(argmin_x - 1, 0)
    argmax_x = min(argmax_x + 1, xsh-1) + 1
    argmin_y = max(argmin_y - 1, 0)
    argmax_y = min(argmax_y + 1, ysh-1) + 1
    argmin_z = max(argmin_z - 1, 0)
    argmax_z = min(argmax_z + 1, zsh-1) + 1
    data = np.copy(data[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x])
    return data, argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x

def clean(image, threshold=0.9):
    image_i = np.copy(image)
    allLabels = np.unique(image_i)
    mask = np.empty_like(image_i)
    s = [[[0,0,0], [0,1,0], [0,0,0]], [[0,1,0], [1,1,1], [0,1,0]], [[0,0,0], [0,1,0], [0,0,0]]]
    for k in allLabels[1:]:

        # get mask
        label = image_i==k
        mask.fill(0)
        mask[label] = 1

        # reduce size
        reduced, argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = reduce_blocksize(mask)

        # get clusters
        labeled_array, _ = ndimage.label(reduced, structure=s)
        size = np.bincount(labeled_array.ravel())

        # get reference size
        biggest_label = np.argmax(size[1:]) + 1
        label_size = size[biggest_label]

        # preserve large segments
        reduced.fill(0)
        for l, m in enumerate(size[1:]):
            if m > threshold * label_size:
                reduced[labeled_array==l+1] = 1

        # get original size
        mask.fill(0)
        mask[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x] = reduced

        # write cleaned label to array
        image_i[label] = 0
        image_i[mask==1] = k

    return image_i

def fill(image, threshold=0.9):
    image_i = np.copy(image)
    allLabels = np.unique(image_i)
    mask = np.empty_like(image_i)
    s = [[[0,0,0], [0,1,0], [0,0,0]], [[0,1,0], [1,1,1], [0,1,0]], [[0,0,0], [0,1,0], [0,0,0]]]
    for k in allLabels[1:]:

        # get mask
        label = image_i==k
        mask.fill(0)
        mask[label] = 1

        # reduce size
        reduced, argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = reduce_blocksize(mask)

        # reference size
        label_size = np.sum(reduced)

        # invert
        reduced = 1 - reduced # background and holes of object

        # get clusters
        labeled_array, _ = ndimage.label(reduced, structure=s)
        size = np.bincount(labeled_array.ravel())
        biggest_label = np.argmax(size)

        # get label with all holes filled
        reduced.fill(1)
        reduced[labeled_array==biggest_label] = 0

        # preserve large holes
        for l, m in enumerate(size[1:]):
            if m > threshold * label_size and l+1 != biggest_label:
                reduced[labeled_array==l+1] = 0

        # get original size
        mask.fill(0)
        mask[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x] = reduced

        # write filled label to array
        image_i[label] = 0
        image_i[mask==1] = k

    return image_i

def remove_outlier(image_id, final_id, friend_id, label_id, fill_holes=True):

    # get objects
    try:
        image = Upload.objects.get(pk=image_id)
        final = Upload.objects.get(pk=final_id)
        friend = Upload.objects.get(pk=friend_id)
        label = Upload.objects.get(pk=label_id)
        success = True
    except Upload.DoesNotExist:
        success = False

    # path to data
    path_to_data = image.pic.path
    path_to_final = final.pic.path

    if success:

        # final filenames
        filename, extension = os.path.splitext(path_to_final)
        if extension == '.gz':
            extension = '.nii.gz'
            filename = filename[:-4]
        path_to_cleaned = filename + '.cleaned' + extension
        path_to_filled = filename + '.filled' + extension
        path_to_cleaned_filled = filename + '.cleaned.filled' + extension

        # load data and header
        final, header = load_data(path_to_final, 'cleanup')
        if extension not in ['.tif','.am']:
            _, header = load_data(label.pic.path, 'cleanup')

        # remove outlier
        final_cleaned = clean(final, label.delete_outliers)

        try:
            # check if final still exists
            friend = Upload.objects.get(pk=friend_id)

            # save results
            path_to_cleaned = unique_file_path(path_to_cleaned, label.user.username)
            save_data(path_to_cleaned, final_cleaned, header, extension, label.compression)

            # save django object
            shortfilename = os.path.basename(path_to_cleaned)
            pic_path = 'images/' + friend.user.username + '/' + shortfilename
            if fill_holes:
                Upload.objects.create(pic=pic_path, user=friend.user, project=friend.project, final=2, imageType=3, shortfilename=shortfilename, friend=friend_id)
            else:
                Upload.objects.create(pic=pic_path, user=friend.user, project=friend.project, final=6, imageType=3, shortfilename=shortfilename, friend=friend_id)

            # create slices for sliceviewer
            if config['OS'] == 'linux':
                q_slices = Queue('slices', connection=Redis())
                job = q_slices.enqueue_call(create_slices, args=(path_to_data, path_to_cleaned,), timeout=-1)
            elif config['OS'] == 'windows':
                p = Process(target=create_slices, args=(path_to_data, path_to_cleaned))
                p.start()
                p.join()

        except Upload.DoesNotExist:
            success = False

        # fill holes
        if fill_holes and success:

            final_filled = fill(final, label.fill_holes)
            final_cleaned_filled = final_cleaned + (final_filled - final)

            try:
                # check if final still exists
                friend = Upload.objects.get(pk=friend_id)

                # save results
                path_to_filled = unique_file_path(path_to_filled, label.user.username)
                save_data(path_to_filled, final_filled, header, extension, label.compression)
                path_to_cleaned_filled = unique_file_path(path_to_cleaned_filled, label.user.username)
                save_data(path_to_cleaned_filled, final_cleaned_filled, header, extension, label.compression)

                # save django object
                shortfilename = os.path.basename(path_to_cleaned_filled)
                pic_path = 'images/' + friend.user.username + '/' + shortfilename
                Upload.objects.create(pic=pic_path, user=friend.user, project=friend.project, final=8, imageType=3, shortfilename=shortfilename, friend=friend_id)
                shortfilename = os.path.basename(path_to_filled)
                pic_path = 'images/' + friend.user.username + '/' + shortfilename
                Upload.objects.create(pic=pic_path, user=friend.user, project=friend.project, final=7, imageType=3, shortfilename=shortfilename, friend=friend_id)

                # create slices for sliceviewer
                if config['OS'] == 'linux':
                    q_slices = Queue('slices', connection=Redis())
                    job = q_slices.enqueue_call(create_slices, args=(path_to_data, path_to_filled,), timeout=-1)
                    job = q_slices.enqueue_call(create_slices, args=(path_to_data, path_to_cleaned_filled,), timeout=-1)
                elif config['OS'] == 'windows':
                    p = Process(target=create_slices, args=(path_to_data, path_to_filled))
                    p.start()
                    p.join()
                    p = Process(target=create_slices, args=(path_to_data, path_to_cleaned_filled))
                    p.start()
                    p.join()

            except Upload.DoesNotExist:
                pass

