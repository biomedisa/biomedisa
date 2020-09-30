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
import biomedisa_app.views
from biomedisa_app.config import config
from biomedisa_app.models import Upload
from biomedisa_features.curvop_numba import curvop, evolution
from biomedisa_features.create_slices import create_slices
from biomedisa_features.biomedisa_helper import (load_data, save_data, pre_processing, 
                                id_generator, img_to_uint8)
from multiprocessing import Process

import numpy as np
import time

if config['OS'] == 'linux':
    from redis import Redis
    from rq import Queue

def reduce_blocksize(raw, slices):
    zsh, ysh, xsh = slices.shape
    argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = zsh, 0, ysh, 0, xsh, 0
    for k in range(zsh):
        y, x = np.nonzero(slices[k])
        if x.any():
            argmin_x = min(argmin_x, np.amin(x))
            argmax_x = max(argmax_x, np.amax(x))
            argmin_y = min(argmin_y, np.amin(y))
            argmax_y = max(argmax_y, np.amax(y))
            argmin_z = min(argmin_z, k)
            argmax_z = max(argmax_z, k)
    argmin_x = argmin_x - 100 if argmin_x - 100 > 0 else 0
    argmax_x = argmax_x + 100 if argmax_x + 100 < xsh else xsh
    argmin_y = argmin_y - 100 if argmin_y - 100 > 0 else 0
    argmax_y = argmax_y + 100 if argmax_y + 100 < ysh else ysh
    argmin_z = argmin_z - 100 if argmin_z - 100 > 0 else 0
    argmax_z = argmax_z + 100 if argmax_z + 100 < zsh else zsh
    raw = raw[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x]
    slices = slices[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x]
    return raw, slices, argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x

def activeContour(data, A, alpha=1.0, smooth=1, steps=3):
    print("alpha:", alpha, "smooth:", smooth, "steps:", steps)
    zsh, ysh, xsh = A.shape
    tmp = np.zeros((1, ysh, xsh), dtype=A.dtype)
    A = np.append(A, tmp, axis=0)
    A = np.append(tmp, A, axis=0)
    tmp = np.zeros((1, ysh, xsh), dtype=data.dtype)
    data = np.append(data, tmp, axis=0)
    data = np.append(tmp, data, axis=0)
    zsh, ysh, xsh = A.shape
    tmp = np.zeros((zsh, 1, xsh), dtype=A.dtype)
    A = np.append(A, tmp, axis=1)
    A = np.append(tmp, A, axis=1)
    tmp = np.zeros((zsh, 1, xsh), dtype=data.dtype)
    data = np.append(data, tmp, axis=1)
    data = np.append(tmp, data, axis=1)
    zsh, ysh, xsh = A.shape
    tmp = np.zeros((zsh, ysh, 1), dtype=A.dtype)
    A = np.append(A, tmp, axis=2)
    A = np.append(tmp, A, axis=2)
    tmp = np.zeros((zsh, ysh, 1), dtype=data.dtype)
    data = np.append(data, tmp, axis=2)
    data = np.append(tmp, data, axis=2)
    zsh, ysh, xsh = A.shape
    data, A, argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = reduce_blocksize(data, A)
    A = np.ascontiguousarray(A)
    data = np.ascontiguousarray(data)
    for n in range(steps):
        allLabels = np.unique(A)
        mean = np.zeros(256, dtype=np.float32)
        for k in allLabels:
            inside = A==k
            if np.any(inside):
                mean[k] = np.mean(data[inside])
        A = evolution(mean, A, data, alpha)
        for k in allLabels:
            A = curvop(A, smooth, k, allLabels)
    final = np.zeros((zsh, ysh, xsh), dtype=np.uint8)
    final[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x] = A
    return final[1:-1, 1:-1, 1:-1]

class Biomedisa(object):
     pass

def active_contour(path_to_data, path_to_labels, friend_id, label_id, image_id):

    # time
    TIC = time.time()

    # create biomedisa
    bm = Biomedisa()

    # get paths
    bm.path_to_data = path_to_data
    bm.path_to_labels = path_to_labels

    # path to logfiles
    bm.path_to_time = config['PATH_TO_BIOMEDISA'] + '/log/time.txt'
    bm.path_to_logfile = config['PATH_TO_BIOMEDISA'] + '/log/logfile.txt'

    # get objects
    bm.success = True
    try:
        bm.image = Upload.objects.get(pk=image_id)
        bm.label = Upload.objects.get(pk=label_id)
        friend = Upload.objects.get(pk=friend_id)
    except Upload.DoesNotExist:
        bm.success = False
        if label_id == friend_id:
            bm.image.status = 0
            bm.image.pid = 0
            bm.image.save()
            message = 'Files have been removed.'
            Upload.objects.create(user=bm.image.user, project=bm.image.project, log=1, imageType=None, shortfilename=message)

    # set PID
    if label_id == friend_id and bm.success:
        if bm.image.status == 1:
            bm.image.status = 2
            bm.image.message = 'Processing'
        bm.image.pid = int(os.getpid())
        bm.image.save()

    # pre-processing
    if bm.success:
        bm.process = 'acwe'
        bm = pre_processing(bm)

    if bm.success:

        # active contour as stand alone
        if label_id == friend_id:

            # write in logfile and send start notification
            biomedisa_app.views.send_start_notification(bm.image)

            # write in logfile
            with open(bm.path_to_logfile, 'a') as logfile:
                print('%s %s %s %s' %(time.ctime(), bm.image.user, bm.image.shortfilename, 'Process was started.'), file=logfile)

            # create path_to_final
            filename, extension = os.path.splitext(bm.image.shortfilename)
            if extension == '.gz':
                filename = filename[:-4]
            filename = 'final.' + filename
            dir_path = config['PATH_TO_BIOMEDISA'] + '/private_storage/'
            pic_path = 'images/%s/%s' %(bm.image.user, filename)
            limit = 100 - len(bm.final_image_type) - len('.acwe')
            bm.path_to_acwe = dir_path + pic_path[:limit] + '.acwe' + bm.final_image_type

            # if path_to_final exists create new path_to_final
            if os.path.exists(bm.path_to_acwe):
                CHARACTERS, CODE_SIZE = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz23456789', 7
                newending = id_generator(CODE_SIZE, CHARACTERS)
                limit = 100 - len(bm.final_image_type) - len('.acwe') - 8
                bm.path_to_acwe = dir_path + pic_path[:limit] + '_' + newending + '.acwe' + bm.final_image_type

        else:
            # final filename
            filename, extension = os.path.splitext(bm.path_to_labels)
            if extension == '.gz':
                filename = filename[:-4]
            bm.path_to_acwe = filename + '.acwe' + bm.final_image_type

        # data type
        bm.data = img_to_uint8(bm.data)

        # get acwe parameters
        alpha = bm.label.ac_alpha
        smooth = bm.label.ac_smooth
        steps = bm.label.ac_steps

        # run acwe
        final = activeContour(bm.data, bm.labelData, alpha, smooth, steps)

        try:
            # check if final has already been removed
            friend = Upload.objects.get(pk=friend_id)

            # image size
            bm.imageSize = int(final.nbytes * 10e-7)

            # save result
            save_data(bm.path_to_acwe, final, bm.header, bm.final_image_type, bm.label.compression)

            # create slices
            if config['OS'] == 'linux':
                q = Queue('slices', connection=Redis())
                job = q.enqueue_call(create_slices, args=(bm.path_to_data, bm.path_to_acwe,), timeout=-1)
            elif config['OS'] == 'windows':
                Process(target=create_slices, args=(bm.path_to_data, bm.path_to_acwe)).start()

            # save django object
            shortfilename = os.path.basename(bm.path_to_acwe)
            filename = 'images/' + bm.label.user.username + '/' + shortfilename

            if label_id != friend_id:
                tmp = Upload.objects.create(pic=filename, user=bm.label.user, project=friend.project, final=3, imageType=3, shortfilename=shortfilename, friend=friend_id)

            elif label_id == friend_id:
                tmp = Upload.objects.create(pic=filename, user=bm.label.user, project=bm.label.project, final=3, imageType=3, shortfilename=shortfilename, active=1)
                tmp.friend = tmp.id
                tmp.save()
                bm.image.status = 0
                bm.image.pid = 0
                bm.image.save()

                # write in logs
                t = int(time.time() - TIC)
                if t < 60:
                    time_str = str(t) + ' sec'
                elif 60 <= t < 3600:
                    time_str = str(t // 60) + ' min ' + str(t % 60) + ' sec'
                elif 3600 < t:
                    time_str = str(t // 3600) + ' h ' + str((t % 3600) // 60) + ' min ' + str(t % 60) + ' sec'
                with open(bm.path_to_time, 'a') as timefile:
                    print('%s %s %s %s MB %s on %s active_contour' %(time.ctime(), bm.image.user.username, \
                    bm.image.shortfilename, bm.imageSize, time_str, config['SERVER_ALIAS']), file=timefile)
                print('Total calculation time:', time_str)

                # send notification
                biomedisa_app.views.send_notification(bm.image.user.username, bm.image.shortfilename, time_str, config['SERVER_ALIAS'])

        except Upload.DoesNotExist:
            pass
