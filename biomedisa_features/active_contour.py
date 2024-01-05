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

import os
from biomedisa_features.curvop_numba import curvop, evolution
import numpy as np
import numba

@numba.jit(nopython=True)
def geodis(c,sqrt2,sqrt3,iterations):
    zsh, ysh, xsh = c.shape
    for i in range(iterations):
        for z in range(1,zsh):
            for y in range(1,ysh-1):
                for x in range(1,xsh-1):
                    a1 = c[z-1,y-1,x-1] + sqrt3
                    a2 = c[z-1,y-1,x] + sqrt2
                    a3 = c[z-1,y-1,x+1] + sqrt3
                    a4 = c[z-1,y,x-1] + sqrt2
                    a5 = c[z-1,y,x] + 1
                    a6 = c[z-1,y,x+1] + sqrt2
                    a7 = c[z-1,y+1,x-1] + sqrt3
                    a8 = c[z-1,y+1,x] + sqrt2
                    a9 = c[z-1,y+1,x+1] + sqrt3
                    a10 = c[z,y-1,x-1] + sqrt2
                    a11 = c[z,y-1,x] + 1
                    a12 = c[z,y-1,x+1] + sqrt2
                    a13 = c[z,y,x-1] + 1
                    a14 = c[z,y,x]
                    c[z,y,x] = min(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14)
        for z in range(zsh-2,-1,-1):
            for y in range(ysh-2,0,-1):
                for x in range(xsh-2,0,-1):
                    a1 = c[z+1,y-1,x-1] + sqrt3
                    a2 = c[z+1,y-1,x] + sqrt2
                    a3 = c[z+1,y-1,x+1] + sqrt3
                    a4 = c[z+1,y,x-1] + sqrt2
                    a5 = c[z+1,y,x] + 1
                    a6 = c[z+1,y,x+1] + sqrt2
                    a7 = c[z+1,y+1,x-1] + sqrt3
                    a8 = c[z+1,y+1,x] + sqrt2
                    a9 = c[z+1,y+1,x+1] + sqrt3
                    a10 = c[z,y+1,x+1] + sqrt2
                    a11 = c[z,y+1,x] + 1
                    a12 = c[z,y+1,x-1] + sqrt2
                    a13 = c[z,y,x+1] + 1
                    a14 = c[z,y,x]
                    c[z,y,x] = min(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14)
    return c

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

def activeContour(data, labelData, alpha=1.0, smooth=1, steps=3, allLabels=None):
    print("alpha:", alpha, "smooth:", smooth, "steps:", steps)
    zsh, ysh, xsh = data.shape
    tmp = np.zeros((2+zsh, 2+ysh, 2+xsh), dtype=data.dtype)
    tmp[1:-1,1:-1,1:-1] = data
    data = np.copy(tmp)
    tmp = np.zeros((2+zsh, 2+ysh, 2+xsh), dtype=labelData.dtype)
    tmp[1:-1,1:-1,1:-1] = labelData
    labelData = np.copy(tmp)
    zsh, ysh, xsh = data.shape
    data, labelData, argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = reduce_blocksize(data, labelData)
    labelData = np.copy(labelData)
    data = np.copy(data)
    if allLabels is None:
        allLabels = np.unique(labelData)
    for n in range(steps):
        mean = np.zeros(256, dtype=np.float32)
        for k in allLabels:
            inside = labelData==k
            if np.any(inside):
                mean[k] = np.mean(data[inside])
        labelData = evolution(mean, labelData, data, alpha)
        for k in allLabels:
            labelData = curvop(labelData, smooth, k, allLabels)
    final = np.zeros((zsh, ysh, xsh), dtype=np.uint8)
    final[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x] = labelData
    final = np.copy(final[1:-1, 1:-1, 1:-1])
    return final

def refinement(data, labelData, allLabels=None):
    if allLabels is None:
        allLabels = np.unique(labelData)
    zsh, ysh, xsh = data.shape
    distance = np.zeros(data.shape, dtype=np.float32)
    distance[labelData==0] = 100
    distance = geodis(distance,np.sqrt(2),np.sqrt(3),1)
    distance[distance<=10] = 0
    distance[distance>10] = 1
    distance = distance.astype(np.uint8)
    result = np.zeros_like(labelData)
    for l in allLabels[1:]:
        for k in range(zsh):
            img = data[k]
            mask = labelData[k]
            d = distance[k]
            if np.any(np.logical_and(d==0,mask==l)) and np.any(np.logical_and(d==0,mask!=l)):
                m1,s1 = np.mean(img[np.logical_and(d==0,mask==l)]), np.std(img[np.logical_and(d==0,mask==l)])
                m2,s2 = np.mean(img[np.logical_and(d==0,mask!=l)]), np.std(img[np.logical_and(d==0,mask!=l)])
                s1 = max(s1,1)
                s2 = max(s2,1)
                p1 = np.exp(-(img-m1)**2/(2*s1**2))/np.sqrt(2*np.pi*s1**2)
                p2 = np.exp(-(img-m2)**2/(2*s2**2))/np.sqrt(2*np.pi*s2**2)
                result[k] = np.logical_and(d==0, p1 > p2) * l
    return result

class Biomedisa(object):
     pass

def active_contour(image_id, friend_id, label_id, simple=False):

    import django
    django.setup()
    from biomedisa_app.models import Upload
    from biomedisa.settings import BASE_DIR, WWW_DATA_ROOT, PRIVATE_STORAGE_ROOT
    from biomedisa_features.create_slices import create_slices
    from biomedisa_features.biomedisa_helper import (unique_file_path,
        save_data, pre_processing, img_to_uint8)
    from redis import Redis
    from rq import Queue

    # create biomedisa
    bm = Biomedisa()
    bm.process = 'acwe'
    bm.django_env = True
    bm.data = None
    bm.labelData = None
    bm.remote, bm.queue = False, 0

    # path to logfiles
    bm.path_to_time = BASE_DIR + '/log/time.txt'
    bm.path_to_logfile = BASE_DIR + '/log/logfile.txt'

    # get objects
    try:
        bm.image = Upload.objects.get(pk=image_id)
        bm.label = Upload.objects.get(pk=label_id)
        friend = Upload.objects.get(pk=friend_id)
        bm.success = True
    except Upload.DoesNotExist:
        bm.success = False

    # pre-processing
    if bm.success:
        bm.path_to_data = bm.image.pic.path.replace(WWW_DATA_ROOT, PRIVATE_STORAGE_ROOT)
        bm.path_to_labels = friend.pic.path.replace(WWW_DATA_ROOT, PRIVATE_STORAGE_ROOT)
        bm.img_id = image_id
        bm.username = bm.image.user.username
        bm.shortfilename = bm.image.shortfilename
        bm = pre_processing(bm)

    if bm.success:

        # final filename
        filename, extension = os.path.splitext(bm.path_to_labels)
        if extension == '.gz':
            filename = filename[:-4]

        # data type
        bm.data = img_to_uint8(bm.data)

        # process data
        if simple:
            final_value = 10
            bm.path_to_acwe = filename + '.refined' + bm.final_image_type
            final = refinement(bm.data, bm.labelData, bm.allLabels)
        else:
            final_value = 3
            bm.path_to_acwe = filename + '.acwe' + bm.final_image_type
            final = activeContour(bm.data, bm.labelData, bm.label.ac_alpha, bm.label.ac_smooth, bm.label.ac_steps, bm.allLabels)

        try:
            # check if final still exists
            friend = Upload.objects.get(pk=friend_id)

            # save result
            bm.path_to_acwe = unique_file_path(bm.path_to_acwe)
            save_data(bm.path_to_acwe, final, bm.header, bm.final_image_type, bm.label.compression)

            # save django object
            shortfilename = os.path.basename(bm.path_to_acwe)
            pic_path = 'images/' + bm.image.user.username + '/' + shortfilename
            Upload.objects.create(pic=pic_path, user=bm.image.user, project=friend.project, final=final_value, imageType=3, shortfilename=shortfilename, friend=friend_id)

            # create slices
            q = Queue('slices', connection=Redis())
            job = q.enqueue_call(create_slices, args=(bm.path_to_data, bm.path_to_acwe,), timeout=-1)

        except Upload.DoesNotExist:
            pass

