#!/usr/bin/python3
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

import os
import biomedisa
from biomedisa.features.curvop_numba import curvop, evolution
from biomedisa.features.biomedisa_helper import (unique_file_path, load_data, save_data,
    pre_processing, img_to_uint8, silent_remove)
import numpy as np
import numba
import argparse
import traceback
import subprocess

class Biomedisa(object):
     pass

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

def activeContour(data, labelData, alpha=1.0, smooth=1, steps=3,
    path_to_data=None, path_to_labels=None, compression=True,
    ignore='none', only='all', simple=False,
    img_id=None, friend_id=None, remote=False):

    # create biomedisa
    bm = Biomedisa()
    bm.process = 'acwe'
    bm.success = True

    # transfer arguments
    key_copy = tuple(locals().keys())
    for arg in key_copy:
        bm.__dict__[arg] = locals()[arg]

    # django environment
    if bm.img_id is not None:
        bm.django_env = True
    else:
        bm.django_env = False

    # disable file saving when called as a function
    if bm.data is not None:
        bm.path_to_data = None
        bm.path_to_labels = None

    if bm.django_env:
        bm.username = os.path.basename(os.path.dirname(bm.path_to_data))
        bm.shortfilename = os.path.basename(bm.path_to_data)
        bm.path_to_logfile = biomedisa.BASE_DIR + '/log/logfile.txt'

    # pre-processing
    bm = pre_processing(bm)

    # create path_to_acwe
    if bm.path_to_data:
        filename, extension = os.path.splitext(bm.path_to_labels)
        if extension == '.gz':
            filename = filename[:-4]
        suffix='.refined' if simple else '.acwe'
        path_to_acwe = filename + suffix + bm.final_image_type

    if bm.success:

        # data type
        bm.data = img_to_uint8(bm.data)

        # append data
        zsh, ysh, xsh = bm.data.shape
        tmp = np.zeros((2+zsh, 2+ysh, 2+xsh), dtype=bm.data.dtype)
        tmp[1:-1,1:-1,1:-1] = bm.data
        bm.data = np.copy(tmp, order='C')
        tmp = np.zeros((2+zsh, 2+ysh, 2+xsh), dtype=bm.labelData.dtype)
        tmp[1:-1,1:-1,1:-1] = bm.labelData
        bm.labelData = np.copy(tmp, order='C')
        zsh, ysh, xsh = bm.data.shape

        # reduce blocksize
        bm.data, bm.labelData, argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = reduce_blocksize(bm.data, bm.labelData)
        bm.labelData = np.copy(bm.labelData, order='C')
        bm.data = np.copy(bm.data, order='C')

        # active contour
        if simple:
            bm.labelData = refinement(bm)
        else:
            for n in range(bm.steps):
                print('Step:', n+1, '/', bm.steps)
                mean = np.zeros(256, dtype=np.float32)
                for k in bm.allLabels:
                    inside = bm.labelData==k
                    if np.any(inside):
                        mean[k] = np.mean(bm.data[inside])
                bm.labelData = evolution(mean, bm.labelData, bm.data, bm.alpha)
                for k in bm.allLabels:
                    bm.labelData = curvop(bm.labelData, bm.smooth, k, bm.allLabels)

        # return to original data size
        final = np.zeros((zsh, ysh, xsh), dtype=np.uint8)
        final[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x] = bm.labelData
        final = np.copy(final[1:-1, 1:-1, 1:-1], order='C')

        # save result
        if bm.django_env and not bm.remote:
            path_to_acwe = unique_file_path(path_to_acwe)
        if bm.path_to_data:
            save_data(path_to_acwe, final, bm.header, bm.final_image_type, bm.compression)

        # post processing
        if bm.django_env:
            post_processing(path_to_acwe, bm.img_id, bm.friend_id, bm.simple, bm.remote)

        return final

def refinement(bm):
    zsh, ysh, xsh = bm.data.shape
    distance = np.zeros(bm.data.shape, dtype=np.float32)
    distance[bm.labelData==0] = 100
    distance = geodis(distance,np.sqrt(2),np.sqrt(3),1)
    distance[distance<=10] = 0
    distance[distance>10] = 1
    distance = distance.astype(np.uint8)
    result = np.zeros_like(bm.labelData)
    for l in bm.allLabels[1:]:
        for k in range(zsh):
            img = bm.data[k]
            mask = bm.labelData[k]
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

def post_processing(path_to_acwe, image_id=None, friend_id=None, simple=False, remote=False):
    if remote:
        with open(biomedisa.BASE_DIR + '/log/config_4', 'w') as configfile:
            print(path_to_acwe, 'phantom', file=configfile)
    else:
        import django
        django.setup()
        from biomedisa_app.models import Upload
        from biomedisa.features.create_slices import create_slices
        from redis import Redis
        from rq import Queue

        # check if reference data still exists
        image = Upload.objects.filter(pk=image_id)
        friend = Upload.objects.filter(pk=friend_id)
        if len(friend)>0:
            friend = friend[0]

            # create django object
            shortfilename = os.path.basename(path_to_acwe)
            pic_path = 'images/' + friend.user.username + '/' + shortfilename
            Upload.objects.create(pic=pic_path, user=friend.user, project=friend.project, final=(10 if simple else 3), imageType=3, shortfilename=shortfilename, friend=friend_id)

            # create slices
            if len(image)>0:
                q = Queue('slices', connection=Redis())
                job = q.enqueue_call(create_slices, args=(image[0].pic.path, path_to_acwe,), timeout=-1)
        else:
            silent_remove(path_to_acwe)

def init_active_contour(image_id, friend_id, label_id, simple=False):
    '''
    Runs activeContour() within django environment/webbrowser version

    Parameters
    ---------
    image_id: int
        Django id of image data
    friend_id: int
        Django id of result data to be processed
    label_id: int
        Django id of label data used for configuration parameters
    simple: bool
        Use simplified version of active contour

    Returns
    -------
    No returns
        Fails silently
    '''

    import django
    django.setup()
    from biomedisa_app.models import Upload
    from biomedisa_app.config import config
    from biomedisa_app.views import send_data_to_host, qsub_start, qsub_stop

    # get objects
    try:
        image = Upload.objects.get(pk=image_id)
        label = Upload.objects.get(pk=label_id)
        friend = Upload.objects.get(pk=friend_id)
        success = True
    except Upload.DoesNotExist:
        success = False

    # get host information
    host = ''
    host_base = biomedisa.BASE_DIR
    subhost, qsub_pid = None, None
    if 'REMOTE_QUEUE_HOST' in config:
        host = config['REMOTE_QUEUE_HOST']
    if host and 'REMOTE_QUEUE_BASE_DIR' in config:
        host_base = config['REMOTE_QUEUE_BASE_DIR']

    if success:

        # remote server
        if host:

            # command
            cmd = ['python3', host_base+'/biomedisa/features/active_contour.py']
            cmd += [image.pic.path.replace(biomedisa.BASE_DIR,host_base), friend.pic.path.replace(biomedisa.BASE_DIR,host_base)]
            cmd += [f'-iid={image.id}', f'-fid={friend.id}', '-r']

            # command (append only on demand)
            if simple:
                cmd += ['-si']
            if not label.compression:
                cmd += ['-nc']
            if label.ignore != 'none':
                cmd += [f'-i={label.ignore}']
            if label.only != 'all':
                cmd += [f'-o={label.only}']
            if label.ac_smooth != 1:
                cmd += [f'-s={label.ac_smooth}']
            if label.ac_steps != 3:
                cmd += [f'-st={label.ac_steps}']
            if label.ac_alpha != 1.0:
                cmd += [f'-a={label.ac_alpha}']

            # create user directory
            subprocess.Popen(['ssh', host, 'mkdir', '-p', host_base+'/private_storage/images/'+image.user.username]).wait()

            # send data to host
            success=0
            success+=send_data_to_host(image.pic.path, host+':'+image.pic.path.replace(biomedisa.BASE_DIR,host_base))
            success+=send_data_to_host(friend.pic.path, host+':'+friend.pic.path.replace(biomedisa.BASE_DIR,host_base))

            if success==0:

                # qsub start
                if 'REMOTE_QUEUE_QSUB' in config and config['REMOTE_QUEUE_QSUB']:
                    subhost, qsub_pid = qsub_start(host, host_base, 4)

                # start active contour
                if subhost:
                    cmd = ['ssh', '-t', host, 'ssh', subhost] + cmd
                else:
                    cmd = ['ssh', host] + cmd
                subprocess.Popen(cmd).wait()

                # config
                success = subprocess.Popen(['scp', host+':'+host_base+'/log/config_4', biomedisa.BASE_DIR+'/log/config_4']).wait()

                if success==0:
                    with open(biomedisa.BASE_DIR + '/log/config_4', 'r') as configfile:
                        acwe_on_host, _ = configfile.read().split()

                    # local file names
                    path_to_acwe = unique_file_path(acwe_on_host.replace(host_base,biomedisa.BASE_DIR))

                    # get results
                    subprocess.Popen(['scp', host+':'+acwe_on_host, path_to_acwe]).wait()

                    # post processing
                    post_processing(path_to_acwe, image_id=image_id, friend_id=friend_id, simple=simple)

                    # remove config file
                    subprocess.Popen(['ssh', host, 'rm', host_base + '/log/config_4']).wait()

        # local server
        else:
            try:
                activeContour(None, None, path_to_data=image.pic.path, path_to_labels=friend.pic.path,
                    alpha=label.ac_alpha, smooth=label.ac_smooth, steps=label.ac_steps, compression=label.compression,
                    simple=simple, img_id=image_id, friend_id=friend_id, remote=False)
            except Exception as e:
                print(traceback.format_exc())

    # qsub stop
    if 'REMOTE_QUEUE_QSUB' in config and config['REMOTE_QUEUE_QSUB']:
        qsub_stop(host, host_base, 4, 'acwe', subhost, qsub_pid)

if __name__ == '__main__':

    # initialize arguments
    parser = argparse.ArgumentParser(description='Biomedisa active contour.',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required arguments
    parser.add_argument('path_to_data', type=str, metavar='PATH_TO_IMAGE',
                        help='Location of image data')
    parser.add_argument('path_to_labels', type=str, metavar='PATH_TO_LABELS',
                        help='Location of label data')

    # optional arguments
    parser.add_argument('-v', '--version', action='version', version=f'{biomedisa.__version__}',
                        help='Biomedisa version')
    parser.add_argument('-si','--simple', action='store_true', default=False,
                        help='Simplified version of active contour')
    parser.add_argument('-a', '--alpha', type=float, default=1.0,
                        help='Driving force of contour')
    parser.add_argument('-s', '--smooth', type=int, default=1,
                        help='Number of smoothing steps')
    parser.add_argument('-st', '--steps', type=int, default=3,
                        help='Number of iterations')
    parser.add_argument('-nc', '--no-compression', dest='compression', action='store_false',
                        help='Disable compression of segmentation results')
    parser.add_argument('-i', '--ignore', type=str, default='none',
                        help='Ignore specific label(s), e.g. 2,5,6')
    parser.add_argument('-o', '--only', type=str, default='all',
                        help='Segment only specific label(s), e.g. 1,3,5')
    parser.add_argument('-iid','--img_id', type=str, default=None,
                        help='Image ID within django environment/browser version')
    parser.add_argument('-fid','--friend_id', type=str, default=None,
                        help='Label ID within django environment/browser version')
    parser.add_argument('-r','--remote', action='store_true', default=False,
                        help='Process is carried out on a remote server. Must be set up in config.py')

    kwargs = vars(parser.parse_args())

    # run active contour
    try:
        activeContour(None, None, **kwargs)
    except Exception as e:
        print(traceback.format_exc())

