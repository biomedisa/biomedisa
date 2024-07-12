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
from biomedisa.features.biomedisa_helper import (load_data, save_data,
    unique_file_path, silent_remove)
import numpy as np
from scipy import ndimage
import argparse
import traceback
import subprocess

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
    data = np.copy(data[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x], order='C')
    return data, argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x

def clean(image, threshold=0.1):
    image_i = np.copy(image, order='C')
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
    image_i = np.copy(image, order='C')
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

def main_helper(path_to_labels, img_id=None, friend_id=None, fill_holes=True,
    clean_threshold=0.1, fill_threshold=0.9, remote=False, compression=True):

    # django environment
    if img_id is not None:
        django_env = True
    else:
        django_env = False

    # final filenames
    filename, extension = os.path.splitext(path_to_labels)
    if extension == '.gz':
        extension = '.nii.gz'
        filename = filename[:-4]
    path_to_cleaned = filename + '.cleaned' + extension
    path_to_filled = filename + '.filled' + extension
    path_to_cleaned_filled = filename + '.cleaned.filled' + extension

    # load data
    final, header = load_data(path_to_labels, 'cleanup')

    # process data
    final_cleaned = clean(final, clean_threshold)
    if fill_holes:
        final_filled = fill(final, fill_threshold)
        final_cleaned_filled = final_cleaned + (final_filled - final)

    # unique_file_paths
    if django_env and not remote:
        path_to_cleaned = unique_file_path(path_to_cleaned)
        path_to_filled = unique_file_path(path_to_filled)
        path_to_cleaned_filled = unique_file_path(path_to_cleaned_filled)

    # save results
    save_data(path_to_cleaned, final_cleaned, header, extension, compression)
    if fill_holes:
        save_data(path_to_filled, final_filled, header, extension, compression)
        save_data(path_to_cleaned_filled, final_cleaned_filled, header, extension, compression)

    # post processing
    post_processing(path_to_cleaned, path_to_filled, path_to_cleaned_filled, img_id, friend_id, fill_holes, remote)

def post_processing(path_to_cleaned, path_to_filled, path_to_cleaned_filled, img_id=None, friend_id=None, fill_holes=False, remote=False):
    if remote:
        with open(biomedisa.BASE_DIR + '/log/config_6', 'w') as configfile:
            print(path_to_cleaned, path_to_filled, path_to_cleaned_filled, file=configfile)
    else:
        import django
        django.setup()
        from biomedisa_app.models import Upload
        from biomedisa.features.create_slices import create_slices
        from redis import Redis
        from rq import Queue

        # check if reference data still exists
        image = Upload.objects.filter(pk=img_id)
        friend = Upload.objects.filter(pk=friend_id)
        if len(friend)>0:
            friend = friend[0]

            # save django object
            shortfilename = os.path.basename(path_to_cleaned)
            pic_path = 'images/' + friend.user.username + '/' + shortfilename
            Upload.objects.create(pic=pic_path, user=friend.user, project=friend.project, final=(2 if fill_holes else 6), imageType=3, shortfilename=shortfilename, friend=friend_id)

            # create slices for sliceviewer
            if len(image)>0:
                q_slices = Queue('slices', connection=Redis())
                job = q_slices.enqueue_call(create_slices, args=(image[0].pic.path, path_to_cleaned,), timeout=-1)

            # fill holes
            if fill_holes:
                # save django object
                shortfilename = os.path.basename(path_to_cleaned_filled)
                pic_path = 'images/' + friend.user.username + '/' + shortfilename
                Upload.objects.create(pic=pic_path, user=friend.user, project=friend.project, final=8, imageType=3, shortfilename=shortfilename, friend=friend_id)
                shortfilename = os.path.basename(path_to_filled)
                pic_path = 'images/' + friend.user.username + '/' + shortfilename
                Upload.objects.create(pic=pic_path, user=friend.user, project=friend.project, final=7, imageType=3, shortfilename=shortfilename, friend=friend_id)

                # create slices for sliceviewer
                if len(image)>0:
                    q_slices = Queue('slices', connection=Redis())
                    job = q_slices.enqueue_call(create_slices, args=(image[0].pic.path, path_to_filled,), timeout=-1)
                    job = q_slices.enqueue_call(create_slices, args=(image[0].pic.path, path_to_cleaned_filled,), timeout=-1)
        else:
            silent_remove(path_to_cleaned)
            silent_remove(path_to_filled)
            silent_remove(path_to_cleaned_filled)

def init_remove_outlier(image_id, final_id, label_id, fill_holes=True):
    '''
    Runs clean() and fill() within django environment/webbrowser version

    Parameters
    ---------
    image_id: int
        Django id of image data used for creating slice preview
    final_id: int
        Django id of result data to be processed
    label_id: int
        Django id of label data used for configuration parameters
    fill_holes: bool
        Fill holes and save as an optional result

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
        final = Upload.objects.get(pk=final_id)
        label = Upload.objects.get(pk=label_id)
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
            cmd = ['python3', host_base+'/biomedisa/features/remove_outlier.py', final.pic.path.replace(biomedisa.BASE_DIR,host_base)]
            cmd += [f'-iid={image.id}', f'-fid={final.friend}', '-r']

            # command (append only on demand)
            if fill_holes:
                cmd += ['-fh']
            if not label.compression:
                cmd += ['-nc']
            if label.delete_outliers != 0.1:
                cmd += [f'-c={label.delete_outliers}']
            if label.fill_holes != 0.9:
                cmd += [f'-f={label.fill_holes}']

            # create user directory
            subprocess.Popen(['ssh', host, 'mkdir', '-p', host_base+'/private_storage/images/'+image.user.username]).wait()

            # send data to host
            success = send_data_to_host(final.pic.path, host+':'+final.pic.path.replace(biomedisa.BASE_DIR,host_base))

            if success==0:

                # qsub start
                if 'REMOTE_QUEUE_QSUB' in config and config['REMOTE_QUEUE_QSUB']:
                    subhost, qsub_pid = qsub_start(host, host_base, 6)

                # start removing outliers
                if subhost:
                    cmd = ['ssh', '-t', host, 'ssh', subhost] + cmd
                else:
                    cmd = ['ssh', host] + cmd
                subprocess.Popen(cmd).wait()

                # config
                success = subprocess.Popen(['scp', host+':'+host_base+'/log/config_6', biomedisa.BASE_DIR+'/log/config_6']).wait()

                if success==0:
                    with open(biomedisa.BASE_DIR + '/log/config_6', 'r') as configfile:
                        cleaned_on_host, filled_on_host, cleaned_filled_on_host = configfile.read().split()

                    # local file names
                    path_to_cleaned = unique_file_path(cleaned_on_host.replace(host_base,biomedisa.BASE_DIR))
                    path_to_filled = unique_file_path(filled_on_host.replace(host_base,biomedisa.BASE_DIR))
                    path_to_cleaned_filled = unique_file_path(cleaned_filled_on_host.replace(host_base,biomedisa.BASE_DIR))

                    # get results
                    subprocess.Popen(['scp', host+':'+cleaned_on_host, path_to_cleaned]).wait()
                    if fill_holes:
                        subprocess.Popen(['scp', host+':'+filled_on_host, path_to_filled]).wait()
                        subprocess.Popen(['scp', host+':'+cleaned_filled_on_host, path_to_cleaned_filled]).wait()

                    # post processing
                    post_processing(path_to_cleaned, path_to_filled, path_to_cleaned_filled, image_id, final.friend, fill_holes)

                    # remove config file
                    subprocess.Popen(['ssh', host, 'rm', host_base + '/log/config_6']).wait()

        # local server
        else:
            try:
                main_helper(final.pic.path, img_id=image_id, friend_id=final.friend,
                    fill_holes=fill_holes, clean_threshold=label.delete_outliers, fill_threshold=label.fill_holes, remote=False,
                    compression=label.compression)
            except Exception as e:
                print(traceback.format_exc())

    # qsub stop
    if 'REMOTE_QUEUE_QSUB' in config and config['REMOTE_QUEUE_QSUB']:
        qsub_stop(host, host_base, 6, 'cleanup', subhost, qsub_pid)

if __name__ == '__main__':

    # initialize arguments
    parser = argparse.ArgumentParser(description='Biomedisa remove outliers.',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required arguments
    parser.add_argument('path_to_labels', type=str, metavar='PATH_TO_LABELS',
                        help='Location of label data')

    # optional arguments
    parser.add_argument('-v', '--version', action='version', version=f'{biomedisa.__version__}',
                        help='Biomedisa version')
    parser.add_argument('-fh','--fill_holes', action='store_true', default=False,
                        help='Fill holes and save as an optional result')
    parser.add_argument('-c', '--clean_threshold', type=float, default=0.1,
                        help='Remove outliers, e.g. 0.5 means that objects smaller than 50 percent of the size of the largest object will be removed')
    parser.add_argument('-f', '--fill_threshold', type=float, default=0.9,
                        help='Fill holes, e.g. 0.5 means that all holes smaller than 50 percent of the entire label will be filled')
    parser.add_argument('-nc', '--no-compression', dest='compression', action='store_false',
                        help='Disable compression of segmentation results')
    parser.add_argument('-iid','--img_id', type=str, default=None,
                        help='Image ID within django environment/browser version')
    parser.add_argument('-fid','--friend_id', type=str, default=None,
                        help='Label ID within django environment/browser version')
    parser.add_argument('-r','--remote', action='store_true', default=False,
                        help='Process is carried out on a remote server. Must be set up in config.py')

    kwargs = vars(parser.parse_args())

    # main function
    try:
        main_helper(**kwargs)
    except Exception as e:
        print(traceback.format_exc())

