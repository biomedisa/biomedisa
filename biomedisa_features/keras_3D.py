##########################################################################
##                                                                      ##
##  Copyright (c) 2023 Philipp Lösel. All rights reserved.              ##
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
import django
django.setup()
import biomedisa_app.views
from biomedisa_app.models import Upload
from biomedisa.settings import BASE_DIR, WWW_DATA_ROOT, PRIVATE_STORAGE_ROOT
from biomedisa_features.active_contour import active_contour
from biomedisa_features.remove_outlier import remove_outlier
from biomedisa_features.create_slices import create_slices
from biomedisa_features.biomedisa_helper import unique_file_path
from biomedisa_app.config import config
from biomedisa_features.keras_helper import *
import biomedisa_features.crop_helper as ch
from multiprocessing import Process
import subprocess

from tensorflow.python.framework.errors_impl import ResourceExhaustedError
import tensorflow as tf
import numpy as np
import h5py
import time

from redis import Redis
from rq import Queue

def conv_network(train, predict, img_list, label_list, path_to_model,
            compress, epochs, batch_size, stride_size, channels, normalize, path_to_img,
            x_scale, y_scale, z_scale, balance, image, label, crop_data, cropping_epochs):

    # get number of GPUs
    strategy = tf.distribute.MirroredStrategy()
    ngpus = int(strategy.num_replicas_in_sync)

    # devide batch size by number of GPUs to maintain consistent training behavior as effective batch size does increase with the number of GPUs
    if train:
        batch_size = batch_size // ngpus

    # batch size must be divisible by two when balancing foreground and background patches
    if balance and train:
        batch_size = batch_size + (batch_size % 2)

    success = False
    path_to_final, path_to_cropped_image = None, None

    # dimensions of patches for regular training
    z_patch, y_patch, x_patch = 64, 64, 64

    # adapt scaling and stridesize to patchsize
    stride_size = max(1, min(stride_size, 64))
    x_scale = x_scale - (x_scale - 64) % stride_size
    y_scale = y_scale - (y_scale - 64) % stride_size
    z_scale = z_scale - (z_scale - 64) % stride_size

    if train:

        try:
            # train automatic cropping
            cropping_weights, cropping_config = None, None
            if crop_data:
                cropping_weights, cropping_config = ch.load_and_train(normalize, img_list, label_list, path_to_model,
                    cropping_epochs, batch_size * ngpus, label.validation_split, x_scale, y_scale, z_scale,
                    label.flip_x, label.flip_y, label.flip_z, label.rotate, label.only, label.ignore)

            # train network
            train_semantic_segmentation(normalize, img_list, label_list, x_scale, y_scale,
                    z_scale, crop_data, path_to_model, z_patch, y_patch, x_patch, epochs,
                    batch_size, channels, label.validation_split, stride_size, balance,
                    label.flip_x, label.flip_y, label.flip_z, label.rotate, image,
                    label.early_stopping, label.val_tf, int(label.validation_freq),
                    cropping_weights, cropping_config, label.only, label.ignore,
                    label.filters, label.resnet)

        except InputError:
            return success, InputError.message, None, None, None
        except MemoryError:
            print('MemoryError')
            return success, 'MemoryError', None, None, None
        except ResourceExhaustedError:
            print('GPU out of memory')
            return success, 'GPU out of memory', None, None, None
        except Exception as e:
            print('Error:', e)
            return success, e, None, None, None

    if predict:

        try:
            # get meta data
            hf = h5py.File(path_to_model, 'r')
            meta = hf.get('meta')
            configuration = meta.get('configuration')
            channels, x_scale, y_scale, z_scale, normalize, mu, sig = np.array(configuration)[:]
            channels, x_scale, y_scale, z_scale, normalize, mu, sig = int(channels), int(x_scale), \
                                    int(y_scale), int(z_scale), int(normalize), float(mu), float(sig)
            allLabels = np.array(meta.get('labels'))
            header = np.array(meta.get('header'))
            try:
                extension = meta.get('extension').asstr()[()]
            except:
                extension = str(np.array(meta.get('extension')))
            crop_data = True if 'cropping_weights' in hf else False
            hf.close()
        except Exception as e:
            print('Error:', e)
            return success, 'Invalid Biomedisa Network', None

        # if header is not Amira falling back to Multi-TIFF
        if extension != '.am':
            extension = '.tif'
            header = None

        # create path_to_final
        filename = os.path.basename(path_to_img)
        filename = os.path.splitext(filename)[0]
        if filename[-4:] in ['.nii','.tar']:
            filename = filename[:-4]
        filename = 'final.' + filename
        path_to_final = os.path.dirname(path_to_img) + '/' + filename + extension
        path_to_final = unique_file_path(path_to_final, image.user.username)

        try:
            # crop data
            region_of_interest = None
            if crop_data:
                filename = os.path.basename(path_to_img)
                filename = os.path.splitext(filename)[0]
                if filename[-4:] in ['.nii','.tar']:
                    filename = filename[:-4]
                filename = filename + '.cropped.tif'
                path_to_cropped_image = os.path.dirname(path_to_img) + '/' + filename
                path_to_cropped_image = unique_file_path(path_to_cropped_image, image.user.username)
                region_of_interest, cropped_volume = ch.crop_data(path_to_img, path_to_model, path_to_cropped_image, batch_size)

            # load prediction data
            img, img_header, position, z_shape, y_shape, x_shape, region_of_interest = load_prediction_data(path_to_img,
                channels, x_scale, y_scale, z_scale, normalize, mu, sig, region_of_interest)

            # make prediction
            predict_semantic_segmentation(img, position, path_to_model, path_to_final,
                z_patch, y_patch, x_patch,  z_shape, y_shape, x_shape, compress, header,
                img_header, channels, stride_size, allLabels, batch_size, region_of_interest)

        except InputError:
            return success, InputError.message, None, None, None
        except MemoryError:
            print('MemoryError')
            return success, 'MemoryError', None, None, None
        except ResourceExhaustedError:
            print('GPU out of memory')
            return success, 'GPU out of memory', None, None, None
        except Exception as e:
            print('Error:', e)
            return success, e, None, None, None

    success = True
    return success, None, path_to_final, path_to_cropped_image

if __name__ == '__main__':

    # time
    TIC = time.time()

    # get objects
    try:
        image = Upload.objects.get(pk=sys.argv[1])
        label = Upload.objects.get(pk=sys.argv[2])
        success = True
    except Upload.DoesNotExist:
        success = False
        message = 'Files have been removed.'
        Upload.objects.create(user=image.user, project=image.project, log=1, imageType=None, shortfilename=message)

    # check if aborted or files have been removed
    if image.status > 0 and success:

        # set PID
        image.pid = int(os.getpid())
        image.path_to_model = ''
        image.save()

        # get arguments
        model_id = int(sys.argv[3])
        refine_model_id = int(sys.argv[4])  # dead end
        predict = int(sys.argv[5])
        img_list = sys.argv[6].split(';')[:-1]
        label_list = sys.argv[7].split(';')[:-1]
        for k in range(len(img_list)):
            img_list[k] = img_list[k].replace(WWW_DATA_ROOT, PRIVATE_STORAGE_ROOT)
            label_list[k] = label_list[k].replace(WWW_DATA_ROOT, PRIVATE_STORAGE_ROOT)

        # path to image
        path_to_img = image.pic.path.replace(WWW_DATA_ROOT, PRIVATE_STORAGE_ROOT)

        # path to files
        path_to_time = BASE_DIR + '/log/time.txt'
        path_to_logfile = BASE_DIR + '/log/logfile.txt'

        # train or predict
        if predict:
            train = False
            model = Upload.objects.get(pk=model_id)
            project = os.path.splitext(model.shortfilename)[0]
        else:
            train = True
            project = os.path.splitext(image.shortfilename)[0]

        # write in logs and send notification
        if train:
            biomedisa_app.views.send_start_notification(image)
        with open(path_to_logfile, 'a') as logfile:
            print('%s %s %s %s' %(time.ctime(), image.user.username, image.shortfilename, 'Process was started.'), file=logfile)
            print('PROJECT:%s PREDICT:%s IMG:%s LABEL:%s IMG_LIST:%s LABEL_LIST:%s'
                 %(project, predict, image.shortfilename, label.shortfilename, img_list, label_list), file=logfile)

        # create path_to_model
        if train:
            extension = '.h5'
            dir_path = BASE_DIR + '/private_storage/'
            model_path = 'images/%s/%s' %(image.user.username, project)
            path_to_model = dir_path + model_path + extension
            path_to_model = unique_file_path(path_to_model, image.user.username)
            image.path_to_model = path_to_model.replace(PRIVATE_STORAGE_ROOT, WWW_DATA_ROOT)
            image.save()
        else:
            path_to_model = model.pic.path.replace(WWW_DATA_ROOT, PRIVATE_STORAGE_ROOT)

        # parameters
        compress = 1 if label.compression else 0            # wheter final result should be compressed or not
        epochs =  int(label.epochs)                         # epochs the network is trained
        channels = 2 if label.position else 1               # use voxel coordinates
        normalize = 1 if label.normalize else 0             # normalize images before training
        balance = 1 if label.balance else 0                 # the number of training patches must be equal for foreground and background
        x_scale = int(label.x_scale)                        # images are scaled at x-axis to this size before training
        y_scale = int(label.y_scale)                        # images are scaled at y-axis to this size before training
        z_scale = int(label.z_scale)                        # images are scaled at z-axis to this size before training
        crop_data = 1 if label.automatic_cropping else 0    # crop data automatically to region of interest
        cropping_epochs = 50                                # Epochs the network for auto-cropping is trained

        # batch size and stride size which is made for generating patches
        if model_id > 0:
            stride_size = int(model.stride_size)
            batch_size = int(model.batch_size)
        else:
            stride_size = int(label.stride_size)
            batch_size = int(label.batch_size)

        # train network or predict segmentation
        success, error_message, path_to_final, path_to_cropped_image = conv_network(
            train, predict, img_list, label_list, path_to_model, compress,
            epochs, batch_size, stride_size, channels,
            normalize, path_to_img, x_scale, y_scale, z_scale, balance, image, label, crop_data,
            cropping_epochs)

        if success:

            if train:
                # create model object
                shortfilename = os.path.basename(path_to_model)
                pic_path = 'images/' + image.user.username + '/' + shortfilename
                Upload.objects.create(pic=pic_path, user=image.user, project=image.project, imageType=4, shortfilename=shortfilename)

            if predict:
                # create final object
                shortfilename = os.path.basename(path_to_final)
                pic_path = 'images/' + image.user.username + '/' + shortfilename
                tmp = Upload.objects.create(pic=pic_path, user=image.user, project=image.project, final=1, active=1, imageType=3, shortfilename=shortfilename)
                tmp.friend = tmp.id
                tmp.save()

                # save cropped image object
                if path_to_cropped_image:
                    shortfilename = os.path.basename(path_to_cropped_image)
                    pic_path = 'images/' + image.user.username + '/' + shortfilename
                    Upload.objects.create(pic=pic_path, user=image.user, project=image.project, final=9, active=0, imageType=3, shortfilename=shortfilename, friend=tmp.id)

                # create slices, cleanup and acwe
                q = Queue('slices', connection=Redis())
                job = q.enqueue_call(create_slices, args=(path_to_img, path_to_final,), timeout=-1)
                q = Queue('acwe', connection=Redis())
                job = q.enqueue_call(active_contour, args=(image.id, tmp.id, model.id,), timeout=-1)
                q = Queue('cleanup', connection=Redis())
                job = q.enqueue_call(remove_outlier, args=(image.id, tmp.id, tmp.id, model.id,), timeout=-1)
                if path_to_cropped_image:
                    q = Queue('slices', connection=Redis())
                    job = q.enqueue_call(create_slices, args=(path_to_cropped_image, None,), timeout=-1)

            # write in logs and send notification
            if predict:
                message = 'Successfully segmented ' + image.shortfilename
            else:
                message = 'Successfully trained ' + project
            t = int(time.time() - TIC)
            if t < 60:
                time_str = str(t) + " sec"
            elif 60 <= t < 3600:
                time_str = str(t // 60) + " min " + str(t % 60) + " sec"
            elif 3600 < t:
                time_str = str(t // 3600) + " h " + str((t % 3600) // 60) + " min " + str(t % 60) + " sec"
            with open(path_to_time, 'a') as timefile:
                print('%s %s %s %s on %s' %(time.ctime(), image.user.username, message, time_str, config['SERVER_ALIAS']), file=timefile)
            print('Total calculation time:', time_str)
            biomedisa_app.views.send_notification(image.user.username, image.shortfilename, time_str, config['SERVER_ALIAS'], train, predict)

        else:
            # return error message
            Upload.objects.create(user=image.user, project=image.project, log=1, imageType=None, shortfilename=error_message)
            with open(path_to_logfile, 'a') as logfile:
                print('%s %s %s %s' %(time.ctime(), image.user.username, image.shortfilename, error_message), file=logfile)
            biomedisa_app.views.send_error_message(image.user.username, image.shortfilename, error_message)

