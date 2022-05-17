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

from amira_to_np.amira_helper import amira_to_np, np_to_amira
from tifffile import imread, imwrite
from medpy.io import load, save
from PIL import Image
import numpy as np
import glob
import os
import random
import cv2
import time
import zipfile

def img_resize(a, z_shape, y_shape, x_shape, interpolation=None):
    zsh, ysh, xsh = a.shape
    if interpolation == None:
        if z_shape < zsh or y_shape < ysh or x_shape < xsh:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC
    b = np.empty((zsh, y_shape, x_shape), dtype=a.dtype)
    for k in range(zsh):
        b[k] = cv2.resize(a[k], (x_shape, y_shape), interpolation=interpolation)
    c = np.empty((y_shape, z_shape, x_shape), dtype=a.dtype)
    b = np.swapaxes(b, 0, 1)
    b = np.copy(b, order='C')
    for k in range(y_shape):
        c[k] = cv2.resize(b[k], (x_shape, z_shape), interpolation=interpolation)
    c = np.swapaxes(c, 1, 0)
    c = np.copy(c, order='C')
    return c

def img_to_uint8(img):
    if img.dtype != 'uint8':
        img = img.astype(np.float32)
        img -= np.amin(img)
        img /= np.amax(img)
        img *= 255.0
        img = img.astype(np.uint8)
    return img

def id_generator(size, chars):
    return ''.join(random.choice(chars) for x in range(size))

def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

def load_data_(path_to_data, process):

    if os.path.isdir(path_to_data):
        path_to_data = path_to_data+'.zip'

    extension = os.path.splitext(path_to_data)[1]
    if extension == '.gz':
        extension = '.nii.gz'

    if extension == '.am':
        try:
            data, header = amira_to_np(path_to_data)
            if len(data) > 1:
                extension, header = '.tif', None
                print('Warning! Multiple data streams are not supported. Falling back to TIFF.')
            data = data[0]
        except Exception as e:
            print(e)
            data, header = None, None

    elif extension in ['.hdr', '.mhd', '.mha', '.nrrd', '.nii', '.nii.gz']:
        try:
            data, header = load(path_to_data)
            data = np.swapaxes(data, 0, 2)
            data = np.copy(data, order='C')
        except Exception as e:
            print(e)
            data, header = None, None

    elif extension == '.zip':
        try:
            # extract data if necessary
            if not os.path.isdir(path_to_data[:-4]):
                zip_ref = zipfile.ZipFile(path_to_data, 'r')
                zip_ref.extractall(path=path_to_data[:-4])
                zip_ref.close()

            files = glob.glob(path_to_data[:-4]+'/**/*', recursive=True)
            for name in files:
                if os.path.isfile(name):
                    try:
                        img, _ = load(name)
                    except:
                        files.remove(name)
                else:
                    files.remove(name)
            files.sort()

            img, _ = load(files[0])
            data = np.zeros((len(files), img.shape[0], img.shape[1]), dtype=img.dtype)
            header, image_data_shape = [], []
            for k, file_name in enumerate(files):
                img, img_header = load(file_name)
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = rgb2gray(img)
                    extension = '.tif'
                elif len(img.shape) == 3 and img.shape[2] == 1:
                    img = img[:,:,0]
                data[k] = img
                header.append(img_header)
            header = [header, files, data.dtype]
            data = np.swapaxes(data, 1, 2)
            data = np.copy(data, order='C')
        except Exception as e:
            print(e)
            data, header = None, None

    elif extension in ['.tif', '.tiff']:
        try:
            data = imread(path_to_data)
            header = None
        except Exception as e:
            print(e)
            data, header = None, None

    else:
        data, header = None, None

    return data, header, extension

def load_data(path_to_data, process='None', return_extension=False):
    data, header, extension = load_data_(path_to_data, process)
    if return_extension:
        return data, header, extension
    else:
        return data, header

def _error_(bm, message):
    print(message)
    bm.success = False
    return bm

def pre_processing(bm):

    # load data
    bm.data, _ = load_data(bm.path_to_data, bm.process)

    # error handling
    if bm.data is None:
        return _error_(bm, 'Invalid image data.')

    # load label data
    bm.labelData, bm.header, bm.final_image_type = load_data(bm.path_to_labels, bm.process, True)

    # error handling
    if bm.labelData is None:
        return _error_(bm, 'Invalid label data.')

    if len(bm.labelData.shape) != 3:
        return _error_(bm, 'Label must be three-dimensional.')

    if bm.data.shape != bm.labelData.shape:
        return _error_(bm, 'Image and label must have the same x,y,z-dimensions.')

    # pre-process label data
    bm.labelData = color_to_gray(bm.labelData)
    bm.labelData = bm.labelData.astype(np.int32)
    bm.allLabels = np.unique(bm.labelData)

    # add background label if not existing
    if not np.any(bm.allLabels==0):
        bm.allLabels = np.append(0, bm.allLabels)

    # compute only specific labels
    labels_to_compute = (bm.label.only).split(',')
    if not any([x in ['all', 'All', 'ALL'] for x in labels_to_compute]):
        labels_to_remove = [k for k in bm.allLabels if str(k) not in labels_to_compute and k > 0]
        for k in labels_to_remove:
            bm.labelData[bm.labelData == k] = 0
            index = np.argwhere(bm.allLabels==k)
            bm.allLabels = np.delete(bm.allLabels, index)

    # ignore specific labels
    labels_to_remove = (bm.label.ignore).split(',')
    if not any([x in ['none', 'None', 'NONE'] for x in labels_to_remove]):
        for k in labels_to_remove:
            try:
                k = int(k)
                bm.labelData[bm.labelData == k] = 0
                index = np.argwhere(bm.allLabels==k)
                bm.allLabels = np.delete(bm.allLabels, index)
            except:
                pass

    # number of labels
    bm.nol = len(bm.allLabels)

    if bm.nol < 2:
        return _error_(bm, 'No labeled slices found.')

    if np.any(bm.allLabels > 255):
        return _error_(bm, 'No labels higher than 255 allowed.')

    if np.any(bm.allLabels < 0):
        return _error_(bm, 'No negative labels allowed.')

    bm.success = True
    return bm

def save_data(path_to_final, final, header=None, final_image_type=None, compress=True):
    if final_image_type == None:
        final_image_type = os.path.splitext(path_to_final)[1]
        if final_image_type == '.gz':
            final_image_type = '.nii.gz'
    if final_image_type == '.am':
        np_to_amira(path_to_final, [final], header)
    elif final_image_type in ['.hdr', '.mhd', '.mha', '.nrrd', '.nii', '.nii.gz']:
        final = np.swapaxes(final, 0, 2)
        save(final, path_to_final, header)
    elif final_image_type == '.zip':
        header, file_names, final_dtype = header[0], header[1], header[2]
        final = final.astype(final_dtype)
        final = np.swapaxes(final, 2, 1)
        filename, _ = os.path.splitext(path_to_final)
        os.makedirs(filename)
        os.chmod(filename, 0o777)
        for k, file in enumerate(file_names):
            save(final[k], filename + '/' + os.path.basename(file), header[k])
        with zipfile.ZipFile(path_to_final, 'w') as zip:
            for file in file_names:
                zip.write(filename + '/' + os.path.basename(file), os.path.basename(file))
    else:
        imageSize = int(final.nbytes * 10e-7)
        bigtiff = True if imageSize > 2000 else False
        compress = 'zlib' if compress else None
        imwrite(path_to_final, final, bigtiff=bigtiff, compression=compress)

def color_to_gray(labelData):
    if len(labelData.shape) == 4 and labelData.shape[1] == 3:
        labelData = labelData.astype(np.float32)
        labelData -= np.amin(labelData)
        labelData /= np.amax(labelData)
        labelData = 0.299 * labelData[:,0] + 0.587 * labelData[:,1] + 0.114 * labelData[:,2]
        labelData *= 255.0
        labelData = labelData.astype(np.uint8)
        labelData = delbackground(labelData)
    elif len(labelData.shape) == 4 and labelData.shape[3] == 3:
        labelData = labelData.astype(np.float32)
        labelData -= np.amin(labelData)
        labelData /= np.amax(labelData)
        labelData = 0.299 * labelData[:,:,:,0] + 0.587 * labelData[:,:,:,1] + 0.114 * labelData[:,:,:,2]
        labelData *= 255.0
        labelData = labelData.astype(np.uint8)
        labelData = delbackground(labelData)
    return labelData

def delbackground(labels):
    allLabels, labelcounts = np.unique(labels, return_counts=True)
    index = np.argmax(labelcounts)
    labels[labels==allLabels[index]] = 0
    return labels
