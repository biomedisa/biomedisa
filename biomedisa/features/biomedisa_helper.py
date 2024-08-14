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
from biomedisa.features.amira_to_np.amira_helper import amira_to_np, np_to_amira
from biomedisa.features.nc_reader import nc_to_np, np_to_nc
from tifffile import imread, imwrite
from medpy.io import load, save
import SimpleITK as sitk
from PIL import Image
import numpy as np
import glob
import random
import cv2
import time
import zipfile
import numba
import subprocess
import re
import math
import tempfile

def silent_remove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

# create a unique filename
def unique_file_path(path, dir_path=biomedisa.BASE_DIR+'/private_storage/'):

    # get extension
    username = os.path.basename(os.path.dirname(path))
    filename = os.path.basename(path)
    filename, extension = os.path.splitext(filename)
    if extension == '.gz':
        filename, extension = os.path.splitext(filename)
        if extension == '.nii':
            extension = '.nii.gz'
        elif extension == '.tar':
            extension = '.tar.gz'

    # get suffix
    suffix = re.search("-[0-999]"+extension, path)
    if suffix:
        suffix = suffix.group()
        filename = os.path.basename(path)
        filename = filename[:-len(suffix)]
        i = int(suffix[1:-len(extension)]) + 1
    else:
        suffix = extension
        i = 1

    # get finaltype
    addon = ''
    for feature in ['.filled','.smooth','.acwe','.cleaned','.8bit','.refined', '.cropped',
                    '.uncertainty','.smooth.cleaned','.cleaned.filled','.denoised']:
        if filename[-len(feature):] == feature:
            addon = feature

    if addon:
        filename = filename[:-len(addon)]

    # maximum lenght of path
    pic_path = f'images/{username}/{filename}'
    limit = 100 - len(addon) - len(suffix)
    path = dir_path + pic_path[:limit] + addon + suffix

    # check if file already exists
    file_already_exists = os.path.exists(path)
    while file_already_exists:
        limit = 100 - len(addon) - len('-') - len(str(i)) - len(extension)
        path = dir_path + pic_path[:limit] + addon + '-' + str(i) + extension
        file_already_exists = os.path.exists(path)
        i += 1

    return path

def Dice_score(ground_truth, result, average_dice=False):
    if average_dice:
        dice = 0
        allLabels = np.unique(ground_truth)
        for l in allLabels[1:]:
            dice += 2 * np.logical_and(ground_truth==l, result==l).sum() / float((ground_truth==l).sum() + (result==l).sum())
        dice /= float(len(allLabels)-1)
    else:
        dice = 2 * np.logical_and(ground_truth==result, (ground_truth+result)>0).sum() / \
        float((ground_truth>0).sum() + (result>0).sum())
    return dice

def ASSD(ground_truth, result):
    try:
        from biomedisa.features.assd import ASSD_one_label
        number_of_elements = 0
        distances = 0
        hausdorff = 0
        for label in np.unique(ground_truth)[1:]:
            d, n, h = ASSD_one_label(ground_truth, result, label)
            number_of_elements += n
            distances += d
            hausdorff = max(h, hausdorff)
        assd = distances / float(number_of_elements)
        return assd, hausdorff
    except:
        print('Error: no CUDA device found. ASSD is not available.')
        return None, None

def img_resize(a, z_shape, y_shape, x_shape, interpolation=None, labels=False):
    if len(a.shape) > 3:
        zsh, ysh, xsh, csh = a.shape
    else:
        zsh, ysh, xsh = a.shape
    if interpolation == None:
        if z_shape < zsh or y_shape < ysh or x_shape < xsh:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC

    def __resize__(arr):
        b = np.empty((zsh, y_shape, x_shape), dtype=arr.dtype)
        for k in range(zsh):
            b[k] = cv2.resize(arr[k], (x_shape, y_shape), interpolation=interpolation)
        c = np.empty((y_shape, z_shape, x_shape), dtype=arr.dtype)
        b = np.swapaxes(b, 0, 1)
        b = np.copy(b, order='C')
        for k in range(y_shape):
            c[k] = cv2.resize(b[k], (x_shape, z_shape), interpolation=interpolation)
        c = np.swapaxes(c, 1, 0)
        c = np.copy(c, order='C')
        return c

    if labels:
        data = np.zeros((z_shape, y_shape, x_shape), dtype=a.dtype)
        for k in np.unique(a):
            if k!=0:
                tmp = np.zeros(a.shape, dtype=np.uint8)
                tmp[a==k] = 1
                tmp = __resize__(tmp)
                data[tmp==1] = k
    elif len(a.shape) > 3:
        data = np.empty((z_shape, y_shape, x_shape, csh), dtype=a.dtype)
        for channel in range(csh):
            data[:,:,:,channel] = __resize__(a[:,:,:,channel])
    else:
        data = __resize__(a)
    return data

@numba.jit(nopython=True)
def smooth_img_3x3(img):
    zsh, ysh, xsh = img.shape
    out = np.copy(img)
    for z in range(zsh):
        for y in range(ysh):
            for x in range(xsh):
                tmp,i = 0,0
                for k in range(-1,2):
                    for l in range(-1,2):
                        for m in range(-1,2):
                            if 0<=z+k<zsh and 0<=y+l<ysh and 0<=x+m<xsh:
                                tmp += img[z+k,y+l,x+m]
                                i += 1
                out[z,y,x] = tmp / i
    return out

def set_labels_to_zero(label, labels_to_compute, labels_to_remove):

    # compute only specific labels (set rest to zero)
    labels_to_compute = labels_to_compute.split(',')
    if not any([x in ['all', 'All', 'ALL'] for x in labels_to_compute]):
        allLabels = np.unique(label)
        labels_to_del = [k for k in allLabels if str(k) not in labels_to_compute and k > 0]
        for k in labels_to_del:
            label[label == k] = 0

    # ignore specific labels (set to zero)
    labels_to_remove = labels_to_remove.split(',')
    if not any([x in ['none', 'None', 'NONE'] for x in labels_to_remove]):
        for k in labels_to_remove:
            k = int(k)
            if np.any(label == k):
                label[label == k] = 0

    return label

def img_to_uint8(img):
    if img.dtype != 'uint8':
        img = img.astype(np.float32)
        img -= np.amin(img)
        img /= np.amax(img)
        img *= 255.0
        img = img.astype(np.uint8)
    return img

def id_generator(size, chars='abcdefghijklmnopqrstuvwxyz0123456789'):
    return ''.join(random.choice(chars) for x in range(size))

def rgb2gray(img, channel='last'):
    """Convert a RGB image to gray scale."""
    if channel=='last':
        out =  0.2989*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    elif channel=='first':
        out = 0.2989*img[0,:,:] + 0.587*img[1,:,:] + 0.114*img[2,:,:]
    out = out.astype(img.dtype)
    return out

def recursive_file_permissions(path_to_dir):
    files = glob.glob(path_to_dir+'/**/*', recursive=True) + [path_to_dir]
    for file in files:
        try:
            if os.path.isdir(file):
                os.chmod(file, 0o770)
            else:
                os.chmod(file, 0o660)
        except:
            pass

def load_data(path_to_data, process='None', return_extension=False):

    if not os.path.exists(path_to_data):
        print(f"Error: No such file or directory '{path_to_data}'")

    # get file extension
    extension = os.path.splitext(path_to_data)[1]
    if extension == '.gz':
        extension = '.nii.gz'
    elif extension == '.bz2':
        extension = os.path.splitext(os.path.splitext(path_to_data)[0])[1]

    if extension == '.am':
        try:
            data, header = amira_to_np(path_to_data)
            header = [header]
            if len(data) > 1:
                for arr in data[1:]:
                    header.append(arr)
            data = data[0]
        except Exception as e:
            print(e)
            data, header = None, None

    elif extension == '.nc':
        try:
            data, header = nc_to_np(path_to_data)
        except Exception as e:
            print(e)
            data, header = None, None

    elif extension in ['.hdr', '.mhd', '.mha', '.nrrd', '.nii', '.nii.gz']:
        try:
            header = sitk.ReadImage(path_to_data)
            data = sitk.GetArrayViewFromImage(header).copy()
        except Exception as e:
            print(e)
            data, header = None, None

    elif extension == '.zip' or os.path.isdir(path_to_data):
        with tempfile.TemporaryDirectory() as temp_dir:

            # extract files
            if extension=='.zip':
                try:
                    zip_ref = zipfile.ZipFile(path_to_data, 'r')
                    zip_ref.extractall(path=temp_dir)
                    zip_ref.close()
                except Exception as e:
                    print(e)
                    print('Using unzip package...')
                    try:
                        success = subprocess.Popen(['unzip',path_to_data,'-d',temp_dir]).wait()
                        if success != 0:
                            data, header = None, None
                    except Exception as e:
                        print(e)
                        data, header = None, None
                path_to_data = temp_dir

            # load files
            if os.path.isdir(path_to_data):
                files = []
                for data_type in ['.[pP][nN][gG]','.[tT][iI][fF]','.[tT][iI][fF][fF]','.[dD][cC][mM]','.[dD][iI][cC][oO][mM]','.[bB][mM][pP]','.[jJ][pP][gG]','.[jJ][pP][eE][gG]','.nc','.nc.bz2']:
                    files += [file for file in glob.glob(path_to_data+'/**/*'+data_type, recursive=True) if not os.path.basename(file).startswith('.')]
                nc_extension = False
                for file in files:
                    if os.path.splitext(file)[1] == '.nc' or os.path.splitext(os.path.splitext(file)[0])[1] == '.nc':
                        nc_extension = True
                if nc_extension:
                    try:
                        data, header = nc_to_np(path_to_data)
                    except Exception as e:
                        print(e)
                        data, header = None, None
                else:
                    try:
                        # load data slice by slice
                        file_names = []
                        img_slices = []
                        header = []
                        files.sort()
                        for file_name in files:
                            if os.path.isfile(file_name):
                                try:
                                    img, img_header = load(file_name)
                                    file_names.append(file_name)
                                    img_slices.append(img)
                                    header.append(img_header)
                                except:
                                    pass

                        # get data size
                        img = img_slices[0]
                        if len(img.shape)==3:
                            ysh, xsh, csh = img.shape[0], img.shape[1], img.shape[2]
                            channel = 'last'
                            if ysh < csh:
                                csh, ysh, xsh = img.shape[0], img.shape[1], img.shape[2]
                                channel = 'first'
                        else:
                            ysh, xsh = img.shape[0], img.shape[1]
                            csh, channel = 0, None

                        # create 3D volume
                        data = np.empty((len(file_names), ysh, xsh), dtype=img.dtype)
                        for k, img in enumerate(img_slices):
                            if csh==3:
                                img = rgb2gray(img, channel)
                            elif csh==1 and channel=='last':
                                img = img[:,:,0]
                            elif csh==1 and channel=='first':
                                img = img[0,:,:]
                            data[k] = img
                        header = [header, file_names, data.dtype]
                        data = np.swapaxes(data, 1, 2)
                        data = np.copy(data, order='C')
                    except Exception as e:
                        print(e)
                        data, header = None, None

    elif extension == '.mrc':
        try:
            import mrcfile
            with mrcfile.open(path_to_data, permissive=True) as mrc:
                data = mrc.data
            data = np.flip(data,1)
            extension, header = '.tif', None
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

    if return_extension:
        return data, header, extension
    else:
        return data, header

def _error_(bm, message):
    if bm.django_env:
        from biomedisa.features.django_env import create_error_object
        create_error_object(message, bm.remote, bm.queue, bm.img_id)
        with open(bm.path_to_logfile, 'a') as logfile:
            print('%s %s %s %s' %(time.ctime(), bm.username, bm.shortfilename, message), file=logfile)
    print('Error:', message)
    bm.success = False
    return bm

def pre_processing(bm):

    # load data
    if bm.data is None:
        bm.data, _ = load_data(bm.path_to_data, bm.process)

    # error handling
    if bm.data is None:
        return _error_(bm, 'Invalid image data.')

    # load label data
    if bm.labelData is None:
        bm.labelData, bm.header, bm.final_image_type = load_data(bm.path_to_labels, bm.process, True)

    # error handling
    if bm.labelData is None:
        return _error_(bm, 'Invalid label data.')

    # dimension errors
    if len(bm.labelData.shape) != 3:
        return _error_(bm, 'Label data must be three-dimensional.')
    if bm.data.shape != bm.labelData.shape:
        return _error_(bm, 'Image and label data must have the same x,y,z-dimensions.')

    # label data type
    if bm.labelData.dtype in ['float16','float32','float64']:
        if bm.django_env:
            return _error_(bm, 'Label data must be of integer type.')
        print(f'Warning: Potential label loss during conversion from {bm.labelData.dtype} to int32.')
        bm.labelData = bm.labelData.astype(np.int32)

    # get labels
    bm.allLabels = np.unique(bm.labelData)
    index = np.argwhere(bm.allLabels<0)
    bm.allLabels = np.delete(bm.allLabels, index)

    # labels greater than 255
    if np.any(bm.allLabels > 255):
        if bm.django_env:
            return _error_(bm, 'No labels greater than 255 allowed.')
        else:
            bm.labelData[bm.labelData > 255] = 0
            index = np.argwhere(bm.allLabels > 255)
            bm.allLabels = np.delete(bm.allLabels, index)
            print('Warning: Only labels <=255 are allowed. Labels greater than 255 will be removed.')

    # add background label if not existing
    if not np.any(bm.allLabels==0):
        bm.allLabels = np.append(0, bm.allLabels)

    # compute only specific labels
    labels_to_compute = (bm.only).split(',')
    if not any([x in ['all', 'All', 'ALL'] for x in labels_to_compute]):
        labels_to_remove = [k for k in bm.allLabels if str(k) not in labels_to_compute and k > 0]
        for k in labels_to_remove:
            bm.labelData[bm.labelData == k] = 0
            index = np.argwhere(bm.allLabels==k)
            bm.allLabels = np.delete(bm.allLabels, index)

    # ignore specific labels
    labels_to_remove = (bm.ignore).split(',')
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

    bm.success = True
    return bm

def save_data(path_to_final, final, header=None, final_image_type=None, compress=True):
    if final_image_type == None:
        final_image_type = os.path.splitext(path_to_final)[1]
        if final_image_type == '.gz':
            final_image_type = '.nii.gz'
    if final_image_type == '.am':
        final = [final]
        if len(header) > 1:
            for arr in header[1:]:
                final.append(arr)
        header = header[0]
        np_to_amira(path_to_final, final, header)
    elif final_image_type == '.nc':
        np_to_nc(path_to_final, final, header)
    elif final_image_type in ['.hdr', '.mhd', '.mha', '.nrrd', '.nii', '.nii.gz']:
        simg = sitk.GetImageFromArray(final)
        if header is not None:
            simg.CopyInformation(header)
        sitk.WriteImage(simg, path_to_final, useCompression=compress)
    elif final_image_type in ['.zip', 'directory', '']:
        with tempfile.TemporaryDirectory() as temp_dir:
            # make results directory
            if final_image_type == '.zip':
                results_dir = temp_dir
            else:
                results_dir = path_to_final
                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)
                    os.chmod(results_dir, 0o770)
            # save data as NC blocks
            if os.path.splitext(header[1][0])[1] == '.nc':
                np_to_nc(results_dir, final, header)
                file_names = header[1]
            # save data as PNG, TIF, DICOM slices
            else:
                header, file_names, final_dtype = header[0], header[1], header[2]
                final = final.astype(final_dtype)
                final = np.swapaxes(final, 2, 1)
                for k, file in enumerate(file_names):
                    save(final[k], results_dir + '/' + os.path.basename(file), header[k])
            # zip data
            if final_image_type == '.zip':
                with zipfile.ZipFile(path_to_final, 'w') as zip:
                    for file in file_names:
                        zip.write(results_dir + '/' + os.path.basename(file), os.path.basename(file))
    else:
        imageSize = int(final.nbytes * 10e-7)
        bigtiff = True if imageSize > 2000 else False
        try:
            compress = 'zlib' if compress else None
            imwrite(path_to_final, final, bigtiff=bigtiff, compression=compress)
        except:
            compress = 6 if compress else 0
            imwrite(path_to_final, final, bigtiff=bigtiff, compress=compress)

def color_to_gray(labelData):
    if len(labelData.shape) == 4 and labelData.shape[1] == 3:
        labelData = labelData.astype(np.float16)
        labelData -= np.amin(labelData)
        labelData /= np.amax(labelData)
        labelData = 0.299 * labelData[:,0] + 0.587 * labelData[:,1] + 0.114 * labelData[:,2]
        labelData *= 255.0
        labelData = labelData.astype(np.uint8)
        labelData = delbackground(labelData)
    elif len(labelData.shape) == 4 and labelData.shape[3] == 3:
        labelData = labelData.astype(np.float16)
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

def _get_platform(bm):

    # import PyCUDA
    if bm.platform in ['cuda', None]:
        try:
            import pycuda.driver as cuda
            import pycuda.gpuarray as gpuarray
            cuda.init()
            bm.available_devices = cuda.Device.count()
            if bm.available_devices > 0:
                dev = cuda.Device(0)
                ctx = dev.make_context()
                a_gpu = gpuarray.to_gpu(np.random.randn(4,4).astype(np.float32))
                a_doubled = (2*a_gpu).get()
                ctx.pop()
                del ctx
                bm.platform = 'cuda'
                return bm
            elif bm.platform == 'cuda':
                print('Error: No CUDA device found.')
                bm.success = False
                return bm
        except:
            pass

    # import PyOpenCL
    try:
        import pyopencl as cl
    except ImportError:
        cl = None

    # select the first detected device
    if cl and bm.platform is None:
        for vendor in ['NVIDIA', 'Intel', 'AMD', 'Apple']:
            for dev, device_type in [('GPU',cl.device_type.GPU),('CPU',cl.device_type.CPU)]:
                all_platforms = cl.get_platforms()
                my_devices = []
                for p in all_platforms:
                    if p.get_devices(device_type=device_type) and vendor in p.name:
                        my_devices = p.get_devices(device_type=device_type)
                if my_devices:
                    bm.platform = 'opencl_'+vendor+'_'+dev
                    if 'OMPI_COMMAND' in os.environ and dev == 'CPU':
                        print("Error: OpenCL CPU does not support MPI. Start Biomedisa without 'mpirun' or 'mpiexec'.")
                        bm.success = False
                        return bm
                    else:
                        bm.available_devices = len(my_devices)
                        print('Detected platform:', bm.platform)
                        print('Detected devices:', my_devices)
                        return bm

    # explicitly select the OpenCL device
    elif cl and len(bm.platform.split('_')) == 3:
        plat, vendor, dev = bm.platform.split('_')
        device_type=cl.device_type.GPU if dev=='GPU' else cl.device_type.CPU
        all_platforms = cl.get_platforms()
        my_devices = []
        for p in all_platforms:
            if p.get_devices(device_type=device_type) and vendor in p.name:
                my_devices = p.get_devices(device_type=device_type)
        if my_devices:
            if 'OMPI_COMMAND' in os.environ and dev == 'CPU':
                print("Error: OpenCL CPU does not support MPI. Start Biomedisa without 'mpirun' or 'mpiexec'.")
                bm.success = False
                return bm
            else:
                bm.available_devices = len(my_devices)
                print('Detected platform:', bm.platform)
                print('Detected devices:', my_devices)
                return bm

    # stop the process if no device is detected
    if bm.platform is None:
        bm.platform = 'OpenCL or CUDA'
    print(f'Error: No {bm.platform} device found.')
    bm.success = False
    return bm

def _get_device(platform, dev_id):
    import pyopencl as cl
    plat, vendor, dev = platform.split('_')
    device_type=cl.device_type.GPU if dev=='GPU' else cl.device_type.CPU
    all_platforms = cl.get_platforms()
    for p in all_platforms:
        if p.get_devices(device_type=device_type) and vendor in p.name:
            my_devices = p.get_devices(device_type=device_type)
    context = cl.Context(devices=my_devices)
    queue = cl.CommandQueue(context, my_devices[dev_id % len(my_devices)])
    return context, queue

def read_labeled_slices(arr):
    data = np.zeros((0, arr.shape[1], arr.shape[2]), dtype=np.int32)
    indices = []
    for k, slc in enumerate(arr[:]):
        if np.any(slc):
            data = np.append(data, [arr[k]], axis=0)
            indices.append(k)
    return indices, data

def read_labeled_slices_allx(arr, ax):
    gradient = np.zeros(arr.shape, dtype=np.int8)
    ones = np.zeros_like(gradient)
    ones[arr != 0] = 1
    tmp = ones[:,:-1] - ones[:,1:]
    tmp = np.abs(tmp)
    gradient[:,:-1] += tmp
    gradient[:,1:] += tmp
    ones[gradient == 2] = 0
    gradient.fill(0)
    tmp = ones[:,:,:-1] - ones[:,:,1:]
    tmp = np.abs(tmp)
    gradient[:,:,:-1] += tmp
    gradient[:,:,1:] += tmp
    ones[gradient == 2] = 0
    indices = []
    data = np.zeros((0, arr.shape[1], arr.shape[2]), dtype=np.int32)
    for k, slc in enumerate(ones[:]):
        if np.any(slc):
            data = np.append(data, [arr[k]], axis=0)
            indices.append((k, ax))
    return indices, data

def read_indices_allx(arr, ax):
    gradient = np.zeros(arr.shape, dtype=np.int8)
    ones = np.zeros_like(gradient)
    ones[arr != 0] = 1
    tmp = ones[:,:-1] - ones[:,1:]
    tmp = np.abs(tmp)
    gradient[:,:-1] += tmp
    gradient[:,1:] += tmp
    ones[gradient == 2] = 0
    gradient.fill(0)
    tmp = ones[:,:,:-1] - ones[:,:,1:]
    tmp = np.abs(tmp)
    gradient[:,:,:-1] += tmp
    gradient[:,:,1:] += tmp
    ones[gradient == 2] = 0
    indices = []
    for k, slc in enumerate(ones[:]):
        if np.any(slc):
            indices.append((k, ax))
    return indices

def read_labeled_slices_large(arr):
    data = np.zeros((0, arr.shape[1], arr.shape[2]), dtype=np.int32)
    indices = []
    i = 0
    while i < arr.shape[0]:
        slc = arr[i]
        if np.any(slc):
            data = np.append(data, [arr[i]], axis=0)
            indices.append(i)
            i += 5
        else:
            i += 1
    return indices, data

def read_labeled_slices_allx_large(arr):
    gradient = np.zeros(arr.shape, dtype=np.int8)
    ones = np.zeros_like(gradient)
    ones[arr > 0] = 1
    tmp = ones[:,:-1] - ones[:,1:]
    tmp = np.abs(tmp)
    gradient[:,:-1] += tmp
    gradient[:,1:] += tmp
    ones[gradient == 2] = 0
    gradient.fill(0)
    tmp = ones[:,:,:-1] - ones[:,:,1:]
    tmp = np.abs(tmp)
    gradient[:,:,:-1] += tmp
    gradient[:,:,1:] += tmp
    ones[gradient == 2] = 0
    indices = []
    data = np.zeros((0, arr.shape[1], arr.shape[2]), dtype=np.int32)
    for k, slc in enumerate(ones[:]):
        if np.any(slc):
            data = np.append(data, [arr[k]], axis=0)
            indices.append(k)
    return indices, data

def predict_blocksize(bm):
    zsh, ysh, xsh = bm.labelData.shape
    argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = zsh, 0, ysh, 0, xsh, 0
    for k in range(zsh):
        y, x = np.nonzero(bm.labelData[k])
        if x.any():
            argmin_x = min(argmin_x, np.amin(x))
            argmax_x = max(argmax_x, np.amax(x))
            argmin_y = min(argmin_y, np.amin(y))
            argmax_y = max(argmax_y, np.amax(y))
            argmin_z = min(argmin_z, k)
            argmax_z = max(argmax_z, k)
    zmin, zmax = argmin_z, argmax_z
    bm.argmin_x = argmin_x - 100 if argmin_x - 100 > 0 else 0
    bm.argmax_x = argmax_x + 100 if argmax_x + 100 < xsh else xsh
    bm.argmin_y = argmin_y - 100 if argmin_y - 100 > 0 else 0
    bm.argmax_y = argmax_y + 100 if argmax_y + 100 < ysh else ysh
    bm.argmin_z = argmin_z - 100 if argmin_z - 100 > 0 else 0
    bm.argmax_z = argmax_z + 100 if argmax_z + 100 < zsh else zsh
    return bm

def splitlargedata(data):
    dataMemory = data.nbytes
    dataListe = []
    if dataMemory > 1500000000:
        mod = dataMemory / float(1500000000)
        mod2 = int(math.ceil(mod))
        mod3 = divmod(data.shape[0], mod2)[0]
        for k in range(mod2):
            dataListe.append(data[mod3*k:mod3*(k+1)])
        dataListe.append(data[mod3*mod2:])
    else:
        dataListe.append(data)
    return dataListe

def sendToChildLarge(comm, indices, dest, dataListe, Labels, nbrw, sorw, blocks,
                allx, allLabels, smooth, uncertainty, platform):
    from mpi4py import MPI
    comm.send(len(dataListe), dest=dest, tag=0)
    for k, tmp in enumerate(dataListe):
        tmp = tmp.copy(order='C')
        comm.send([tmp.shape[0], tmp.shape[1], tmp.shape[2], tmp.dtype], dest=dest, tag=10+(2*k))
        if tmp.dtype == 'uint8':
            comm.Send([tmp, MPI.BYTE], dest=dest, tag=10+(2*k+1))
        else:
            comm.Send([tmp, MPI.FLOAT], dest=dest, tag=10+(2*k+1))

    comm.send([nbrw, sorw, allx, smooth, uncertainty, platform], dest=dest, tag=1)

    if allx:
        for k in range(3):
            labelsListe = splitlargedata(Labels[k])
            comm.send(len(labelsListe), dest=dest, tag=2+k)
            for l, tmp in enumerate(labelsListe):
                tmp = tmp.copy(order='C')
                comm.send([tmp.shape[0], tmp.shape[1], tmp.shape[2]], dest=dest, tag=100+(2*k))
                comm.Send([tmp, MPI.INT], dest=dest, tag=100+(2*k+1))
    else:
        labelsListe = splitlargedata(Labels)
        comm.send(len(labelsListe), dest=dest, tag=2)
        for k, tmp in enumerate(labelsListe):
            tmp = tmp.copy(order='C')
            comm.send([tmp.shape[0], tmp.shape[1], tmp.shape[2]], dest=dest, tag=100+(2*k))
            comm.Send([tmp, MPI.INT], dest=dest, tag=100+(2*k+1))

    comm.send(allLabels, dest=dest, tag=99)
    comm.send(indices, dest=dest, tag=8)
    comm.send(blocks, dest=dest, tag=9)

def sendToChild(comm, indices, indices_child, dest, data, Labels, nbrw, sorw, allx, platform):
    from mpi4py import MPI
    data = data.copy(order='C')
    comm.send([data.shape[0], data.shape[1], data.shape[2], data.dtype], dest=dest, tag=0)
    if data.dtype == 'uint8':
        comm.Send([data, MPI.BYTE], dest=dest, tag=1)
    else:
        comm.Send([data, MPI.FLOAT], dest=dest, tag=1)
    comm.send([allx, nbrw, sorw, platform], dest=dest, tag=2)
    if allx:
        for k in range(3):
            labels = Labels[k].copy(order='C')
            comm.send([labels.shape[0], labels.shape[1], labels.shape[2]], dest=dest, tag=k+3)
            comm.Send([labels, MPI.INT], dest=dest, tag=k+6)
    else:
        labels = Labels.copy(order='C')
        comm.send([labels.shape[0], labels.shape[1], labels.shape[2]], dest=dest, tag=3)
        comm.Send([labels, MPI.INT], dest=dest, tag=6)
    comm.send(indices, dest=dest, tag=9)
    comm.send(indices_child, dest=dest, tag=10)

def _split_indices(indices, ngpus):
    ngpus = ngpus if ngpus < len(indices) else len(indices)
    nindices = len(indices)
    parts = []
    for i in range(0, ngpus):
        slice_idx = indices[i]
        parts.append([slice_idx])
    if ngpus < nindices:
        for i in range(ngpus, nindices):
            gid = i % ngpus
            slice_idx = indices[i]
            parts[gid].append(slice_idx)
    return parts

def get_labels(pre_final, labels):
    numos = np.unique(pre_final)
    final = np.zeros_like(pre_final)
    for k in numos[1:]:
        final[pre_final == k] = labels[k]
    return final

