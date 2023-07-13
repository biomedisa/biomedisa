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

try:
    import django
    django.setup()
    from django.shortcuts import get_object_or_404
    from biomedisa_app.models import Upload
    from biomedisa_app.config import config
    from redis import Redis
    from rq import Queue
except:
    from biomedisa_app.config_example import config

from biomedisa.settings import BASE_DIR, WWW_DATA_ROOT, PRIVATE_STORAGE_ROOT
from biomedisa_features.create_mesh import save_mesh, get_voxel_spacing
from biomedisa_features.amira_to_np.amira_helper import amira_to_np, np_to_amira
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
import numba
from shutil import copytree
from multiprocessing import Process
import re
import math

def Dice_score(ground_truth, result):
    dice = 2 * np.logical_and(ground_truth==result, (ground_truth+result)>0).sum() / \
    float((ground_truth>0).sum() + (result>0).sum())
    return dice

def ASSD(ground_truth, result):
    try:
        from biomedisa_features.assd import ASSD_one_label
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
            tmp = np.zeros(a.shape, dtype=np.uint8)
            tmp[a==k] = 1
            tmp = __resize__(tmp)
            data[tmp==1] = k
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

def unique_file_path(path, username, dir_path=PRIVATE_STORAGE_ROOT+'/'):

    # get extension
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
        i = int(suffix[1:-len(extension)])
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
    pic_path = 'images/%s/%s' %(username, filename)
    limit = 100 - len(addon) - len(suffix)
    path = dir_path + pic_path[:limit] + addon + suffix

    # check if file already exists
    file_already_exists = os.path.exists(path)
    while file_already_exists:
        limit = 100 - len(addon) - len('-') + len(str(i)) - len(extension)
        path = dir_path + pic_path[:limit] + addon + '-' + str(i) + extension
        file_already_exists = os.path.exists(path)
        i += 1

    return path

def id_generator(size, chars):
    return ''.join(random.choice(chars) for x in range(size))

def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

def load_data_(path_to_data, process):

    if os.path.isdir(path_to_data):
        path_to_data = path_to_data + '.zip'

    extension = os.path.splitext(path_to_data)[1]
    if extension == '.gz':
        extension = '.nii.gz'

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

    if config['SECURE_MODE']:
        if extension not in ['.am','.tif']:
            extension, header = '.tif', None
        if data is None:
            data = 'None'
        if header is None:
            header = 'None'
        for file, data_type in [(data, 'data'), (header, 'header'), (extension, 'extension')]:
            src = BASE_DIR + '/tmp/tmp.' + data_type + '_' + process + '.npy'
            dest = BASE_DIR + '/tmp/' + data_type + '_' + process + '.npy'
            np.save(src, file, allow_pickle=False)
            os.rename(src, dest)
    else:
        return data, header, extension

def load_data(path_to_data, process='None', return_extension=False):
    if config['SECURE_MODE']:
        q = Queue('load_data', connection=Redis())
        job = q.enqueue_call(load_data_, args=(path_to_data, process), timeout=-1)
        for k, data_type in enumerate(['data', 'header', 'extension']):
            file_path = BASE_DIR + '/tmp/' + data_type + '_' + process + '.npy'
            while not os.path.exists(file_path):
                time.sleep(1)
            if k == 0:
                data = np.load(file_path)
                if data.dtype == '<U4':
                    if str(data) == 'None':
                        data = None
            elif k == 1:
                header = np.load(file_path)
                if header.dtype == '<U4':
                    if str(header) == 'None':
                        header = None
            elif k == 2:
                extension = str(np.load(file_path))
                if extension == 'None':
                    extension = None
            os.remove(file_path)
    else:
        data, header, extension = load_data_(path_to_data, process)
    if return_extension:
        return data, header, extension
    else:
        return data, header

def _error_(bm, message):
    print('Error:', message)
    if bm.django_env:
        Upload.objects.create(user=bm.image.user, project=bm.image.project, log=1, imageType=None, shortfilename=message)
        bm.path_to_logfile = BASE_DIR + '/log/logfile.txt'
        with open(bm.path_to_logfile, 'a') as logfile:
            print('%s %s %s %s' %(time.ctime(), bm.image.user.username, bm.image.shortfilename, message), file=logfile)
        from biomedisa_app.views import send_error_message
        send_error_message(bm.image.user.username, bm.image.shortfilename, message)
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

    if len(bm.labelData.shape) != 3:
        return _error_(bm, 'Label must be three-dimensional.')

    if bm.data.shape != bm.labelData.shape:
        return _error_(bm, 'Image and label must have the same x,y,z-dimensions.')

    # get labels
    bm.allLabels = np.unique(bm.labelData)

    if bm.django_env and np.any(bm.allLabels > 255):
        return _error_(bm, 'No labels higher than 255 allowed.')

    if bm.django_env and np.any(bm.allLabels < 0):
        return _error_(bm, 'No negative labels allowed.')

    if np.any(bm.allLabels > 255):
        bm.labelData[bm.labelData > 255] = 0
        index = np.argwhere(bm.allLabels > 255)
        bm.allLabels = np.delete(bm.allLabels, index)
        print('Warning: Only labels 0-255 are allowed. Labels higher than 255 will be removed.')
    if np.any(bm.allLabels < 0):
        bm.labelData[bm.labelData < 0] = 0
        index = np.argwhere(bm.allLabels < 0)
        bm.allLabels = np.delete(bm.allLabels, index)
        print('Warning: Only labels 0-255 are allowed. Labels smaller than 0 will be removed.')
    bm.labelData = bm.labelData.astype(np.uint8)

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
    elif final_image_type in ['.hdr', '.mhd', '.mha', '.nrrd', '.nii', '.nii.gz']:
        final = np.swapaxes(final, 0, 2)
        save(final, path_to_final, header)
    elif final_image_type in ['.zip', 'directory', '']:
        header, file_names, final_dtype = header[0], header[1], header[2]
        final = final.astype(final_dtype)
        final = np.swapaxes(final, 2, 1)
        filename, _ = os.path.splitext(path_to_final)
        if not os.path.isdir(filename):
            os.makedirs(filename)
            os.chmod(filename, 0o777)
        for k, file in enumerate(file_names):
            save(final[k], filename + '/' + os.path.basename(file), header[k])
        if final_image_type == '.zip':
            with zipfile.ZipFile(path_to_final, 'w') as zip:
                for file in file_names:
                    zip.write(filename + '/' + os.path.basename(file), os.path.basename(file))
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

def convert_image(id):

    # get object
    img = get_object_or_404(Upload, pk=id)

    # set PID
    if img.status == 1:
        img.status = 2
        img.message = 'Processing'
    img.pid = int(os.getpid())
    img.save()

    # load data
    data, _ = load_data(img.pic.path.replace(WWW_DATA_ROOT, PRIVATE_STORAGE_ROOT), 'converter')

    if data is None:

        # return error
        message = 'Invalid data.'
        Upload.objects.create(user=img.user, project=img.project, log=1, imageType=None, shortfilename=message)

        # close process
        img.status = 0
        img.pid = 0
        img.save()

    else:

        # convert data
        data = img_to_uint8(data)

        # create pic path
        filename, extension = os.path.splitext(img.pic.path.replace(WWW_DATA_ROOT, PRIVATE_STORAGE_ROOT))
        if extension == '.gz':
            filename = filename[:-4]
        path_to_data = unique_file_path(filename+'.8bit.tif', img.user.username)
        new_short_name = os.path.basename(path_to_data)
        pic_path = 'images/%s/%s' %(img.user.username, new_short_name)

        # save data
        save_data(path_to_data, data, final_image_type='.tif')

        # copy slices for sliceviewer
        path_to_source = img.pic.path.replace(WWW_DATA_ROOT, PRIVATE_STORAGE_ROOT).replace('images', 'sliceviewer', 1)
        path_to_dest = path_to_data.replace('images', 'sliceviewer', 1)
        if os.path.exists(path_to_source):
            copytree(path_to_source, path_to_dest, copy_function=os.link)

        # create object
        active = 1 if img.imageType == 3 else 0
        Upload.objects.create(pic=pic_path, user=img.user, project=img.project,
            imageType=img.imageType, shortfilename=new_short_name, active=active)

        # close process
        img.status = 0
        img.pid = 0
        img.save()

def smooth_image(id):

    # get object
    img = get_object_or_404(Upload, pk=id)

    # set PID
    if img.status == 1:
        img.status = 2
        img.message = 'Processing'
    img.pid = int(os.getpid())
    img.save()

    # load data
    data, _ = load_data(img.pic.path.replace(WWW_DATA_ROOT, PRIVATE_STORAGE_ROOT), 'converter')

    if data is None:

        # return error
        message = 'Invalid data.'
        Upload.objects.create(user=img.user, project=img.project, log=1, imageType=None, shortfilename=message)

        # close process
        img.status = 0
        img.pid = 0
        img.save()

    else:

        # smooth image data
        out = smooth_img_3x3(data)

        # create pic path
        filename, extension = os.path.splitext(img.pic.path.replace(WWW_DATA_ROOT, PRIVATE_STORAGE_ROOT))
        if extension == '.gz':
            filename = filename[:-4]
        path_to_data = unique_file_path(filename+'.denoised.tif', img.user.username)
        new_short_name = os.path.basename(path_to_data)
        pic_path = 'images/%s/%s' %(img.user.username, new_short_name)

        # save data
        save_data(path_to_data, out, final_image_type='.tif')

        # create slices
        from biomedisa_features.create_slices import create_slices
        q = Queue('slices', connection=Redis())
        job = q.enqueue_call(create_slices, args=(path_to_data, None,), timeout=-1)

        # create object
        active = 1 if img.imageType == 3 else 0
        Upload.objects.create(pic=pic_path, user=img.user, project=img.project,
            imageType=img.imageType, shortfilename=new_short_name, active=active)

        # close process
        img.status = 0
        img.pid = 0
        img.save()

def convert_to_stl(id):

    # get object
    img = get_object_or_404(Upload, pk=id)

    # return error
    if img.imageType not in [2,3]:

        # return error
        message = 'No valid label data.'
        Upload.objects.create(user=img.user, project=img.project, log=1, imageType=None, shortfilename=message)

        # close process
        img.status = 0
        img.pid = 0
        img.save()

    else:

        # set PID
        if img.status == 1:
            img.status = 2
            img.message = 'Processing'
        img.pid = int(os.getpid())
        img.save()

        # load data
        data, header, extension = load_data(img.pic.path.replace(WWW_DATA_ROOT, PRIVATE_STORAGE_ROOT), process='converter', return_extension=True)

        if data is None:

            # return error
            message = 'Invalid data.'
            Upload.objects.create(user=img.user, project=img.project, log=1, imageType=None, shortfilename=message)

            # close process
            img.status = 0
            img.pid = 0
            img.save()

        else:

            # get voxel spacing
            xres, yres, zres = get_voxel_spacing(header, data, extension)
            print(f'Voxel spacing: x_spacing, y_spacing, z_spacing = {xres}, {yres}, {zres}')

            # create pic path
            filename, extension = os.path.splitext(img.pic.path.replace(WWW_DATA_ROOT, PRIVATE_STORAGE_ROOT))
            if extension == '.gz':
                filename = filename[:-4]
            path_to_data = unique_file_path(filename+'.stl', img.user.username)
            new_short_name = os.path.basename(path_to_data)
            pic_path = 'images/%s/%s' %(img.user.username, new_short_name)

            # create stl file
            save_mesh(path_to_data, data, xres, yres, zres)

            # create biomedisa object
            Upload.objects.create(pic=pic_path, user=img.user, project=img.project, imageType=5, shortfilename=new_short_name)

            # close process
            img.status = 0
            img.pid = 0
            img.save()

def _get_platform(bm):

    # import PyCUDA
    if bm.platform in ['cuda', None]:
        try:
            import pycuda.driver as cuda
            cuda.init()
            bm.available_devices = cuda.Device.count()
            bm.platform = 'cuda'
            return bm
        except:
            pass

    # import PyOpenCL
    try:
        import pyopencl as cl
    except ImportError:
        cl = None

    # select the first detected device
    if bm.platform is None and cl:
        for vendor in ['NVIDIA', 'Intel', 'AMD', 'Apple']:
            for dev, device_type in [('CPU',cl.device_type.CPU), ('GPU',cl.device_type.GPU)]:
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
    elif len(bm.platform.split('_')) == 3 and cl:
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

