##########################################################################
##                                                                      ##
##  Copyright (c) 2022 Philipp Lösel. All rights reserved.              ##
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

import django
django.setup()
from django.shortcuts import get_object_or_404
from biomedisa_app.models import Upload
from biomedisa_app.config import config
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
import itk,vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from stl import mesh
import re

if config['OS'] == 'linux':
    from redis import Redis
    from rq import Queue

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

@numba.jit(nopython=True)
def smooth_img_3x3(img, out):
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

def img_to_uint8(img):
    if img.dtype != 'uint8':
        img = img.astype(np.float32)
        img -= np.amin(img)
        img /= np.amax(img)
        img *= 255.0
        img = img.astype(np.uint8)
    return img

def unique_file_path(path, username):

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
    for feature in ['.filled','.smooth','.acwe','.cleaned','.8bit','.refined',
                    '.uncertainty','.smooth.cleaned','.cleaned.filled','.denoised']:
        if filename[-len(feature):] == feature:
            addon = feature

    if addon:
        filename = filename[:-len(addon)]

    # maximum lenght of path
    dir_path = config['PATH_TO_BIOMEDISA'] + '/private_storage/'
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

    if config['SECURE_MODE'] and config['OS'] == 'linux':
        if extension not in ['.am','.tif']:
            extension, header = '.tif', None
        if data is None:
            data = 'None'
        if header is None:
            header = 'None'
        for file, data_type in [(data, 'data'), (header, 'header'), (extension, 'extension')]:
            src = config['PATH_TO_BIOMEDISA'] + '/tmp/tmp.' + data_type + '_' + process + '.npy'
            dest = config['PATH_TO_BIOMEDISA'] + '/tmp/' + data_type + '_' + process + '.npy'
            np.save(src, file, allow_pickle=False)
            os.rename(src, dest)
    else:
        return data, header, extension

def load_data(path_to_data, process='None', return_extension=False):
    if config['SECURE_MODE'] and config['OS'] == 'linux':
        q = Queue('load_data', connection=Redis())
        job = q.enqueue_call(load_data_, args=(path_to_data, process), timeout=-1)
        for k, data_type in enumerate(['data', 'header', 'extension']):
            file_path = config['PATH_TO_BIOMEDISA'] + '/tmp/' + data_type + '_' + process + '.npy'
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
    Upload.objects.create(user=bm.image.user, project=bm.image.project, log=1, imageType=None, shortfilename=message)
    bm.path_to_logfile = config['PATH_TO_BIOMEDISA'] + '/log/logfile.txt'
    with open(bm.path_to_logfile, 'a') as logfile:
        print('%s %s %s %s' %(time.ctime(), bm.image.user.username, bm.image.shortfilename, message), file=logfile)
    from biomedisa_app.views import send_error_message
    send_error_message(bm.image.user.username, bm.image.shortfilename, message)
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
    data, _ = load_data(img.pic.path, 'converter')

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
        filename, extension = os.path.splitext(img.pic.path)
        if extension == '.gz':
            filename = filename[:-4]
        path_to_data = unique_file_path(filename+'.8bit.tif', img.user.username)
        new_short_name = os.path.basename(path_to_data)
        pic_path = 'images/%s/%s' %(img.user.username, new_short_name)

        # save data
        save_data(path_to_data, data, final_image_type='.tif')

        # copy slices for sliceviewer
        path_to_source = img.pic.path.replace('images', 'sliceviewer', 1)
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
    data, _ = load_data(img.pic.path, 'converter')

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
        out = np.copy(data)
        out = smooth_img_3x3(data, out)

        # create pic path
        filename, extension = os.path.splitext(img.pic.path)
        if extension == '.gz':
            filename = filename[:-4]
        path_to_data = unique_file_path(filename+'.denoised.tif', img.user.username)
        new_short_name = os.path.basename(path_to_data)
        pic_path = 'images/%s/%s' %(img.user.username, new_short_name)

        # save data
        save_data(path_to_data, out, final_image_type='.tif')

        # create slices
        from biomedisa_features.create_slices import create_slices
        if config['OS'] == 'linux':
            q = Queue('slices', connection=Redis())
            job = q.enqueue_call(create_slices, args=(path_to_data, None,), timeout=-1)
        elif config['OS'] == 'windows':
            Process(target=create_slices, args=(path_to_data, None)).start()

        # create object
        active = 1 if img.imageType == 3 else 0
        Upload.objects.create(pic=pic_path, user=img.user, project=img.project,
            imageType=img.imageType, shortfilename=new_short_name, active=active)

        # close process
        img.status = 0
        img.pid = 0
        img.save()

def MarchingCubes(image,threshold):

    # marching cubes
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(image)
    mc.ComputeNormalsOn()
    mc.ComputeGradientsOn()
    mc.SetValue(0, threshold)
    mc.Update()

    # To remain largest region
    #confilter = vtk.vtkPolyDataConnectivityFilter()
    #confilter.SetInputData(mc.GetOutput())
    #confilter.SetExtractionModeToLargestRegion()
    #confilter.Update()

    # reduce poly data
    inputPoly = vtk.vtkPolyData()
    inputPoly.ShallowCopy(mc.GetOutput())

    print("Before decimation\n"
          "-----------------\n"
          "There are " + str(inputPoly.GetNumberOfPoints()) + "points.\n"
          "There are " + str(inputPoly.GetNumberOfPolys()) + "polygons.\n")

    #decimate = vtk.vtkDecimatePro()
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(inputPoly)
    decimate.SetTargetReduction(.90)
    decimate.Update()

    decimatedPoly = vtk.vtkPolyData()
    decimatedPoly.ShallowCopy(decimate.GetOutput())

    print("After decimation \n"
          "-----------------\n"
          "There are " + str(decimatedPoly.GetNumberOfPoints()) + "points.\n"
          "There are " + str(decimatedPoly.GetNumberOfPolys()) + "polygons.\n")

    # smooth surface
    smoothFilter = vtk.vtkSmoothPolyDataFilter()
    smoothFilter.SetInputData(decimatedPoly)
    smoothFilter.SetNumberOfIterations(15)
    smoothFilter.SetRelaxationFactor(0.1)
    smoothFilter.FeatureEdgeSmoothingOff()
    smoothFilter.BoundarySmoothingOn()
    smoothFilter.Update()

    decimatedPoly = vtk.vtkPolyData()
    decimatedPoly.ShallowCopy(smoothFilter.GetOutput())

    return decimatedPoly#confilter.GetOutput()

def CreateSTL(image,path_to_data):

    # get labels
    zsh,ysh,xsh=image.shape
    allLabels = np.unique(image)
    b = np.empty_like(image)
    arr = np.empty((0,3,3))
    nTotalCells = [0]

    for label in allLabels[1:]:

        # get label
        b.fill(0)
        b[image==label] = 1

        # numpy to vtk
        sc = numpy_to_vtk(num_array=b.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        imageData = vtk.vtkImageData()
        imageData.SetOrigin(0, 0, 0)
        imageData.SetSpacing(1, 1, 1)
        #imageData.SetDimensions(zsh, ysh, xsh)
        imageData.SetExtent(0,xsh-1,0,ysh-1,0,zsh-1)
        imageData.GetPointData().SetScalars(sc)

        # get poly data
        poly = MarchingCubes(imageData,1)

        # get number of cells
        nPoints = poly.GetNumberOfPoints()
        nPolys = poly.GetNumberOfPolys()
        if nPoints and nPolys:

            # get points
            points = poly.GetPoints()
            array = points.GetData()
            numpy_points = vtk_to_numpy(array)

            # get cells
            cells = poly.GetPolys()
            array = cells.GetData()
            numpy_cells = vtk_to_numpy(array)
            numpy_cells = numpy_cells.reshape(-1,4)

            # create mesh
            nCells, nCols = numpy_cells.shape
            tmp = np.empty((nCells,3,3))
            for k in range(nCells):
                for l in range(1,4):
                    id = numpy_cells[k,l]
                    tmp[k,l-1] = numpy_points[id]

            # append output
            arr = np.append(arr, tmp, axis=0)
            nTotalCells.append(nTotalCells[-1]+nCells)

    # save data as mesh
    data = np.zeros(nTotalCells[-1], dtype=mesh.Mesh.dtype)
    data['vectors'] = arr
    mesh_final = mesh.Mesh(data.copy())
    for i in range(len(nTotalCells)-1):
        start = nTotalCells[i]
        stop = nTotalCells[i+1]
        mesh_final.attr[start:stop,0] = i+1
    mesh_final.save(path_to_data)

def create_mesh(id):

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
        data, _ = load_data(img.pic.path, 'converter')

        if data is None:

            # return error
            message = 'Invalid data.'
            Upload.objects.create(user=img.user, project=img.project, log=1, imageType=None, shortfilename=message)

            # close process
            img.status = 0
            img.pid = 0
            img.save()

        else:

            # create pic path
            filename, extension = os.path.splitext(img.pic.path)
            if extension == '.gz':
                filename = filename[:-4]
            path_to_data = unique_file_path(filename+'.stl', img.user.username)
            new_short_name = os.path.basename(path_to_data)
            pic_path = 'images/%s/%s' %(img.user.username, new_short_name)

            # create stl file
            CreateSTL(data, path_to_data)

            # create biomedisa object
            tmp = Upload.objects.create(pic=pic_path, user=img.user, project=img.project, 
                imageType=5, shortfilename=new_short_name)

            # close process
            img.status = 0
            img.pid = 0
            img.save()
