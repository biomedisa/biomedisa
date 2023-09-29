#!/usr/bin/python3
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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import biomedisa
from biomedisa_features.biomedisa_helper import (_get_platform, smooth_img_3x3,
    pre_processing, _error_, read_labeled_slices, read_labeled_slices_allx,
    read_indices_allx, predict_blocksize)
from multiprocessing import freeze_support
import numpy as np
import argparse
import time

class Biomedisa(object):
     pass

class build_label(object):
     pass

def smart_interpolation(data, labelData, nbrw=10, sorw=4000, acwe=False, acwe_alpha=1.0, acwe_smooth=1, acwe_steps=3,
    path_to_data=None, path_to_labels=None, denoise=False, uncertainty=False, create_slices=False, platform=None,
    allaxis=False, ignore='none', only='all', clean=None, fill=None, smooth=0, no_compression=False):

    freeze_support()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:

        # create biomedisa
        bm = Biomedisa()
        bm.label = build_label()
        bm.process = 'biomedisa_interpolation'
        bm.django_env = False

        # time
        bm.TIC = time.time()

        # transfer arguments
        for arg in ['nbrw','sorw','allaxis','uncertainty','ignore','only','smooth','clean','fill']:
            bm.label.__dict__[arg] = locals()[arg]
        for arg in ['acwe_alpha','acwe_smooth','acwe','acwe_steps']:
            bm.label.__dict__[arg] = locals()[arg]
        for arg in ['data','labelData','path_to_data','path_to_labels','denoise','platform','create_slices']:
            bm.__dict__[arg] = locals()[arg]

        # compression
        if no_compression:
            bm.label.compression = False
        else:
            bm.label.compression = True

        # disable file saving when called as a function
        if bm.data is not None:
            bm.path_to_data = None
            bm.path_to_labels = None
            bm.create_slices = False

        # load and preprocess data
        bm = pre_processing(bm)

        # get platform
        bm = _get_platform(bm)

        # smooth, uncertainty and allx are not supported for opencl
        if bm.success and bm.platform.split('_')[0] == 'opencl':
            if bm.label.smooth:
                bm.label.smooth = 0
                print('Warning: Smoothing is not yet supported for opencl. Process starts without smoothing.')
            if bm.label.uncertainty:
                bm.label.uncertainty = False
                print('Warning: Uncertainty is not yet supported for opencl. Process starts without uncertainty.')
            if bm.label.allaxis:
                bm = _error_(bm, 'Allx is not yet supported for opencl.')

        if not bm.success:

            # send not executable
            for dest in range(1, size):
                comm.send(bm.success, dest=dest, tag=0)

        else:

            # create path_to_final
            if bm.path_to_data:
                filename, extension = os.path.splitext(os.path.basename(bm.path_to_data))
                if extension == '.gz':
                    filename = filename[:-4]
                filename = 'final.' + filename
                bm.path_to_final = bm.path_to_data.replace(os.path.basename(bm.path_to_data), filename + bm.final_image_type)

                # path to optional results
                filename, extension = os.path.splitext(bm.path_to_final)
                if extension == '.gz':
                    filename = filename[:-4]
                bm.path_to_smooth = filename + '.smooth' + bm.final_image_type
                bm.path_to_smooth_cleaned = filename + '.smooth.cleand' + bm.final_image_type
                bm.path_to_cleaned = filename + '.cleaned' + bm.final_image_type
                bm.path_to_filled = filename + '.filled' + bm.final_image_type
                bm.path_to_cleaned_filled = filename + '.cleaned.filled' + bm.final_image_type
                bm.path_to_acwe = filename + '.acwe' + bm.final_image_type
                bm.path_to_uq = filename + '.uncertainty.tif'

            # data type
            if bm.data.dtype == 'uint8':
                pass
            elif bm.data.dtype == 'int8':
                bm.data = bm.data.astype(np.int16)
                bm.data += 128
                bm.data = bm.data.astype(np.uint8)
            else:
                bm.data = bm.data.astype(float)
                bm.data -= np.amin(bm.data)
                bm.data /= np.amax(bm.data)
                bm.data *= 255.0
                bm.data = bm.data.astype(np.float32)

            # denoise image data
            if bm.denoise:
                bm.data = smooth_img_3x3(bm.data)

            # image size
            bm.imageSize = int(bm.data.nbytes * 10e-7)

            # add boundaries
            zsh, ysh, xsh = bm.data.shape
            tmp = np.zeros((1+zsh+1, 1+ysh+1, 1+xsh+1), dtype=bm.labelData.dtype)
            tmp[1:-1, 1:-1, 1:-1] = bm.labelData
            bm.labelData = tmp.copy(order='C')
            tmp = np.zeros((1+zsh+1, 1+ysh+1, 1+xsh+1), dtype=bm.data.dtype)
            tmp[1:-1, 1:-1, 1:-1] = bm.data
            bm.data = tmp.copy(order='C')
            bm.zsh, bm.ysh, bm.xsh = bm.data.shape

            # check if labeled slices are adjacent
            if bm.label.allaxis:
                bm.indices = []
                indices_tmp = read_indices_allx(bm.labelData, 0)
                bm.indices.extend(indices_tmp)
                tmp = np.swapaxes(bm.labelData, 0, 1)
                tmp = np.ascontiguousarray(tmp)
                indices_tmp = read_indices_allx(tmp, 1)
                bm.indices.extend(indices_tmp)
                tmp = np.swapaxes(tmp, 0, 2)
                tmp = np.ascontiguousarray(tmp)
                indices_tmp = read_indices_allx(tmp, 2)
                bm.indices.extend(indices_tmp)

                # labels must not be adjacent
                neighbours = False
                for k in range(3):
                    sub_indices = [x for (x, y) in bm.indices if y == k]
                    sub_indices_minus_one = [x - 1 for x in sub_indices]
                    if any(i in sub_indices for i in sub_indices_minus_one):
                        neighbours = True
                if neighbours:
                    bm = _error_(bm, 'At least one empty slice between labels required.')

            # send executable
            for dest in range(1, size):
                comm.send(bm.success, dest=dest, tag=0)

            if bm.success:

                # When is domain decomposition faster?
                bm = predict_blocksize(bm)
                nbytes = (bm.argmax_z - bm.argmin_z) * (bm.argmax_y - bm.argmin_y) * (bm.argmax_x - bm.argmin_x) * 4

                # small or large
                if nbytes * bm.nol < 1e10 and nbytes < 2e9:

                    # send "small" to childs
                    for dest in range(1, size):
                        comm.send(1, dest=dest, tag=1)

                    # reduce blocksize
                    bm.data = np.copy(bm.data[bm.argmin_z:bm.argmax_z, bm.argmin_y:bm.argmax_y, bm.argmin_x:bm.argmax_x], order='C')
                    bm.labelData = np.copy(bm.labelData[bm.argmin_z:bm.argmax_z, bm.argmin_y:bm.argmax_y, bm.argmin_x:bm.argmax_x], order='C')

                    # read labeled slices
                    if bm.label.allaxis:
                        bm.indices, bm.labels = [], []
                        indices_tmp, labels_tmp = read_labeled_slices_allx(bm.labelData, 0)
                        bm.indices.extend(indices_tmp)
                        bm.labels.append(labels_tmp)
                        tmp = np.swapaxes(bm.labelData, 0, 1)
                        tmp = np.ascontiguousarray(tmp)
                        indices_tmp, labels_tmp = read_labeled_slices_allx(tmp, 1)
                        bm.indices.extend(indices_tmp)
                        bm.labels.append(labels_tmp)
                        tmp = np.swapaxes(tmp, 0, 2)
                        tmp = np.ascontiguousarray(tmp)
                        indices_tmp, labels_tmp = read_labeled_slices_allx(tmp, 2)
                        bm.indices.extend(indices_tmp)
                        bm.labels.append(labels_tmp)
                    else:
                        bm.indices, bm.labels = read_labeled_slices(bm.labelData)

                    # number of ngpus
                    ngpus = min(len(bm.indices), size)

                    # send number of GPUs to childs
                    for dest in range(1, size):
                        comm.send(ngpus, dest=dest, tag=2)

                    # create subcommunicator
                    sub_comm = MPI.Comm.Split(comm, 1, rank)

                    from demo.biomedisa_small import _diffusion_child
                    results = _diffusion_child(sub_comm, bm)

                else:

                    # send "large" to childs
                    for dest in range(1, size):
                        comm.send(0, dest=dest, tag=1)

                    # number of ngpus
                    ngpus = min((bm.argmax_z - bm.argmin_z) // 100, size)
                    ngpus = max(ngpus, 1)

                    # send number of GPUs to childs
                    for dest in range(1, size):
                        comm.send(ngpus, dest=dest, tag=2)

                    # create subcommunicator
                    sub_comm = MPI.Comm.Split(comm, 1, rank)

                    from demo.biomedisa_large import _diffusion_child
                    results = _diffusion_child(sub_comm, bm)

                print('------------------------------------------------------------')
                print('Warning: This is deprecated and will be removed in the future')
                print('please use `biomedisa_features.biomedisa_interpolation` instead')
                print('------------------------------------------------------------')
                return results

    else:

        # check if executable
        executable = comm.recv(source=0, tag=0)

        if executable:

            # get small or large
            small = comm.recv(source=0, tag=1)

            # get number of gpus
            ngpus = comm.recv(source=0, tag=2)

            # create sub communicator
            if rank >= ngpus:
                sub_comm = MPI.Comm.Split(comm, 0, rank)     # set process to idle
            else:
                sub_comm = MPI.Comm.Split(comm, 1, rank)

                if small:
                    from demo.biomedisa_small import _diffusion_child
                    _diffusion_child(sub_comm)
                else:
                    from demo.biomedisa_large import _diffusion_child
                    _diffusion_child(sub_comm)

if __name__ == '__main__':

    # initialize arguments
    parser = argparse.ArgumentParser(description='Biomedisa interpolation.',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required arguments
    parser.add_argument('path_to_data', type=str, metavar='PATH_TO_IMAGE',
                        help='Location of image data')
    parser.add_argument('path_to_labels', type=str, metavar='PATH_TO_LABELS',
                        help='Location of label data')

    # optional arguments
    parser.add_argument('-v', '--version', action='version', version=f'{biomedisa.__version__}',
                        help='Biomedisa version')
    parser.add_argument('--nbrw', type=int, default=10,
                        help='Number of random walks starting at each pre-segmented pixel')
    parser.add_argument('--sorw', type=int, default=4000,
                        help='Steps of a random walk')
    parser.add_argument('--acwe', action='store_true', default=False,
                        help='Post-processing with active contour')
    parser.add_argument('--acwe_alpha', metavar='ALPHA', type=float, default=1.0,
                        help='Pushing force of active contour')
    parser.add_argument('--acwe_smooth', metavar='SMOOTH', type=int, default=1,
                        help='Smoothing of active contour')
    parser.add_argument('--acwe_steps', metavar='STEPS', type=int, default=3,
                        help='Iterations of active contour')
    parser.add_argument('-nc', '--no_compression', action='store_true', default=False,
                        help='Disable compression of segmentation results')
    parser.add_argument('-allx', '--allaxis', action='store_true', default=False,
                        help='If pre-segmentation is not exlusively in xy-plane')
    parser.add_argument('-d','--denoise', action='store_true', default=False,
                        help='Smooth/denoise image data before processing')
    parser.add_argument('-u','--uncertainty', action='store_true', default=False,
                        help='Return uncertainty of segmentation result')
    parser.add_argument('-cs','--create_slices', action='store_true', default=False,
                        help='Create slices of segmentation results')
    parser.add_argument('--ignore', type=str, default='none',
                        help='Ignore specific label(s), e.g. 2,5,6')
    parser.add_argument('--only', type=str, default='all',
                        help='Segment only specific label(s), e.g. 1,3,5')
    parser.add_argument('-s','--smooth', nargs='?', type=int, const=100, default=0,
                        help='Number of smoothing iterations for segmentation result')
    parser.add_argument('-c','--clean', nargs='?', type=float, const=0.1, default=None,
                        help='Remove outliers, e.g. 0.5 means that objects smaller than 50 percent of the size of the largest object will be removed')
    parser.add_argument('-f','--fill', nargs='?', type=float, const=0.9, default=None,
                        help='Fill holes, e.g. 0.5 means that all holes smaller than 50 percent of the entire label will be filled')
    parser.add_argument('-p','--platform', default=None,
                        help='One of "cuda", "opencl_NVIDIA_GPU", "opencl_Intel_CPU"')
    kwargs = vars(parser.parse_args())

    # run interpolation
    smart_interpolation(None, None, **kwargs)

