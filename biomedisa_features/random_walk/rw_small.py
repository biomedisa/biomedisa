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

import django
django.setup()
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404

from biomedisa_app.models import Upload, Profile
from biomedisa_app.views import send_notification
from biomedisa_features.biomedisa_helper import save_data, unique_file_path
from biomedisa_features.active_contour import active_contour
from biomedisa_features.remove_outlier import remove_outlier
from biomedisa_features.create_slices import create_slices
from biomedisa_app.config import config
from multiprocessing import Process
from gpu_kernels import (_build_kernel_uncertainty, _build_kernel_max,
                         _build_update_gpu, _build_curvature_gpu)

from mpi4py import MPI
import os, sys
import numpy as np
import time
import socket
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

if config['OS'] == 'linux':
    from redis import Redis
    from rq import Queue

def sendToChild(comm, indices, indices_child, dest, data, Labels, nbrw, sorw, allx):
    data = data.copy(order='C')
    comm.send([data.shape[0], data.shape[1], data.shape[2], data.dtype], dest=dest, tag=0)
    if data.dtype == 'uint8':
        comm.Send([data, MPI.BYTE], dest=dest, tag=1)
    else:
        comm.Send([data, MPI.FLOAT], dest=dest, tag=1)
    comm.send([allx, nbrw, sorw], dest=dest, tag=2)
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

def _diffusion_child(comm, bm=None):

    rank = comm.Get_rank()
    ngpus = comm.Get_size()

    nodename = socket.gethostname()
    name = '%s %s' %(nodename, rank)
    print(name)

    if rank == 0:

        # split indices on GPUs
        indices_split = _split_indices(bm.indices, ngpus)
        print('Indices:', indices_split)

        # send data to GPUs
        for k in range(1, ngpus):
            sendToChild(comm, bm.indices, indices_split[k], k, bm.data, bm.labels, bm.label.nbrw, bm.label.sorw, bm.label.allaxis)

        # init cuda device
        cuda.init()
        dev = cuda.Device(rank)
        ctx = dev.make_context()

        # select the desired script
        if bm.label.allaxis:
            from pycuda_small_allx import walk
        else:
            from pycuda_small import walk

        # run random walks
        tic = time.time()
        walkmap = walk(bm.data, bm.labels, bm.indices, indices_split[0], bm.label.nbrw, bm.label.sorw, name)
        tac = time.time()
        print('Walktime_%s: ' %(name) + str(int(tac - tic)) + ' ' + 'seconds')

        # gather data
        zsh_tmp = bm.argmax_z - bm.argmin_z
        ysh_tmp = bm.argmax_y - bm.argmin_y
        xsh_tmp = bm.argmax_x - bm.argmin_x
        if ngpus > 1:
            final_zero = np.empty((bm.nol, zsh_tmp, ysh_tmp, xsh_tmp), dtype=np.float32)
            for k in range(bm.nol):
                sendbuf = np.copy(walkmap[k])
                recvbuf = np.empty((zsh_tmp, ysh_tmp, xsh_tmp), dtype=np.float32)
                comm.Barrier()
                comm.Reduce([sendbuf, MPI.FLOAT], [recvbuf, MPI.FLOAT], root=0, op=MPI.SUM)
                final_zero[k] = recvbuf
        else:
            final_zero = walkmap

        # block and grid size
        block = (32, 32, 1)
        x_grid = (xsh_tmp // 32) + 1
        y_grid = (ysh_tmp // 32) + 1
        grid = (int(x_grid), int(y_grid), int(zsh_tmp))
        xsh_gpu = np.int32(xsh_tmp)
        ysh_gpu = np.int32(ysh_tmp)

        # smooth
        if bm.label.smooth:
            try:
                update_gpu = _build_update_gpu()
                curvature_gpu = _build_curvature_gpu()
                a_gpu = gpuarray.empty((zsh_tmp, ysh_tmp, xsh_tmp), dtype=np.float32)
                b_gpu = gpuarray.zeros((zsh_tmp, ysh_tmp, xsh_tmp), dtype=np.float32)
            except Exception as e:
                print('Warning: GPU out of memory to allocate smooth array. Process starts without smoothing.')
                bm.label.smooth = 0

        if bm.label.smooth:
            final_smooth = np.copy(final_zero)
            for k in range(bm.nol):
                a_gpu = gpuarray.to_gpu(final_smooth[k])
                for l in range(bm.label.smooth):
                    curvature_gpu(a_gpu, b_gpu, xsh_gpu, ysh_gpu, block=block, grid=grid)
                    update_gpu(a_gpu, b_gpu, xsh_gpu, ysh_gpu, block=block, grid=grid)
                final_smooth[k] = a_gpu.get()
            final_smooth = np.argmax(final_smooth, axis=0).astype(np.uint8)
            final_smooth = get_labels(final_smooth, bm.allLabels)
            final = np.zeros((bm.zsh, bm.ysh, bm.xsh), dtype=np.uint8)
            final[bm.argmin_z:bm.argmax_z, bm.argmin_y:bm.argmax_y, bm.argmin_x:bm.argmax_x] = final_smooth
            final = final[1:-1, 1:-1, 1:-1]
            bm.path_to_smooth = unique_file_path(bm.path_to_smooth, bm.image.user.username)
            save_data(bm.path_to_smooth, final, bm.header, bm.final_image_type, bm.label.compression)

        # uncertainty
        if bm.label.uncertainty:
            try:
                max_gpu = gpuarray.zeros((3, zsh_tmp, ysh_tmp, xsh_tmp), dtype=np.float32)
                a_gpu = gpuarray.zeros((zsh_tmp, ysh_tmp, xsh_tmp), dtype=np.float32)
                kernel_uncertainty = _build_kernel_uncertainty()
                kernel_max = _build_kernel_max()
                for k in range(bm.nol):
                    a_gpu = gpuarray.to_gpu(final_zero[k])
                    kernel_max(max_gpu, a_gpu, xsh_gpu, ysh_gpu, block=block, grid=grid)
                kernel_uncertainty(max_gpu, a_gpu, xsh_gpu, ysh_gpu, block=block, grid=grid)
                uq = a_gpu.get()
                uq *= 255
                uq = uq.astype(np.uint8)
                final = np.zeros((bm.zsh, bm.ysh, bm.xsh), dtype=np.uint8)
                final[bm.argmin_z:bm.argmax_z, bm.argmin_y:bm.argmax_y, bm.argmin_x:bm.argmax_x] = uq
                final = final[1:-1, 1:-1, 1:-1]
                bm.path_to_uq = unique_file_path(bm.path_to_uq, bm.image.user.username)
                save_data(bm.path_to_uq, final, compress=bm.label.compression)
            except Exception as e:
                print('Warning: GPU out of memory to allocate uncertainty array. Process starts without uncertainty.')
                bm.label.uncertainty = False

        # free device
        ctx.pop()
        del ctx

        # argmax
        final_zero = np.argmax(final_zero, axis=0).astype(np.uint8)

        # save finals
        final_zero = get_labels(final_zero, bm.allLabels)
        final = np.zeros((bm.zsh, bm.ysh, bm.xsh), dtype=np.uint8)
        final[bm.argmin_z:bm.argmax_z, bm.argmin_y:bm.argmax_y, bm.argmin_x:bm.argmax_x] = final_zero
        final = final[1:-1, 1:-1, 1:-1]
        bm.path_to_final = unique_file_path(bm.path_to_final, bm.image.user.username)
        save_data(bm.path_to_final, final, bm.header, bm.final_image_type, bm.label.compression)

        # create final objects
        shortfilename = os.path.basename(bm.path_to_final)
        filename = 'images/' + bm.image.user.username + '/' + shortfilename
        tmp = Upload.objects.create(pic=filename, user=bm.image.user, project=bm.image.project, final=1, active=1, imageType=3, shortfilename=shortfilename)
        tmp.friend = tmp.id
        tmp.save()
        if bm.label.uncertainty:
            shortfilename = os.path.basename(bm.path_to_uq)
            filename = 'images/' + bm.image.user.username + '/' + shortfilename
            Upload.objects.create(pic=filename, user=bm.image.user, project=bm.image.project, final=4, imageType=3, shortfilename=shortfilename, friend=tmp.id)
        if bm.label.smooth:
            shortfilename = os.path.basename(bm.path_to_smooth)
            filename = 'images/' + bm.image.user.username + '/' + shortfilename
            smooth = Upload.objects.create(pic=filename, user=bm.image.user, project=bm.image.project, final=5, imageType=3, shortfilename=shortfilename, friend=tmp.id)

        # write in logs
        t = int(time.time() - bm.TIC)
        if t < 60:
            time_str = str(t) + ' sec'
        elif 60 <= t < 3600:
            time_str = str(t // 60) + ' min ' + str(t % 60) + ' sec'
        elif 3600 < t:
            time_str = str(t // 3600) + ' h ' + str((t % 3600) // 60) + ' min ' + str(t % 60) + ' sec'
        with open(bm.path_to_time, 'a') as timefile:
            print('%s %s %s %s MB %s on %s' %(time.ctime(), bm.image.user.username, bm.image.shortfilename, bm.imageSize, time_str, config['SERVER_ALIAS']), file=timefile)
        print('Total calculation time:', time_str)

        # send notification
        send_notification(bm.image.user.username, bm.image.shortfilename, time_str, config['SERVER_ALIAS'])

        # start subprocesses
        if config['OS'] == 'linux':
            # acwe
            q = Queue('acwe', connection=Redis())
            job = q.enqueue_call(active_contour, args=(bm.image.id, tmp.id, bm.label.id,), timeout=-1)

            # cleanup
            q = Queue('cleanup', connection=Redis())
            job = q.enqueue_call(remove_outlier, args=(bm.image.id, tmp.id, tmp.id, bm.label.id,), timeout=-1)
            if bm.label.smooth:
                job = q.enqueue_call(remove_outlier, args=(bm.image.id, smooth.id, tmp.id, bm.label.id, False,), timeout=-1)

            # create slices
            q = Queue('slices', connection=Redis())
            job = q.enqueue_call(create_slices, args=(bm.path_to_data, bm.path_to_final,), timeout=-1)
            if bm.label.smooth:
                job = q.enqueue_call(create_slices, args=(bm.path_to_data, bm.path_to_smooth,), timeout=-1)
            if bm.label.uncertainty:
                job = q.enqueue_call(create_slices, args=(bm.path_to_uq, None,), timeout=-1)

        elif config['OS'] == 'windows':

            # acwe
            Process(target=active_contour, args=(bm.image.id, tmp.id, bm.label.id)).start()

            # cleanup
            Process(target=remove_outlier, args=(bm.image.id, tmp.id, tmp.id, bm.label.id)).start()
            if bm.label.smooth:
                Process(target=remove_outlier, args=(bm.image.id, smooth.id, tmp.id, bm.label.id, False)).start()

            # create slices
            Process(target=create_slices, args=(bm.path_to_data, bm.path_to_final)).start()
            if bm.label.smooth:
                Process(target=create_slices, args=(bm.path_to_data, bm.path_to_smooth)).start()
            if bm.label.uncertainty:
                Process(target=create_slices, args=(bm.path_to_uq, None)).start()

    else:

        data_z, data_y, data_x, data_dtype = comm.recv(source=0, tag=0)
        data = np.empty((data_z, data_y, data_x), dtype=data_dtype)
        if data_dtype == 'uint8':
            comm.Recv([data, MPI.BYTE], source=0, tag=1)
        else:
            comm.Recv([data, MPI.FLOAT], source=0, tag=1)
        allx, nbrw, sorw = comm.recv(source=0, tag=2)
        if allx:
            labels = []
            for k in range(3):
                labels_z, labels_y, labels_x = comm.recv(source=0, tag=k+3)
                labels_tmp = np.empty((labels_z, labels_y, labels_x), dtype=np.int32)
                comm.Recv([labels_tmp, MPI.INT], source=0, tag=k+6)
                labels.append(labels_tmp)
        else:
            labels_z, labels_y, labels_x = comm.recv(source=0, tag=3)
            labels = np.empty((labels_z, labels_y, labels_x), dtype=np.int32)
            comm.Recv([labels, MPI.INT], source=0, tag=6)
        indices = comm.recv(source=0, tag=9)
        indices_child = comm.recv(source=0, tag=10)

        # init cuda device
        cuda.init()
        dev = cuda.Device(rank)
        ctx = dev.make_context()

        # select the desired script
        if allx:
            from pycuda_small_allx import walk
        else:
            from pycuda_small import walk

        # run random walks
        tic = time.time()
        walkmap = walk(data, labels, indices, indices_child, nbrw, sorw, name)
        tac = time.time()
        print('Walktime_%s: ' %(name) + str(int(tac - tic)) + ' ' + 'seconds')

        # free device
        ctx.pop()
        del ctx

        # send data
        for k in range(walkmap.shape[0]):
            datatemporaer = np.copy(walkmap[k])
            comm.Barrier()
            comm.Reduce([datatemporaer, MPI.FLOAT], None, root=0, op=MPI.SUM)
