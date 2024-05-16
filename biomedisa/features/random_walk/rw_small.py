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

from biomedisa.features.remove_outlier import clean, fill
from biomedisa.features.active_contour import activeContour
from biomedisa.features.biomedisa_helper import (_get_device, save_data, unique_file_path,
    sendToChild, _split_indices, get_labels, Dice_score)
from mpi4py import MPI
import numpy as np
import time
import socket
import os

def _diffusion_child(comm, bm=None):

    rank = comm.Get_rank()
    ngpus = comm.Get_size()

    nodename = socket.gethostname()
    name = '%s %s' %(nodename, rank)
    print(name)

    if rank == 0:

        # initialize results
        results = {}

        # split indices on GPUs
        indices_split = _split_indices(bm.indices, ngpus)
        print('Indices:', indices_split)

        # send data to GPUs
        for k in range(1, ngpus):
            sendToChild(comm, bm.indices, indices_split[k], k, bm.data, bm.labels, bm.nbrw,
                        bm.sorw, bm.allaxis, bm.platform)

        # select platform
        if bm.platform == 'cuda':
            import pycuda.driver as cuda
            import pycuda.gpuarray as gpuarray
            from biomedisa.features.random_walk.gpu_kernels import (_build_kernel_uncertainty, 
                        _build_kernel_max, _build_update_gpu, _build_curvature_gpu)
            cuda.init()
            dev = cuda.Device(rank)
            ctx, queue = dev.make_context(), None
            if bm.allaxis:
                from biomedisa.features.random_walk.pycuda_small_allx import walk
            else:
                from biomedisa.features.random_walk.pycuda_small import walk
        else:
            ctx, queue = _get_device(bm.platform, rank)
            from biomedisa.features.random_walk.pyopencl_small import walk

        # run random walks
        tic = time.time()
        walkmap = walk(bm.data, bm.labels, bm.indices, indices_split[0], bm.nbrw, bm.sorw, name, ctx, queue)
        tac = time.time()
        print('Walktime_%s: ' %(name) + str(int(tac - tic)) + ' ' + 'seconds')

        # gather data
        zsh_tmp = bm.argmax_z - bm.argmin_z
        ysh_tmp = bm.argmax_y - bm.argmin_y
        xsh_tmp = bm.argmax_x - bm.argmin_x
        if ngpus > 1:
            final_zero = np.empty((bm.nol, zsh_tmp, ysh_tmp, xsh_tmp), dtype=np.float32)
            for k in range(bm.nol):
                sendbuf = np.copy(walkmap[k], order='C')
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
        if bm.smooth:
            try:
                update_gpu = _build_update_gpu()
                curvature_gpu = _build_curvature_gpu()
                a_gpu = gpuarray.empty((zsh_tmp, ysh_tmp, xsh_tmp), dtype=np.float32)
                b_gpu = gpuarray.zeros((zsh_tmp, ysh_tmp, xsh_tmp), dtype=np.float32)
                final_smooth = np.copy(final_zero, order='C')
                for k in range(bm.nol):
                    a_gpu = gpuarray.to_gpu(final_smooth[k])
                    for l in range(bm.smooth):
                        curvature_gpu(a_gpu, b_gpu, xsh_gpu, ysh_gpu, block=block, grid=grid)
                        update_gpu(a_gpu, b_gpu, xsh_gpu, ysh_gpu, block=block, grid=grid)
                    final_smooth[k] = a_gpu.get()
                final_smooth = np.argmax(final_smooth, axis=0).astype(np.uint8)
                final_smooth = get_labels(final_smooth, bm.allLabels)
                smooth_result = np.zeros((bm.zsh, bm.ysh, bm.xsh), dtype=np.uint8)
                smooth_result[bm.argmin_z:bm.argmax_z, bm.argmin_y:bm.argmax_y, bm.argmin_x:bm.argmax_x] = final_smooth
                smooth_result = smooth_result[1:-1, 1:-1, 1:-1]
                results['smooth'] = smooth_result
                if bm.django_env and not bm.remote:
                    bm.path_to_smooth = unique_file_path(bm.path_to_smooth)
                if bm.path_to_data:
                    save_data(bm.path_to_smooth, smooth_result, bm.header, bm.final_image_type, bm.compression)
            except Exception as e:
                print('Warning: GPU out of memory to allocate smooth array. Process starts without smoothing.')
                bm.smooth = 0

        # uncertainty
        if bm.uncertainty:
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
                uncertainty_result = np.zeros((bm.zsh, bm.ysh, bm.xsh), dtype=np.uint8)
                uncertainty_result[bm.argmin_z:bm.argmax_z, bm.argmin_y:bm.argmax_y, bm.argmin_x:bm.argmax_x] = uq
                uncertainty_result = uncertainty_result[1:-1, 1:-1, 1:-1]
                results['uncertainty'] = uncertainty_result
                if bm.django_env and not bm.remote:
                    bm.path_to_uq = unique_file_path(bm.path_to_uq)
                if bm.path_to_data:
                    save_data(bm.path_to_uq, uncertainty_result, compress=bm.compression)
            except Exception as e:
                print('Warning: GPU out of memory to allocate uncertainty array. Process starts without uncertainty.')
                bm.uncertainty = False

        # free device
        if bm.platform == 'cuda':
            ctx.pop()
            del ctx

        # return hits
        if bm.return_hits:
            hits = np.zeros((bm.nol, bm.zsh, bm.ysh, bm.xsh), dtype=np.float32)
            hits[:, bm.argmin_z:bm.argmax_z, bm.argmin_y:bm.argmax_y, bm.argmin_x:bm.argmax_x] = final_zero
            hits = hits[:,1:-1,1:-1,1:-1]
            results['hits'] = hits

        # get original labels
        final_zero = np.argmax(final_zero, axis=0).astype(np.uint8)
        final_zero = get_labels(final_zero, bm.allLabels)

        # validate result and check for allaxis
        mask = bm.labelData>0
        dice = Dice_score(bm.labelData, final_zero*mask)
        if dice < 0.3:
            print('Warning: Bad result! Use "--allaxis" if you labeled axes other than the xy-plane.')

        # regular result
        final_result = np.zeros((bm.zsh, bm.ysh, bm.xsh), dtype=np.uint8)
        final_result[bm.argmin_z:bm.argmax_z, bm.argmin_y:bm.argmax_y, bm.argmin_x:bm.argmax_x] = final_zero
        final_result = final_result[1:-1,1:-1,1:-1]
        results['regular'] = final_result
        if bm.django_env and not bm.remote:
            bm.path_to_final = unique_file_path(bm.path_to_final)
        if bm.path_to_data:
            save_data(bm.path_to_final, final_result, bm.header, bm.final_image_type, bm.compression)

        # remove outliers
        if bm.clean:
            cleaned_result = clean(final_result, bm.clean)
            results['cleaned'] = cleaned_result
            if bm.path_to_data:
                save_data(bm.path_to_cleaned, cleaned_result, bm.header, bm.final_image_type, bm.compression)
            if bm.smooth:
                smooth_cleaned = clean(smooth_result, bm.clean)
                results['smooth_cleaned'] = smooth_cleaned
                if bm.path_to_data:
                    save_data(bm.path_to_smooth_cleaned, smooth_cleaned, bm.header, bm.final_image_type, bm.compression)
        if bm.fill:
            filled_result = fill(final_result, bm.fill)
            results['filled'] = filled_result
            if bm.path_to_data:
                save_data(bm.path_to_filled, filled_result, bm.header, bm.final_image_type, bm.compression)
        if bm.clean and bm.fill:
            cleaned_filled_result = cleaned_result + (filled_result - final_result)
            results['cleaned_filled'] = cleaned_filled_result
            if bm.path_to_data:
                save_data(bm.path_to_cleaned_filled, cleaned_filled_result, bm.header, bm.final_image_type, bm.compression)

        # post-processing with active contour
        if bm.acwe:
            data = np.zeros((bm.zsh, bm.ysh, bm.xsh), dtype=bm.data.dtype)
            data[bm.argmin_z:bm.argmax_z, bm.argmin_y:bm.argmax_y, bm.argmin_x:bm.argmax_x] = bm.data
            acwe_result = activeContour(data[1:-1, 1:-1, 1:-1], final_result, bm.acwe_alpha, bm.acwe_smooth, bm.acwe_steps)
            refined_result = activeContour(data[1:-1, 1:-1, 1:-1], final_result, simple=True)
            results['acwe'] = acwe_result
            results['refined'] = refined_result
            if bm.path_to_data:
                save_data(bm.path_to_acwe, acwe_result, bm.header, bm.final_image_type, bm.compression)
                save_data(bm.path_to_refined, refined_result, bm.header, bm.final_image_type, bm.compression)

        # computation time
        t = int(time.time() - bm.TIC)
        if t < 60:
            time_str = str(t) + ' sec'
        elif 60 <= t < 3600:
            time_str = str(t // 60) + ' min ' + str(t % 60) + ' sec'
        elif 3600 < t:
            time_str = str(t // 3600) + ' h ' + str((t % 3600) // 60) + ' min ' + str(t % 60) + ' sec'
        print('Computation time:', time_str)

        # post processing
        if bm.django_env:
            from biomedisa_app.config import config
            from biomedisa.features.django_env import post_processing
            post_processing(bm.path_to_final, time_str, config['SERVER_ALIAS'], bm.remote, bm.queue,
                dice=dice, uncertainty=bm.uncertainty, smooth=bm.smooth,
                path_to_uq=bm.path_to_uq, path_to_smooth=bm.path_to_smooth,
                img_id=bm.img_id, label_id=bm.label_id)

            # write in logfile
            with open(bm.path_to_time, 'a') as timefile:
                print('%s %s %s %s MB %s on %s' %(time.ctime(), bm.username, bm.shortfilename, bm.imageSize, time_str, config['SERVER_ALIAS']), file=timefile)

        # return results
        return results

    else:

        data_z, data_y, data_x, data_dtype = comm.recv(source=0, tag=0)
        data = np.empty((data_z, data_y, data_x), dtype=data_dtype)
        if data_dtype == 'uint8':
            comm.Recv([data, MPI.BYTE], source=0, tag=1)
        else:
            comm.Recv([data, MPI.FLOAT], source=0, tag=1)
        allx, nbrw, sorw, platform = comm.recv(source=0, tag=2)
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

        # select platform
        if platform == 'cuda':
            import pycuda.driver as cuda
            cuda.init()
            dev = cuda.Device(rank % cuda.Device.count())
            ctx, queue = dev.make_context(), None
            if allx:
                from biomedisa.features.random_walk.pycuda_small_allx import walk
            else:
                from biomedisa.features.random_walk.pycuda_small import walk
        else:
            ctx, queue = _get_device(platform, rank)
            from biomedisa.features.random_walk.pyopencl_small import walk

        # run random walks
        tic = time.time()
        walkmap = walk(data, labels, indices, indices_child, nbrw, sorw, name, ctx, queue)
        tac = time.time()
        print('Walktime_%s: ' %(name) + str(int(tac - tic)) + ' ' + 'seconds')

        # free device
        if platform == 'cuda':
            ctx.pop()
            del ctx

        # send data
        for k in range(walkmap.shape[0]):
            datatemporaer = np.copy(walkmap[k], order='C')
            comm.Barrier()
            comm.Reduce([datatemporaer, MPI.FLOAT], None, root=0, op=MPI.SUM)

