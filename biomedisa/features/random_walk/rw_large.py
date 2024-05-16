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
from biomedisa.features.biomedisa_helper import (_get_device, save_data, _error_, unique_file_path,
    splitlargedata, read_labeled_slices_allx_large, read_labeled_slices_large, sendToChildLarge,
    Dice_score)
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

        # reduce blocksize
        bm.data = np.copy(bm.data[bm.argmin_z:bm.argmax_z, bm.argmin_y:bm.argmax_y, bm.argmin_x:bm.argmax_x], order='C')
        bm.labelData = np.copy(bm.labelData[bm.argmin_z:bm.argmax_z, bm.argmin_y:bm.argmax_y, bm.argmin_x:bm.argmax_x], order='C')

        # domain decomposition
        sizeofblocks = (bm.argmax_z - bm.argmin_z) // ngpus
        blocks = [0]
        for k in range(ngpus-1):
            block_temp = blocks[-1] + sizeofblocks
            blocks.append(block_temp)
        blocks.append(bm.argmax_z - bm.argmin_z)
        print('blocks =', blocks)

        # read labeled slices
        if bm.allaxis:
            tmp = np.swapaxes(bm.labelData, 0, 1)
            tmp = np.ascontiguousarray(tmp)
            indices_01, _ = read_labeled_slices_allx_large(tmp)
            tmp = np.swapaxes(tmp, 0, 2)
            tmp = np.ascontiguousarray(tmp)
            indices_02, _ = read_labeled_slices_allx_large(tmp)

        # send data to childs
        for destination in range(ngpus-1,-1,-1):

            # ghost blocks
            blockmin = blocks[destination]
            blockmax = blocks[destination+1]
            datablockmin = blockmin - 100
            datablockmax = blockmax + 100
            datablockmin = 0 if datablockmin < 0 else datablockmin
            datablockmax = (bm.argmax_z - bm.argmin_z) if datablockmax > (bm.argmax_z - bm.argmin_z) else datablockmax
            datablock = np.copy(bm.data[datablockmin:datablockmax], order='C')
            labelblock = np.copy(bm.labelData[datablockmin:datablockmax], order='C')

            # read labeled slices
            if bm.allaxis:
                labelblock = labelblock.astype(np.int32)
                labelblock[:blockmin - datablockmin] = -1
                labelblock[blockmax - datablockmin:] = -1
                indices_child, labels_child = [], []
                indices_00, labels_00 = read_labeled_slices_allx_large(labelblock)
                indices_child.append(indices_00)
                labels_child.append(labels_00)
                tmp = np.swapaxes(labelblock, 0, 1)
                tmp = np.ascontiguousarray(tmp)
                labels_01 = np.zeros((0, tmp.shape[1], tmp.shape[2]), dtype=np.int32)
                for slcIndex in indices_01:
                    labels_01 = np.append(labels_01, [tmp[slcIndex]], axis=0)
                indices_child.append(indices_01)
                labels_child.append(labels_01)
                tmp = np.swapaxes(tmp, 0, 2)
                tmp = np.ascontiguousarray(tmp)
                labels_02 = np.zeros((0, tmp.shape[1], tmp.shape[2]), dtype=np.int32)
                for slcIndex in indices_02:
                    labels_02 = np.append(labels_02, [tmp[slcIndex]], axis=0)
                indices_child.append(indices_02)
                labels_child.append(labels_02)
            else:
                labelblock[:blockmin - datablockmin] = 0
                labelblock[blockmax - datablockmin:] = 0
                indices_child, labels_child = read_labeled_slices_large(labelblock)

            # print indices of labels
            print('indices child %s:' %(destination), indices_child)

            if destination > 0:
                blocks_temp = blocks[:]
                blocks_temp[destination] = blockmin - datablockmin
                blocks_temp[destination+1] = blockmax - datablockmin
                dataListe = splitlargedata(datablock)
                sendToChildLarge(comm, indices_child, destination, dataListe, labels_child,
                            bm.nbrw, bm.sorw, blocks_temp, bm.allaxis,
                            bm.allLabels, bm.smooth, bm.uncertainty, bm.platform)

            else:

                # select platform
                if bm.platform == 'cuda':
                    import pycuda.driver as cuda
                    cuda.init()
                    dev = cuda.Device(rank)
                    ctx, queue = dev.make_context(), None
                    if bm.allaxis:
                        from biomedisa.features.random_walk.pycuda_large_allx import walk
                    else:
                        from biomedisa.features.random_walk.pycuda_large import walk
                else:
                    ctx, queue = _get_device(bm.platform, rank)
                    from biomedisa.features.random_walk.pyopencl_large import walk

                # run random walks
                tic = time.time()
                memory_error, final, final_uncertainty, final_smooth = walk(comm, datablock,
                                    labels_child, indices_child, bm.nbrw, bm.sorw,
                                    blockmin-datablockmin, blockmax-datablockmin, name,
                                    bm.allLabels, bm.smooth, bm.uncertainty,
                                    ctx, queue, bm.platform)
                tac = time.time()
                print('Walktime_%s: ' %(name) + str(int(tac - tic)) + ' ' + 'seconds')

                # free device
                if bm.platform == 'cuda':
                    ctx.pop()
                    del ctx

        if memory_error:
            bm = _error_(bm, 'GPU out of memory. Image too large.')
            return results

        else:

            # gather data
            for source in range(1, ngpus):
                lendataListe = comm.recv(source=source, tag=0)
                for l in range(lendataListe):
                    data_z, data_y, data_x = comm.recv(source=source, tag=10+(2*l))
                    receivedata = np.empty((data_z, data_y, data_x), dtype=np.uint8)
                    comm.Recv([receivedata, MPI.BYTE], source=source, tag=10+(2*l+1))
                    final = np.append(final, receivedata, axis=0)

            # validate result and check for allaxis
            mask = bm.labelData>0
            dice = Dice_score(bm.labelData, final*mask)
            if dice < 0.3:
                print('Warning: Bad result! Use "--allaxis" if you labeled axes other than the xy-plane.')

            # regular result
            final_result = np.zeros((bm.zsh, bm.ysh, bm.xsh), dtype=np.uint8)
            final_result[bm.argmin_z:bm.argmax_z, bm.argmin_y:bm.argmax_y, bm.argmin_x:bm.argmax_x] = final
            final_result = final_result[1:-1, 1:-1, 1:-1]
            results['regular'] = final_result
            if bm.django_env and not bm.remote:
                bm.path_to_final = unique_file_path(bm.path_to_final)
            if bm.path_to_data:
                save_data(bm.path_to_final, final_result, bm.header, bm.final_image_type, bm.compression)

            # uncertainty
            if final_uncertainty is not None:
                final_uncertainty *= 255
                final_uncertainty = final_uncertainty.astype(np.uint8)
                for source in range(1, ngpus):
                    lendataListe = comm.recv(source=source, tag=0)
                    for l in range(lendataListe):
                        data_z, data_y, data_x = comm.recv(source=source, tag=10+(2*l))
                        receivedata = np.empty((data_z, data_y, data_x), dtype=np.uint8)
                        comm.Recv([receivedata, MPI.BYTE], source=source, tag=10+(2*l+1))
                        final_uncertainty = np.append(final_uncertainty, receivedata, axis=0)
                # save finals
                uncertainty_result = np.zeros((bm.zsh, bm.ysh, bm.xsh), dtype=np.uint8)
                uncertainty_result[bm.argmin_z:bm.argmax_z, bm.argmin_y:bm.argmax_y, bm.argmin_x:bm.argmax_x] = final_uncertainty
                uncertainty_result = uncertainty_result[1:-1, 1:-1, 1:-1]
                results['uncertainty'] = uncertainty_result
                if bm.django_env and not bm.remote:
                    bm.path_to_uq = unique_file_path(bm.path_to_uq)
                if bm.path_to_data:
                    save_data(bm.path_to_uq, uncertainty_result, compress=bm.compression)
            else:
                bm.uncertainty = False

            # smooth
            if final_smooth is not None:
                for source in range(1, ngpus):
                    lendataListe = comm.recv(source=source, tag=0)
                    for l in range(lendataListe):
                        data_z, data_y, data_x = comm.recv(source=source, tag=10+(2*l))
                        receivedata = np.empty((data_z, data_y, data_x), dtype=np.uint8)
                        comm.Recv([receivedata, MPI.BYTE], source=source, tag=10+(2*l+1))
                        final_smooth = np.append(final_smooth, receivedata, axis=0)
                # save finals
                smooth_result = np.zeros((bm.zsh, bm.ysh, bm.xsh), dtype=np.uint8)
                smooth_result[bm.argmin_z:bm.argmax_z, bm.argmin_y:bm.argmax_y, bm.argmin_x:bm.argmax_x] = final_smooth
                smooth_result = smooth_result[1:-1, 1:-1, 1:-1]
                results['smooth'] = smooth_result
                if bm.django_env and not bm.remote:
                    bm.path_to_smooth = unique_file_path(bm.path_to_smooth)
                if bm.path_to_data:
                    save_data(bm.path_to_smooth, smooth_result, bm.header, bm.final_image_type, bm.compression)
            else:
                bm.smooth = 0

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

        lendataListe = comm.recv(source=0, tag=0)
        for k in range(lendataListe):
            data_z, data_y, data_x, data_dtype = comm.recv(source=0, tag=10+(2*k))
            if k==0: data = np.zeros((0, data_y, data_x), dtype=data_dtype)
            data_temp = np.empty((data_z, data_y, data_x), dtype=data_dtype)
            if data_dtype == 'uint8':
                comm.Recv([data_temp, MPI.BYTE], source=0, tag=10+(2*k+1))
            else:
                comm.Recv([data_temp, MPI.FLOAT], source=0, tag=10+(2*k+1))
            data = np.append(data, data_temp, axis=0)

        nbrw, sorw, allx, smooth, uncertainty, platform = comm.recv(source=0, tag=1)

        if allx:
            labels = []
            for k in range(3):
                lenlabelsListe = comm.recv(source=0, tag=2+k)
                for l in range(lenlabelsListe):
                    labels_z, labels_y, labels_x = comm.recv(source=0, tag=100+(2*k))
                    if l==0: labels_tmp = np.zeros((0, labels_y, labels_x), dtype=np.int32)
                    tmp = np.empty((labels_z, labels_y, labels_x), dtype=np.int32)
                    comm.Recv([tmp, MPI.INT], source=0, tag=100+(2*k+1))
                    labels_tmp = np.append(labels_tmp, tmp, axis=0)
                labels.append(labels_tmp)
        else:
            lenlabelsListe = comm.recv(source=0, tag=2)
            for k in range(lenlabelsListe):
                labels_z, labels_y, labels_x = comm.recv(source=0, tag=100+(2*k))
                if k==0: labels = np.zeros((0, labels_y, labels_x), dtype=np.int32)
                tmp = np.empty((labels_z, labels_y, labels_x), dtype=np.int32)
                comm.Recv([tmp, MPI.INT], source=0, tag=100+(2*k+1))
                labels = np.append(labels, tmp, axis=0)

        allLabels = comm.recv(source=0, tag=99)
        indices = comm.recv(source=0, tag=8)
        blocks = comm.recv(source=0, tag=9)

        blockmin = blocks[rank]
        blockmax = blocks[rank+1]

        # select platform
        if platform == 'cuda':
            import pycuda.driver as cuda
            cuda.init()
            dev = cuda.Device(rank % cuda.Device.count())
            ctx, queue = dev.make_context(), None
            if allx:
                from biomedisa.features.random_walk.pycuda_large_allx import walk
            else:
                from biomedisa.features.random_walk.pycuda_large import walk
        else:
            ctx, queue = _get_device(platform, rank)
            from biomedisa.features.random_walk.pyopencl_large import walk

        # run random walks
        tic = time.time()
        memory_error, final, final_uncertainty, final_smooth = walk(comm, data, labels, indices, nbrw, sorw,
                blockmin, blockmax, name, allLabels, smooth, uncertainty, ctx, queue, platform)
        tac = time.time()
        print('Walktime_%s: ' %(name) + str(int(tac - tic)) + ' ' + 'seconds')

        # free device
        if platform == 'cuda':
            ctx.pop()
            del ctx

        # send finals
        if not memory_error:
            dataListe = splitlargedata(final)
            comm.send(len(dataListe), dest=0, tag=0)
            for k, dataTemp in enumerate(dataListe):
                dataTemp = dataTemp.copy(order='C')
                comm.send([dataTemp.shape[0], dataTemp.shape[1], dataTemp.shape[2]], dest=0, tag=10+(2*k))
                comm.Send([dataTemp, MPI.BYTE], dest=0, tag=10+(2*k+1))

            if final_uncertainty is not None:
                final_uncertainty *= 255
                final_uncertainty = final_uncertainty.astype(np.uint8)
                dataListe = splitlargedata(final_uncertainty)
                comm.send(len(dataListe), dest=0, tag=0)
                for k, dataTemp in enumerate(dataListe):
                    dataTemp = dataTemp.copy(order='C')
                    comm.send([dataTemp.shape[0], dataTemp.shape[1], dataTemp.shape[2]], dest=0, tag=10+(2*k))
                    comm.Send([dataTemp, MPI.BYTE], dest=0, tag=10+(2*k+1))

            if final_smooth is not None:
                dataListe = splitlargedata(final_smooth)
                comm.send(len(dataListe), dest=0, tag=0)
                for k, dataTemp in enumerate(dataListe):
                    dataTemp = dataTemp.copy(order='C')
                    comm.send([dataTemp.shape[0], dataTemp.shape[1], dataTemp.shape[2]], dest=0, tag=10+(2*k))
                    comm.Send([dataTemp, MPI.BYTE], dest=0, tag=10+(2*k+1))

