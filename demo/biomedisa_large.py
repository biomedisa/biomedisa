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

from biomedisa_helper import save_data
from multiprocessing import Process
import pycuda.driver as cuda
from mpi4py import MPI
import os, sys
import numpy as np
import time
import socket
import math

def sendToChild(comm, indices, dest, dataListe, Labels, nbrw, sorw, blocks, allx, allLabels, smooth, uncertainty):
    comm.send(len(dataListe), dest=dest, tag=0)
    for k, tmp in enumerate(dataListe):
        tmp = tmp.copy(order='C')
        comm.send([tmp.shape[0], tmp.shape[1], tmp.shape[2], tmp.dtype], dest=dest, tag=10+(2*k))
        if tmp.dtype == 'uint8':
            comm.Send([tmp, MPI.BYTE], dest=dest, tag=10+(2*k+1))
        else:
            comm.Send([tmp, MPI.FLOAT], dest=dest, tag=10+(2*k+1))

    comm.send([nbrw, sorw, allx, smooth, uncertainty], dest=dest, tag=1)

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

def read_labeled_slices(volData):
    data = np.zeros((0, volData.shape[1], volData.shape[2]), dtype=np.int32)
    indices = []
    i = 0
    while i < volData.shape[0]:
        slc = volData[i]
        if np.any(slc):
            data = np.append(data, [volData[i]], axis=0)
            indices.append(i)
            i += 5
        else:
            i += 1
    return indices, data

def read_labeled_slices_allx(volData):
    gradient = np.zeros(volData.shape, dtype=np.int8)
    ones = np.zeros_like(gradient)
    ones[volData > 0] = 1
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
    data = np.zeros((0, volData.shape[1], volData.shape[2]), dtype=np.int32)
    for k, slc in enumerate(ones[:]):
        if np.any(slc):
            data = np.append(data, [volData[k]], axis=0)
            indices.append(k)
    return indices, data

def _diffusion_child(comm, bm=None):

    rank = comm.Get_rank()
    ngpus = comm.Get_size()

    nodename = socket.gethostname()
    name = '%s %s' %(nodename, rank)
    print(name)

    if rank == 0:

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
        if bm.label.allaxis:
            tmp = np.swapaxes(bm.labelData, 0, 1)
            tmp = np.ascontiguousarray(tmp)
            indices_01, _ = read_labeled_slices_allx(tmp)
            tmp = np.swapaxes(tmp, 0, 2)
            tmp = np.ascontiguousarray(tmp)
            indices_02, _ = read_labeled_slices_allx(tmp)

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
            if bm.label.allaxis:
                labelblock = labelblock.astype(np.int32)
                labelblock[:blockmin - datablockmin] = -1
                labelblock[blockmax - datablockmin:] = -1
                indices_child, labels_child = [], []
                indices_00, labels_00 = read_labeled_slices_allx(labelblock)
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
                indices_child, labels_child = read_labeled_slices(labelblock)

            # print indices of labels
            print('indices child %s:' %(destination), indices_child)

            if destination > 0:
                blocks_temp = blocks[:]
                blocks_temp[destination] = blockmin - datablockmin
                blocks_temp[destination+1] = blockmax - datablockmin
                dataListe = splitlargedata(datablock)
                sendToChild(comm, indices_child, destination, dataListe, labels_child, \
                            bm.label.nbrw, bm.label.sorw, blocks_temp, bm.label.allaxis, \
                            bm.allLabels, bm.label.smooth, bm.label.uncertainty)

            else:

                # select the desired script
                if bm.label.allaxis:
                    from pycuda_large_allx import walk
                else:
                    from pycuda_large import walk

                # init cuda device
                cuda.init()
                dev = cuda.Device(0)
                ctx = dev.make_context()

                # run random walks
                tic = time.time()
                memory_error, final, final_uncertainty, final_smooth = walk(comm, datablock, \
                                    labels_child, indices_child, bm.label.nbrw, bm.label.sorw, \
                                    blockmin-datablockmin, blockmax-datablockmin, name, \
                                    bm.allLabels, bm.label.smooth, bm.label.uncertainty)
                tac = time.time()
                print('Walktime_%s: ' %(name) + str(int(tac - tic)) + ' ' + 'seconds')

                # free device
                ctx.pop()
                del ctx

        if memory_error:

            print('GPU out of memory. Image too large.')

        else:

            # gather data
            for source in range(1, ngpus):
                lendataListe = comm.recv(source=source, tag=0)
                for l in range(lendataListe):
                    data_z, data_y, data_x = comm.recv(source=source, tag=10+(2*l))
                    receivedata = np.empty((data_z, data_y, data_x), dtype=np.uint8)
                    comm.Recv([receivedata, MPI.BYTE], source=source, tag=10+(2*l+1))
                    final = np.append(final, receivedata, axis=0)

            # save finals
            final2 = np.zeros((bm.zsh, bm.ysh, bm.xsh), dtype=np.uint8)
            final2[bm.argmin_z:bm.argmax_z, bm.argmin_y:bm.argmax_y, bm.argmin_x:bm.argmax_x] = final
            final2 = final2[1:-1, 1:-1, 1:-1]
            save_data(bm.path_to_final, final2, bm.header, bm.final_image_type, bm.label.compression)

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
                final2 = np.zeros((bm.zsh, bm.ysh, bm.xsh), dtype=np.uint8)
                final2[bm.argmin_z:bm.argmax_z, bm.argmin_y:bm.argmax_y, bm.argmin_x:bm.argmax_x] = final_uncertainty
                final2 = final2[1:-1, 1:-1, 1:-1]
                save_data(bm.path_to_uq, final2, compress=bm.label.compression)

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
                final2 = np.zeros((bm.zsh, bm.ysh, bm.xsh), dtype=np.uint8)
                final2[bm.argmin_z:bm.argmax_z, bm.argmin_y:bm.argmax_y, bm.argmin_x:bm.argmax_x] = final_smooth
                final2 = final2[1:-1, 1:-1, 1:-1]
                save_data(bm.path_to_smooth, final2, bm.header, bm.final_image_type, bm.label.compression)

            # print computation time
            t = int(time.time() - bm.TIC)
            if t < 60:
                time_str = str(t) + ' sec'
            elif 60 <= t < 3600:
                time_str = str(t // 60) + ' min ' + str(t % 60) + ' sec'
            elif 3600 < t:
                time_str = str(t // 3600) + ' h ' + str((t % 3600) // 60) + ' min ' + str(t % 60) + ' sec'
            print('Computation time:', time_str)

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

        nbrw, sorw, allx, smooth, uncertainty = comm.recv(source=0, tag=1)

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

        # select the desired script
        if allx:
            from pycuda_large_allx import walk
        else:
            from pycuda_large import walk

        # init cuda device
        cuda.init()
        dev = cuda.Device(rank % cuda.Device.count())
        ctx = dev.make_context()

        # run random walks
        tic = time.time()
        memory_error, final, final_uncertainty, final_smooth = walk(comm, data, \
                                    labels, indices, nbrw, sorw, blockmin, blockmax, \
                                    name, allLabels, smooth, uncertainty)
        tac = time.time()
        print('Walktime_%s: ' %(name) + str(int(tac - tic)) + ' ' + 'seconds')

        # free device
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

