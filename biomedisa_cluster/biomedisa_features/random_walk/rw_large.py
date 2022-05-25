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

from mpi4py import MPI
import pycuda.driver as cuda
import sys
import numpy as np
import time
import socket
import math

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

def _diffusion_child(comm, bm=None):

    rank = comm.Get_rank()
    nodename = socket.gethostname()
    name = '%s %s' %(nodename, rank)
    print(name)

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
    gpu_id = rank % cuda.Device.count()
    dev = cuda.Device(gpu_id)
    ctx = dev.make_context()

    # run random walks
    tic = time.time()
    memory_error, final, final_uncertainty, final_smooth = walk(comm, data, labels, indices, nbrw, sorw, blockmin, blockmax, name, allLabels, smooth, uncertainty)
    tac = time.time()
    print("Walktime_%s: " %(name) + str(int(tac - tic)) + " " + "seconds")

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
