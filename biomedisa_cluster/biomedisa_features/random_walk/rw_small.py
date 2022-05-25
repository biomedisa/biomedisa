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

def _diffusion_child(comm, bm=None):

    rank = comm.Get_rank()
    nodename = socket.gethostname()
    name = '%s %s' %(nodename, rank)
    print(name)

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

    # select the desired script
    if allx:
        from pycuda_small_allx import walk
    else:
        from pycuda_small import walk

    # init cuda device
    cuda.init()
    gpu_id = rank % cuda.Device.count()
    dev = cuda.Device(gpu_id)
    ctx = dev.make_context()

    # run random walks
    tic = time.time()
    walkmap = walk(data, labels, indices, indices_child, nbrw, sorw, name)
    tac = time.time()
    print("Walktime_%s: " %(name) + str(int(tac - tic)) + " " + "seconds")

    # free device
    ctx.pop()
    del ctx

    # send data
    for k in range(walkmap.shape[0]):
        datatemporaer = np.copy(walkmap[k])
        comm.Barrier()
        comm.Reduce([datatemporaer, MPI.FLOAT], None, root=0, op=MPI.SUM)
