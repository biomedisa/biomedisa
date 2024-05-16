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

from mpi4py import MPI
import numba
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from biomedisa.features.random_walk.gpu_kernels import (_build_kernel_uncertainty,
        _build_kernel_max, _build_kernel_fill, _build_update_gpu, _build_curvature_gpu)

def reduceBlocksize(slices):
    zsh, ysh, xsh = slices.shape
    argmin_x, argmax_x, argmin_y, argmax_y = xsh, 0, ysh, 0
    for k in range(zsh):
        y, x = np.nonzero(slices[k])
        if x.any():
            argmin_x = min(argmin_x, np.amin(x))
            argmax_x = max(argmax_x, np.amax(x))
            argmin_y = min(argmin_y, np.amin(y))
            argmax_y = max(argmax_y, np.amax(y))
    argmin_x = argmin_x - 100 if argmin_x - 100 > 0 else 0
    argmax_x = argmax_x + 100 if argmax_x + 100 < xsh else xsh
    argmin_y = argmin_y - 100 if argmin_y - 100 > 0 else 0
    argmax_y = argmax_y + 100 if argmax_y + 100 < ysh else ysh
    slices[:, :argmin_y] = -1
    slices[:, argmax_y:] = -1
    slices[:, :, :argmin_x] = -1
    slices[:, :, argmax_x:] = -1
    return slices

def sendrecv(a, blockmin, blockmax, comm, rank, size):

    sendbuf = np.empty(1, dtype=np.int32)
    recvbuf = np.empty_like(sendbuf)

    if rank == 0:

        # send block
        dest = rank+1
        tmp = a[blockmax:]
        if np.any(tmp):
            sendbuf.fill(1)
            comm.Send([sendbuf, MPI.INT], dest=dest, tag=0)
            send = tmp.copy(order='C')
            comm.send([send.shape[0], send.shape[1], send.shape[2]], dest=dest, tag=1)
            comm.Send([send, MPI.FLOAT], dest=dest, tag=2)
        else:
            sendbuf.fill(0)
            comm.Send([sendbuf, MPI.INT], dest=dest, tag=0)

        # recv block
        source = rank+1
        comm.Recv([recvbuf, MPI.INT], source=source, tag=3)
        if recvbuf:
            data_z, data_y, data_x = comm.recv(source=source, tag=4)
            recv = np.empty((data_z, data_y, data_x), dtype=np.float32)
            comm.Recv([recv, MPI.FLOAT], source=source, tag=5)
            a[blockmax-data_z:blockmax] += recv

    elif rank == size-1:

        if rank % 2 == 1: add = 0
        if rank % 2 == 0: add = 6

        # recv block
        source = rank-1
        comm.Recv([recvbuf, MPI.INT], source=source, tag=0+add)
        if recvbuf:
            data_z, data_y, data_x = comm.recv(source=source, tag=1+add)
            recv = np.empty((data_z, data_y, data_x), dtype=np.float32)
            comm.Recv([recv, MPI.FLOAT], source=source, tag=2+add)
            a[blockmin:blockmin+data_z] += recv

        # send block
        dest = rank-1
        tmp = a[:blockmin]
        if np.any(tmp):
            sendbuf.fill(1)
            comm.Send([sendbuf, MPI.INT], dest=dest, tag=3+add)
            send = tmp.copy(order='C')
            comm.send([send.shape[0], send.shape[1], send.shape[2]], dest=dest, tag=4+add)
            comm.Send([send, MPI.FLOAT], dest=dest, tag=5+add)
        else:
            sendbuf.fill(0)
            comm.Send([sendbuf, MPI.INT], dest=dest, tag=3+add)

    elif rank % 2 == 1:

        # recv block
        source = rank-1
        comm.Recv([recvbuf, MPI.INT], source=source, tag=0)
        if recvbuf:
            data_z, data_y, data_x = comm.recv(source=source, tag=1)
            recv = np.empty((data_z, data_y, data_x), dtype=np.float32)
            comm.Recv([recv, MPI.FLOAT], source=source, tag=2)
            a[blockmin:blockmin+data_z] += recv

        # send block
        dest = rank-1
        tmp = a[:blockmin]
        if np.any(tmp):
            sendbuf.fill(1)
            comm.Send([sendbuf, MPI.INT], dest=dest, tag=3)
            send = tmp.copy(order='C')
            comm.send([send.shape[0], send.shape[1], send.shape[2]], dest=dest, tag=4)
            comm.Send([send, MPI.FLOAT], dest=dest, tag=5)
        else:
            sendbuf.fill(0)
            comm.Send([sendbuf, MPI.INT], dest=dest, tag=3)

        # send block
        dest = rank+1
        tmp = a[blockmax:]
        if np.any(tmp):
            sendbuf.fill(1)
            comm.Send([sendbuf, MPI.INT], dest=dest, tag=6)
            send = tmp.copy(order='C')
            comm.send([send.shape[0], send.shape[1], send.shape[2]], dest=dest, tag=7)
            comm.Send([send, MPI.FLOAT], dest=dest, tag=8)
        else:
            sendbuf.fill(0)
            comm.Send([sendbuf, MPI.INT], dest=dest, tag=6)

        # recv block
        source = rank+1
        comm.Recv([recvbuf, MPI.INT], source=source, tag=9)
        if recvbuf:
            data_z, data_y, data_x = comm.recv(source=source, tag=10)
            recv = np.empty((data_z, data_y, data_x), dtype=np.float32)
            comm.Recv([recv, MPI.FLOAT], source=source, tag=11)
            a[blockmax-data_z:blockmax] += recv

    elif rank % 2 == 0:

        # send block
        dest = rank+1
        tmp = a[blockmax:]
        if np.any(tmp):
            sendbuf.fill(1)
            comm.Send([sendbuf, MPI.INT], dest=dest, tag=0)
            send = tmp.copy(order='C')
            comm.send([send.shape[0], send.shape[1], send.shape[2]], dest=dest, tag=1)
            comm.Send([send, MPI.FLOAT], dest=dest, tag=2)
        else:
            sendbuf.fill(0)
            comm.Send([sendbuf, MPI.INT], dest=dest, tag=0)

        # recv block
        source = rank+1
        comm.Recv([recvbuf, MPI.INT], source=source, tag=3)
        if recvbuf:
            data_z, data_y, data_x = comm.recv(source=source, tag=4)
            recv = np.empty((data_z, data_y, data_x), dtype=np.float32)
            comm.Recv([recv, MPI.FLOAT], source=source, tag=5)
            a[blockmax-data_z:blockmax] += recv

        # recv block
        source = rank-1
        comm.Recv([recvbuf, MPI.INT], source=source, tag=6)
        if recvbuf:
            data_z, data_y, data_x = comm.recv(source=source, tag=7)
            recv = np.empty((data_z, data_y, data_x), dtype=np.float32)
            comm.Recv([recv, MPI.FLOAT], source=source, tag=8)
            a[blockmin:blockmin+data_z] += recv

        # send block
        dest = rank-1
        tmp = a[:blockmin]
        if np.any(tmp):
            sendbuf.fill(1)
            comm.Send([sendbuf, MPI.INT], dest=dest, tag=9)
            send = tmp.copy(order='C')
            comm.send([send.shape[0], send.shape[1], send.shape[2]], dest=dest, tag=10)
            comm.Send([send, MPI.FLOAT], dest=dest, tag=11)
        else:
            sendbuf.fill(0)
            comm.Send([sendbuf, MPI.INT], dest=dest, tag=9)

    return a

@numba.jit(nopython=True)
def max_to_label(a, walkmap, final, blockmin, blockmax, segment):
    zsh, ysh, xsh = a.shape
    for k in range(blockmin, blockmax):
        for l in range(ysh):
            for m in range(xsh):
                if a[k,l,m] > walkmap[k,l,m]:
                    walkmap[k,l,m] = a[k,l,m]
                    final[k-blockmin,l,m] = segment
    return walkmap, final

def walk(comm, raw, slices, indices, nbrw, sorw, blockmin, blockmax, name,
         allLabels, smooth, uncertainty, ctx, queue, platform):

    # get rank and size of mpi process
    rank = comm.Get_rank()
    size = comm.Get_size()

    # build kernels
    if raw.dtype == 'uint8':
        kernel = _build_kernel_int8()
        raw = (raw-128).astype('int8')
    else:   
        kernel = _build_kernel_float32()
        raw = raw.astype(np.float32)
    fill_gpu = _build_kernel_fill()

    # image size
    zsh, ysh, xsh = raw.shape
    xsh_gpu = np.int32(xsh)
    ysh_gpu = np.int32(ysh)
    zsh_gpu = np.int32(zsh)

    # block and gird size
    block = (32, 32, 1)
    x_grid = (xsh // 32) + 1
    y_grid = (ysh // 32) + 1
    grid2 = (int(x_grid), int(y_grid), int(zsh))

    # hyper-parameter
    sorw = np.int32(sorw)
    nbrw = np.int32(nbrw)

    # crop to region of interest
    slices = slices.astype(np.int32)
    slices = reduceBlocksize(slices)

    # allocate host memory
    hits = np.empty(raw.shape, dtype=np.float32)
    final = np.zeros((blockmax-blockmin, ysh, xsh), dtype=np.uint8)

    # allocate GPU memory or use subdomains
    memory_error = False
    subdomains = False
    if zsh * ysh * xsh > 42e8:
        print('Warning: Volume indexes exceed unsigned long int range. The volume is splitted into subdomains.')
        subdomains = True
        sendbuf = np.zeros(1, dtype=np.int32) + 1
        recvbuf = np.zeros(1, dtype=np.int32)
        comm.Barrier()
        comm.Allreduce([sendbuf, MPI.INT], [recvbuf, MPI.INT], op=MPI.MAX)
    else:
        try:
            if np.any(indices):
                slshape = slices.shape[0]
                indices = np.array(indices, dtype=np.int32)
                indices_gpu = gpuarray.to_gpu(indices)
                slices_gpu = gpuarray.to_gpu(slices)
                grid = (int(x_grid), int(y_grid), int(slshape))
            raw_gpu = gpuarray.to_gpu(raw)
            hits_gpu = cuda.mem_alloc(hits.nbytes)
            sendbuf = np.zeros(1, dtype=np.int32)
            recvbuf = np.zeros(1, dtype=np.int32)
            comm.Barrier()
            comm.Allreduce([sendbuf, MPI.INT], [recvbuf, MPI.INT], op=MPI.MAX)
        except Exception as e:
            print('Warning: GPU ran out of memory. The volume is splitted into subdomains.')
            subdomains = True
            sendbuf = np.zeros(1, dtype=np.int32) + 1
            recvbuf = np.zeros(1, dtype=np.int32)
            comm.Barrier()
            comm.Allreduce([sendbuf, MPI.INT], [recvbuf, MPI.INT], op=MPI.MAX)
            try:
                hits_gpu.free()
            except:
                pass

    # disable smoothing and uncertainty for subdomains
    if recvbuf > 0:
        smooth, uncertainty = 0, 0

    if smooth:
        try:
            update_gpu = _build_update_gpu()
            curvature_gpu = _build_curvature_gpu()
            b_npy = np.zeros(raw.shape, dtype=np.float32)
            b_gpu = cuda.mem_alloc(b_npy.nbytes)
            cuda.memcpy_htod(b_gpu, b_npy)
            final_smooth = np.zeros((blockmax-blockmin, ysh, xsh), dtype=np.uint8)
            sendbuf_smooth = np.zeros(1, dtype=np.int32)
            recvbuf_smooth = np.zeros(1, dtype=np.int32)
            comm.Barrier()
            comm.Allreduce([sendbuf_smooth, MPI.INT], [recvbuf_smooth, MPI.INT], op=MPI.MAX)
        except Exception as e:
            print('Warning: GPU out of memory to allocate smooth array. Process starts without smoothing.')
            sendbuf_smooth = np.zeros(1, dtype=np.int32) + 1
            recvbuf_smooth = np.zeros(1, dtype=np.int32)
            comm.Barrier()
            comm.Allreduce([sendbuf_smooth, MPI.INT], [recvbuf_smooth, MPI.INT], op=MPI.MAX)
        if recvbuf_smooth > 0:
            smooth = 0
            try:
                b_gpu.free()
            except:
                pass

    if uncertainty:
        try:
            max_npy = np.zeros((3,)+raw.shape, dtype=np.float32)
            max_gpu = cuda.mem_alloc(max_npy.nbytes)
            cuda.memcpy_htod(max_gpu, max_npy)
            kernel_uncertainty = _build_kernel_uncertainty()
            kernel_max = _build_kernel_max()
            sendbuf_uq = np.zeros(1, dtype=np.int32)
            recvbuf_uq = np.zeros(1, dtype=np.int32)
            comm.Barrier()
            comm.Allreduce([sendbuf_uq, MPI.INT], [recvbuf_uq, MPI.INT], op=MPI.MAX)
        except Exception as e:
            print('Warning: GPU out of memory to allocate uncertainty array. Process starts without uncertainty.')
            sendbuf_uq = np.zeros(1, dtype=np.int32) + 1
            recvbuf_uq = np.zeros(1, dtype=np.int32)
            comm.Barrier()
            comm.Allreduce([sendbuf_uq, MPI.INT], [recvbuf_uq, MPI.INT], op=MPI.MAX)
        if recvbuf_uq > 0:
            uncertainty = False
            try:
                max_gpu.free()
            except:
                pass

    for label_counter, segment in enumerate(allLabels):
        print('%s:' %(name) + ' ' + str(label_counter+1) + '/' + str(len(allLabels)))

        # current segment
        segment_gpu = np.int32(segment)

        # split volume into subdomains
        if subdomains:
            try:
                hits.fill(0)
                sub_n = (blockmax-blockmin) // 100 + 1
                for sub_k in range(sub_n):
                    sub_block_min = sub_k*100+blockmin
                    sub_block_max = (sub_k+1)*100+blockmin
                    data_block_min = max(sub_block_min-100,0)
                    data_block_max = min(sub_block_max+100,zsh)

                    # get subindices
                    sub_indices = []
                    sub_slices = np.empty((0, ysh, xsh), dtype=slices.dtype)
                    for k, sub_i in enumerate(indices):
                        if sub_block_min <= sub_i < sub_block_max and np.any(slices[k]==segment):
                            sub_indices.append(sub_i)
                            sub_slices = np.append(sub_slices, [slices[k]], axis=0)

                    # allocate memory and compute random walks on subdomain
                    if np.any(sub_indices):
                        sub_slshape = sub_slices.shape[0]
                        sub_indices = np.array(sub_indices, dtype=np.int32) - data_block_min
                        sub_indices_gpu = gpuarray.to_gpu(sub_indices)
                        sub_slices_gpu = gpuarray.to_gpu(sub_slices)

                        sub_zsh = data_block_max - data_block_min
                        sub_zsh_gpu = np.int32(sub_zsh)
                        sub_raw = np.copy(raw[data_block_min:data_block_max], order='C')
                        sub_raw_gpu = gpuarray.to_gpu(sub_raw)
                        sub_hits = np.empty(sub_raw.shape, dtype=np.float32)
                        sub_hits_gpu = cuda.mem_alloc(sub_hits.nbytes)
                        fill_gpu(sub_hits_gpu, xsh_gpu, ysh_gpu, block=block, grid=(int(x_grid), int(y_grid), int(sub_zsh)))
                        kernel(segment_gpu, sub_raw_gpu, sub_slices_gpu, sub_hits_gpu, xsh_gpu, ysh_gpu, sub_zsh_gpu, sub_indices_gpu, sorw, nbrw, block=block, grid=(int(x_grid), int(y_grid), int(sub_slshape)))
                        cuda.memcpy_dtoh(sub_hits, sub_hits_gpu)
                        hits[data_block_min:data_block_max] += sub_hits
                        sub_hits_gpu.free()
            except Exception as e:
                print('Error: GPU out of memory. Data too large.')
                memory_error = True
                try:
                    sub_hits_gpu.free()
                except:
                    pass

        # computation of random walks on the entire volume
        else:
            # reset array of hits
            fill_gpu(hits_gpu, xsh_gpu, ysh_gpu, block=block, grid=grid2)

            # compute random walks
            if np.any(indices):
                kernel(segment_gpu, raw_gpu, slices_gpu, hits_gpu, xsh_gpu, ysh_gpu, zsh_gpu, indices_gpu, sorw, nbrw, block=block, grid=grid)

            # get hits
            cuda.memcpy_dtoh(hits, hits_gpu)

        # memory error
        if memory_error:
            sendbuf = np.zeros(1, dtype=np.int32) + 1
            recvbuf = np.zeros(1, dtype=np.int32)
            comm.Barrier()
            comm.Allreduce([sendbuf, MPI.INT], [recvbuf, MPI.INT], op=MPI.MAX)
        else:
            sendbuf = np.zeros(1, dtype=np.int32)
            recvbuf = np.zeros(1, dtype=np.int32)
            comm.Barrier()
            comm.Allreduce([sendbuf, MPI.INT], [recvbuf, MPI.INT], op=MPI.MAX)
        if recvbuf > 0:
            memory_error = True
            try:
                hits_gpu.free()
            except:
                pass
            return memory_error, None, None, None

        # communicate hits
        if size > 1:
            hits = sendrecv(hits, blockmin, blockmax, comm, rank, size)
            if uncertainty or smooth:
                cuda.memcpy_htod(hits_gpu, hits)

        # save the three most occuring hits
        if uncertainty:
            kernel_max(max_gpu, hits_gpu, xsh_gpu, ysh_gpu, block=block, grid=grid2)

        # smooth manifold
        if smooth:
            for k in range(smooth):
                curvature_gpu(hits_gpu, b_gpu, xsh_gpu, ysh_gpu, block=block, grid=grid2)
                update_gpu(hits_gpu, b_gpu, xsh_gpu, ysh_gpu, block=block, grid=grid2)
            hits_smooth = np.empty_like(hits)
            cuda.memcpy_dtoh(hits_smooth, hits_gpu)
            if label_counter == 0:
                hits_smooth[hits_smooth<0] = 0
                walkmap_smooth = np.copy(hits_smooth, order='C')
            else:
                walkmap_smooth, final_smooth = max_to_label(hits_smooth, walkmap_smooth, final_smooth, blockmin, blockmax, segment)

        # get the label with the most hits
        if label_counter == 0:
            walkmap = np.copy(hits, order='C')
        else:
            walkmap, final = max_to_label(hits, walkmap, final, blockmin, blockmax, segment)
            #update = hits[blockmin:blockmax] > walkmap[blockmin:blockmax]
            #walkmap[blockmin:blockmax][update] = hits[blockmin:blockmax][update]
            #final[update] = segment

    # compute uncertainty
    if uncertainty:
        kernel_uncertainty(max_gpu, hits_gpu, xsh_gpu, ysh_gpu, block=block, grid=grid2)
        final_uncertainty = np.empty_like(hits)
        cuda.memcpy_dtoh(final_uncertainty, hits_gpu)
        final_uncertainty = final_uncertainty[blockmin:blockmax]
    else:
        final_uncertainty = None

    if not smooth:
        final_smooth = None

    try:
        hits_gpu.free()
    except:
        pass

    return memory_error, final, final_uncertainty, final_smooth

def _build_kernel_int8():
    code = """

    __device__ float _calc_var(unsigned int position, unsigned int index, float B, float *raw, int segment, int *labels, int xsh) {
        float dev = 0;
        float summe = 0;
        for (int n = -1; n < 2; n++) {
            for (int o = -1; o < 2; o++) {
                if (labels[index + n*xsh + o] == segment) {
                    float tmp = B - (float)(*((char*)(raw) + position + n*xsh + o));
                    dev += tmp * tmp;
                    summe += 1;
                    }
                }
            }
        float var = dev / summe;
        if (var < 1.0) {
            var = 1.0;
            }
        return var;
        }

    __device__ float weight(float B, float *raw, float div1, unsigned int position) {
        float tmp = B - (float)(*((char*)(raw) + position));
        return exp( - tmp * tmp * div1 );
        }

    __global__ void Funktion(int segment, float *raw, int *slices, float *hits, int xsh, int ysh, int zsh, int *indices, int sorw, int nbrw) {

        int flat   = xsh * ysh;
        int column = blockIdx.x * blockDim.x + threadIdx.x;
        int row    = blockIdx.y * blockDim.y + threadIdx.y;
        int slice  = blockIdx.z;
        int plane  = indices[slice];
        unsigned int index = slice * flat + row * xsh + column;
        unsigned int position = plane * flat + row * xsh + column;

        if (index < gridDim.z*flat && plane>0 && row>0 && column>0 && plane<zsh-1 && row<ysh-1 && column<xsh-1) {

            if (slices[index]==segment) {

                /* Adaptive random walks */
                int found = 0;
                if ((column + row) % 4 == 0) {
                    found = 1;
                    }
                else {
                    for (int y = -100; y < 101; y++) {
                        for (int x = -100; x < 101; x++) {
                            if (row+y > 0 && column+x > 0 && row+y < ysh-1 && column+x < xsh-1) {
                                unsigned int tmp = slice * flat + (row+y) * xsh + column+x;
                                if (slices[tmp] != segment && slices[tmp] != -1) {
                                    found = 1;
                                    }
                                }
                            }
                        }
                    }

                if (found == 1) {

                    float rand;
                    float W0,W1,W2,W3,W4,W5;
                    int n,o,p;

                    /* Initialize MRG32k3a */
                    float norm = 2.328306549295728e-10;
                    float m1 = 4294967087.0;
                    float m2 = 4294944443.0;
                    float a12 = 1403580.0;
                    float a13n = 810728.0;
                    float a21 = 527612.0;
                    float a23n = 1370589.0;
                    long k1;
                    float p1, p2;
                    float s10 = index, s11 = index, s12 = index, s20 = index, s21 = index, s22 = index;

                    /* Compute standard deviation */
                    float B = (float)(*((char*)(raw) + position));
                    float var = _calc_var(position, index, B, raw, segment, slices, xsh);
                    float div1 = 1 / (2 * var);

                    int k = plane;
                    int l = row;
                    int m = column;

                    int step = 0;
                    int n_rw = 0;

                    /* Compute random walks */
                    while (n_rw < nbrw) {

                        /* Compute weights */
                        W0 = weight(B, raw, div1, position + flat);
                        W1 = weight(B, raw, div1, position - flat);
                        W2 = weight(B, raw, div1, position + xsh);
                        W3 = weight(B, raw, div1, position - xsh);
                        W4 = weight(B, raw, div1, position + 1);
                        W5 = weight(B, raw, div1, position - 1);

                        W1 += W0;
                        W2 += W1;
                        W3 += W2;
                        W4 += W3;
                        W5 += W4;

                        /* Compute random numbers with MRG32k3a */

                        /* Component 1 */
                        p1 = a12 * s11 - a13n * s10;
                        k1 = p1 / m1;
                        p1 -= k1 * m1;
                        if (p1 < 0.0){
                            p1 += m1;}
                        s10 = s11;
                        s11 = s12;
                        s12 = p1;

                        /* Component 2 */
                        p2 = a21 * s22 - a23n * s20;
                        k1 = p2 / m2;
                        p2 -= k1 * m2;
                        if (p2 < 0.0){
                            p2 += m2;}
                        s20 = s21;
                        s21 = s22;
                        s22 = p2;

                        /* Combination */
                        if (p1 <= p2) {
                            rand = W5 * ((p1 - p2 + m1) * norm);
                            }
                        else {
                            rand = W5 * ((p1 - p2) * norm);
                            }

                        /* Determine new direction of random walk */
                        if (rand<W0 || rand==0){n=1; o=0; p=0;}
                        else if (rand>=W0 && rand<W1){n=-1; o=0; p=0;}
                        else if (rand>=W1 && rand<W2){n=0; o=1; p=0;}
                        else if (rand>=W2 && rand<W3){n=0; o=-1; p=0;}
                        else if (rand>=W3 && rand<W4){n=0; o=0; p=1;}
                        else if (rand>=W4 && rand<=W5){n=0; o=0; p=-1;}

                        /* Move in new direction */
                        if (k+n>0 && k+n<zsh-1 && l+o>0 && l+o<ysh-1 && m+p>0 && m+p<xsh-1) {
                            k += n;
                            l += o;
                            m += p;
                            position = k*flat + l*xsh + m;
                            atomicAdd(&hits[position], 1);
                            }

                        step += 1;

                        if (step==sorw) {
                            k = plane;
                            l = row;
                            m = column;
                            position = k*flat + l*xsh + m;
                            n_rw += 1;
                            step = 0;
                            }
                        }
                    }
                }
            }
        }
    """
    mod = SourceModule(code)
    kernel = mod.get_function("Funktion")
    return kernel

def _build_kernel_float32():
    code = """

    __device__ float _calc_var(unsigned int position, unsigned int index, float B, float *raw, int segment, int *labels, int xsh) {
        float dev = 0;
        float summe = 0;
        for (int n = -1; n < 2; n++) {
            for (int o = -1; o < 2; o++) {
                if (labels[index + n*xsh + o] == segment) {
                    float tmp = B - raw[position + n*xsh + o];
                    dev += tmp * tmp;
                    summe += 1;
                    }
                }
            }
        float var = dev / summe;
        if (var < 1.0) {
            var = 1.0;
            }
        return var;
        }

    __device__ float weight(float B, float A, float div1) {
        float tmp = B - A;
        return exp( - tmp * tmp * div1 );
        }

    __global__ void Funktion(int segment, float *raw, int *slices, float *hits, int xsh, int ysh, int zsh, int *indices, int sorw, int nbrw) {

        int flat   = xsh * ysh;
        int column = blockIdx.x * blockDim.x + threadIdx.x;
        int row    = blockIdx.y * blockDim.y + threadIdx.y;
        int slice  = blockIdx.z;
        int plane  = indices[slice];
        unsigned int index = slice * flat + row * xsh + column;
        unsigned int position = plane * flat + row * xsh + column;

        if (index < gridDim.z*flat && plane>0 && row>0 && column>0 && plane<zsh-1 && row<ysh-1 && column<xsh-1) {

            if (slices[index]==segment) {

                /* Adaptive random walks */
                int found = 0;
                if ((column + row) % 4 == 0) {
                    found = 1;
                    }
                else {
                    for (int y = -100; y < 101; y++) {
                        for (int x = -100; x < 101; x++) {
                            if (row+y > 0 && column+x > 0 && row+y < ysh-1 && column+x < xsh-1) {
                                unsigned int tmp = slice * flat + (row+y) * xsh + column+x;
                                if (slices[tmp] != segment && slices[tmp] != -1) {
                                    found = 1;
                                    }
                                }
                            }
                        }
                    }

                if (found == 1) {

                    float rand;
                    float W0,W1,W2,W3,W4,W5;
                    int n,o,p;

                    /* Initialize MRG32k3a */
                    float norm = 2.328306549295728e-10;
                    float m1 = 4294967087.0;
                    float m2 = 4294944443.0;
                    float a12 = 1403580.0;
                    float a13n = 810728.0;
                    float a21 = 527612.0;
                    float a23n = 1370589.0;
                    long k1;
                    float p1, p2;
                    float s10 = index, s11 = index, s12 = index, s20 = index, s21 = index, s22 = index;

                    /* Compute standard deviation */
                    float B = raw[position];
                    float var = _calc_var(position, index, B, raw, segment, slices, xsh);
                    float div1 = 1 / (2 * var);

                    int k = plane;
                    int l = row;
                    int m = column;

                    int step = 0;
                    int n_rw = 0;

                    /* Compute random walks */
                    while (n_rw < nbrw) {

                        /* Compute weights */
                        W0 = weight(B, raw[position + flat], div1);
                        W1 = weight(B, raw[position - flat], div1);
                        W2 = weight(B, raw[position + xsh], div1);
                        W3 = weight(B, raw[position - xsh], div1);
                        W4 = weight(B, raw[position + 1], div1);
                        W5 = weight(B, raw[position - 1], div1);

                        W1 += W0;
                        W2 += W1;
                        W3 += W2;
                        W4 += W3;
                        W5 += W4;

                        /* Compute random numbers with MRG32k3a */

                        /* Component 1 */
                        p1 = a12 * s11 - a13n * s10;
                        k1 = p1 / m1;
                        p1 -= k1 * m1;
                        if (p1 < 0.0){
                            p1 += m1;}
                        s10 = s11;
                        s11 = s12;
                        s12 = p1;

                        /* Component 2 */
                        p2 = a21 * s22 - a23n * s20;
                        k1 = p2 / m2;
                        p2 -= k1 * m2;
                        if (p2 < 0.0){
                            p2 += m2;}
                        s20 = s21;
                        s21 = s22;
                        s22 = p2;

                        /* Combination */
                        if (p1 <= p2) {
                            rand = W5 * ((p1 - p2 + m1) * norm);
                            }
                        else {
                            rand = W5 * ((p1 - p2) * norm);
                            }

                        /* Determine new direction of random walk */
                        if (rand<W0 || rand==0){n=1; o=0; p=0;}
                        else if (rand>=W0 && rand<W1){n=-1; o=0; p=0;}
                        else if (rand>=W1 && rand<W2){n=0; o=1; p=0;}
                        else if (rand>=W2 && rand<W3){n=0; o=-1; p=0;}
                        else if (rand>=W3 && rand<W4){n=0; o=0; p=1;}
                        else if (rand>=W4 && rand<=W5){n=0; o=0; p=-1;}

                        /* Move in new direction */
                        if (k+n>0 && k+n<zsh-1 && l+o>0 && l+o<ysh-1 && m+p>0 && m+p<xsh-1) {
                            k += n;
                            l += o;
                            m += p;
                            position = k*flat + l*xsh + m;
                            atomicAdd(&hits[position], 1);
                            }

                        step += 1;

                        if (step==sorw) {
                            k = plane;
                            l = row;
                            m = column;
                            position = k*flat + l*xsh + m;
                            n_rw += 1;
                            step = 0;
                            }
                        }
                    }
                }
            }
        }
    """
    mod = SourceModule(code)
    kernel = mod.get_function("Funktion")
    return kernel

