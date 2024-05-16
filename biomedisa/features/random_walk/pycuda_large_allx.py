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
    testSlices = np.copy(slices, order='C')
    testSlices[testSlices==-1] = 0
    zsh, ysh, xsh = slices.shape
    argmin_x, argmax_x, argmin_y, argmax_y = xsh, 0, ysh, 0
    for k in range(zsh):
        y, x = np.nonzero(testSlices[k])
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
            limit = min(a.shape[0], blockmin+data_z) - blockmin
            a[blockmin:blockmin+data_z] += recv[:limit]

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
def _calc_var(raw, A):
    ysh, xsh = raw.shape
    beta = np.zeros((ysh, xsh))
    for l in range(1, ysh-1):
        for m in range(1, xsh-1):
            if A[l, m] == 1:
                dev, summe = 0, 0
                B = raw[l, m]
                for n in range(-1, 2):
                    for o in range(-1, 2):
                        if A[l+n, m+o] == 1:
                            dev += (B - raw[l+n, m+o])**2
                            summe += 1
                var = dev / summe
                if var < 1.0:
                    beta[l, m] = 1.0
                else:
                    beta[l, m] = var
    return beta

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

def _calc_label_walking_area(sliceData, labelValue):
    walkingArea = np.zeros_like(sliceData)
    walkingArea[sliceData == labelValue] = 1
    return walkingArea

def walk(comm, raw, slices, indices, nbrw, sorw, blockmin, blockmax, name,
         allLabels, smooth, uncertainty, ctx, queue, platform):

    rank = comm.Get_rank()
    size = comm.Get_size()

    if raw.dtype == 'uint8':
        kernel = _build_kernel_int8()
        raw = (raw-128).astype('int8')
    else:
        kernel = _build_kernel_float32()
        raw = raw.astype(np.float32)

    foundAxis = [0] * 3
    for k in range(3):
        if indices[k]:
            foundAxis[k] = 1

    zsh, ysh, xsh = raw.shape
    fill_gpu = _build_kernel_fill()

    block = (32, 32, 1)
    x_grid = (xsh // 32) + 1
    y_grid = (ysh // 32) + 1
    grid2 = (int(x_grid), int(y_grid), int(zsh))

    a = np.empty(raw.shape, dtype=np.float32)
    final = np.zeros((blockmax-blockmin, ysh, xsh), dtype=np.uint8)
    segment_npy = np.empty(1, dtype=np.uint8)

    memory_error = False

    try:
        raw_gpu = gpuarray.to_gpu(raw)
        a_gpu = cuda.mem_alloc(a.nbytes)

        if smooth:
            update_gpu = _build_update_gpu()
            curvature_gpu = _build_curvature_gpu()
            b_gpu = gpuarray.zeros(raw.shape, dtype=np.float32)

        zshape = np.int32(zsh)
        yshape = np.int32(ysh)
        xshape = np.int32(xsh)
        sorw = np.int32(sorw)
        nbrw = np.int32(nbrw)

        slshape = [None] * 3
        indices_gpu = [None] * 3
        beta_gpu = [None] * 3
        slices_gpu = [None] * 3
        ysh = [None] * 3
        xsh = [None] * 3

        for k, found in enumerate(foundAxis):
            if found:
                indices_tmp = np.array(indices[k], dtype=np.int32)
                slices_tmp = slices[k].astype(np.int32)
                slices_tmp = reduceBlocksize(slices_tmp)
                slshape[k], ysh[k], xsh[k] = slices_tmp.shape
                indices_gpu[k] = gpuarray.to_gpu(indices_tmp)
                slices_gpu[k] = gpuarray.to_gpu(slices_tmp)
                Beta = np.zeros(slices_tmp.shape, dtype=np.float32)
                for m in range(slshape[k]):
                    for n in allLabels:
                        A = _calc_label_walking_area(slices_tmp[m], n)
                        plane = indices_tmp[m]
                        if k==0: raw_tmp = raw[plane]
                        if k==1: raw_tmp = raw[:,plane]
                        if k==2: raw_tmp = raw[:,:,plane]
                        Beta[m] += _calc_var(raw_tmp.astype(float), A)
                beta_gpu[k] = gpuarray.to_gpu(Beta)

        sendbuf = np.zeros(1, dtype=np.int32)
        recvbuf = np.zeros(1, dtype=np.int32)
        comm.Barrier()
        comm.Allreduce([sendbuf, MPI.INT], [recvbuf, MPI.INT], op=MPI.MAX)

    except Exception as e:
        print('Error: GPU out of memory. Data too large.')
        sendbuf = np.zeros(1, dtype=np.int32) + 1
        recvbuf = np.zeros(1, dtype=np.int32)
        comm.Barrier()
        comm.Allreduce([sendbuf, MPI.INT], [recvbuf, MPI.INT], op=MPI.MAX)

    if recvbuf > 0:
        memory_error = True
        try:
            a_gpu.free()
        except:
            pass
        return memory_error, None, None, None

    if smooth:
        try:
            update_gpu = _build_update_gpu()
            curvature_gpu = _build_curvature_gpu()
            b_npy = np.zeros(raw.shape, dtype=np.float32)
            b_gpu = cuda.mem_alloc(b_npy.nbytes)
            cuda.memcpy_htod(b_gpu, b_npy)
            final_smooth = np.zeros((blockmax-blockmin, yshape, xshape), dtype=np.uint8)
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
        fill_gpu(a_gpu, xshape, yshape, block=block, grid=grid2)
        segment_gpu = np.int32(segment)
        segment_npy.fill(segment)
        for k, found in enumerate(foundAxis):
            if found:
                axis_gpu = np.int32(k)
                x_grid = (xsh[k] // 32) + 1
                y_grid = (ysh[k] // 32) + 1
                grid = (int(x_grid), int(y_grid), int(slshape[k]))
                kernel(axis_gpu, segment_gpu, raw_gpu, slices_gpu[k], a_gpu, xshape, yshape, zshape, indices_gpu[k], sorw, beta_gpu[k], nbrw, block=block, grid=grid)
        cuda.memcpy_dtoh(a, a_gpu)

        if size > 1:
            a = sendrecv(a, blockmin, blockmax, comm, rank, size)

        if smooth or uncertainty:
            cuda.memcpy_htod(a_gpu, a)

        if uncertainty:
            kernel_max(max_gpu, a_gpu, xshape, yshape, block=block, grid=grid2)

        if smooth:
            for k in range(smooth):
                curvature_gpu(a_gpu, b_gpu, xshape, yshape, block=block, grid=grid2)
                update_gpu(a_gpu, b_gpu, xshape, yshape, block=block, grid=grid2)
            a_smooth = np.empty_like(a)
            cuda.memcpy_dtoh(a_smooth, a_gpu)
            if label_counter == 0:
                a_smooth[a_smooth<0] = 0
                walkmap_smooth = np.copy(a_smooth, order='C')
            else:
                walkmap_smooth, final_smooth = max_to_label(a_smooth, walkmap_smooth, final_smooth, blockmin, blockmax, segment)

        if label_counter == 0:
            a[a<0] = 0
            walkmap = np.copy(a, order='C')
        else:
            walkmap, final = max_to_label(a, walkmap, final, blockmin, blockmax, segment)

    if uncertainty:
        kernel_uncertainty(max_gpu, a_gpu, xshape, yshape, block=block, grid=grid2)
        final_uncertainty = np.empty_like(a)
        cuda.memcpy_dtoh(final_uncertainty, a_gpu)
        final_uncertainty = final_uncertainty[blockmin:blockmax]
    else:
        final_uncertainty = None

    if not smooth:
        final_smooth = None

    try:
        a_gpu.free()
    except:
        pass

    return memory_error, final, final_uncertainty, final_smooth

def _build_kernel_int8():
    code = """

    __device__ float weight(float B, float *raw, float div1, unsigned int position) {
        float tmp = B - (float)(*((char*)(raw) + position));
        return exp( - tmp * tmp * div1 );
        }

    __global__ void Funktion(int axis, int segment, float *raw, int *slices, float *a, int xsh, int ysh, int zsh, int *indices, int sorw, float *Beta, int nbrw) {

        int col_g = blockIdx.x * blockDim.x + threadIdx.x;
        int row_g = blockIdx.y * blockDim.y + threadIdx.y;
        int slc_g = blockIdx.z;

        int xsh_g, ysh_g, plane, row, column;

        if (axis == 0) {
            plane  = indices[slc_g];
            row    = row_g;
            column = col_g;
            xsh_g  = xsh;
            ysh_g  = ysh;
            }
        else if (axis == 1) {
            row    = indices[slc_g];
            plane  = row_g;
            column = col_g;
            xsh_g  = xsh;
            ysh_g  = zsh;
            }
        else if (axis == 2) {
            column = indices[slc_g];
            plane  = row_g;
            row    = col_g;
            xsh_g  = ysh;
            ysh_g  = zsh;
            }

        int flat   = xsh * ysh;
        int flat_g = xsh_g * ysh_g;
        unsigned int index = slc_g * flat_g + row_g * xsh_g + col_g;

        if (index<gridDim.z*flat_g && plane>0 && plane<zsh-1 && row>0 && row<ysh-1 && column>0 && column<xsh-1) {

            if (slices[index]==segment) {

                /* Adaptive random walks */
                int found = 0;
                if ((col_g + row_g) % 4 == 0) {
                    found = 1;
                    }
                else {
                    for (int y = -100; y < 101; y++) {
                        for (int x = -100; x < 101; x++) {
                            if (row_g+y > 0 && col_g+x > 0 && row_g+y < ysh_g-1 && col_g+x < xsh_g-1) {
                                unsigned int tmp = slc_g * flat_g + (row_g + y) * xsh_g + (col_g + x);
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
                    unsigned int position = plane*flat + row*xsh + column;
                    float B = (float)(*((char*)(raw) + position));
                    float var = Beta[index];
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
                            atomicAdd(&a[position], 1);
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

    __device__ float weight(float B, float A, float div1) {
        float tmp = B - A;
        return exp( - tmp * tmp * div1 );
        }

    __global__ void Funktion(int axis, int segment, float *raw, int *slices, float *a, int xsh, int ysh, int zsh, int *indices, int sorw, float *Beta, int nbrw) {

        int col_g = blockIdx.x * blockDim.x + threadIdx.x;
        int row_g = blockIdx.y * blockDim.y + threadIdx.y;
        int slc_g = blockIdx.z;

        int xsh_g, ysh_g, plane, row, column;

        if (axis == 0) {
            plane  = indices[slc_g];
            row    = row_g;
            column = col_g;
            xsh_g  = xsh;
            ysh_g  = ysh;
            }
        else if (axis == 1) {
            row    = indices[slc_g];
            plane  = row_g;
            column = col_g;
            xsh_g  = xsh;
            ysh_g  = zsh;
            }
        else if (axis == 2) {
            column = indices[slc_g];
            plane  = row_g;
            row    = col_g;
            xsh_g  = ysh;
            ysh_g  = zsh;
            }

        int flat   = xsh * ysh;
        int flat_g = xsh_g * ysh_g;
        unsigned int index = slc_g * flat_g + row_g * xsh_g + col_g;

        if (index<gridDim.z*flat_g && plane>0 && plane<zsh-1 && row>0 && row<ysh-1 && column>0 && column<xsh-1) {

            if (slices[index]==segment) {

                /* Adaptive random walks */
                int found = 0;
                if ((col_g + row_g) % 4 == 0) {
                    found = 1;
                    }
                else {
                    for (int y = -100; y < 101; y++) {
                        for (int x = -100; x < 101; x++) {
                            if (row_g+y > 0 && col_g+x > 0 && row_g+y < ysh_g-1 && col_g+x < xsh_g-1) {
                                unsigned int tmp = slc_g * flat_g + (row_g + y) * xsh_g + (col_g + x);
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
                    unsigned int position = plane*flat + row*xsh + column;
                    float B = raw[position];
                    float var = Beta[index];
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
                            atomicAdd(&a[position], 1);
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

