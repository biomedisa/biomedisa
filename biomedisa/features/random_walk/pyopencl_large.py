##########################################################################
##                                                                      ##
##  Copyright (c) 2019 Philipp Lösel. All rights reserved.              ##
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
import pyopencl as cl
import pyopencl.array
from biomedisa.features.random_walk.opencl_kernels import (_build_kernel_uncertainty, 
    _build_kernel_max, _build_update_gpu, _build_curvature_gpu)

def reduceBlocksize(slices):
    zsh, ysh, xsh = slices.shape
    argmin_x, argmax_x, argmin_y, argmax_y = xsh, 0, ysh, 0
    for k in range(zsh):
        y, x = np.nonzero(slices[k])
        if x.any():
            argmin_x = min(argmin_x, int(np.amin(x)))
            argmax_x = max(argmax_x, int(np.amax(x)))
            argmin_y = min(argmin_y, int(np.amin(y)))
            argmax_y = max(argmax_y, int(np.amax(y)))
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
            comm.Send([send, MPI.INT], dest=dest, tag=2)
        else:
            sendbuf.fill(0)
            comm.Send([sendbuf, MPI.INT], dest=dest, tag=0)

        # recv block
        source = rank+1
        comm.Recv([recvbuf, MPI.INT], source=source, tag=3)
        if recvbuf:
            data_z, data_y, data_x = comm.recv(source=source, tag=4)
            recv = np.empty((data_z, data_y, data_x), dtype=np.int32)
            comm.Recv([recv, MPI.INT], source=source, tag=5)
            a[blockmax-data_z:blockmax] += recv

    elif rank == size-1:

        if rank % 2 == 1: add = 0
        if rank % 2 == 0: add = 6

        # recv block
        source = rank-1
        comm.Recv([recvbuf, MPI.INT], source=source, tag=0+add)
        if recvbuf:
            data_z, data_y, data_x = comm.recv(source=source, tag=1+add)
            recv = np.empty((data_z, data_y, data_x), dtype=np.int32)
            comm.Recv([recv, MPI.INT], source=source, tag=2+add)
            a[blockmin:blockmin+data_z] += recv

        # send block
        dest = rank-1
        tmp = a[:blockmin]
        if np.any(tmp):
            sendbuf.fill(1)
            comm.Send([sendbuf, MPI.INT], dest=dest, tag=3+add)
            send = tmp.copy(order='C')
            comm.send([send.shape[0], send.shape[1], send.shape[2]], dest=dest, tag=4+add)
            comm.Send([send, MPI.INT], dest=dest, tag=5+add)
        else:
            sendbuf.fill(0)
            comm.Send([sendbuf, MPI.INT], dest=dest, tag=3+add)

    elif rank % 2 == 1:

        # recv block
        source = rank-1
        comm.Recv([recvbuf, MPI.INT], source=source, tag=0)
        if recvbuf:
            data_z, data_y, data_x = comm.recv(source=source, tag=1)
            recv = np.empty((data_z, data_y, data_x), dtype=np.int32)
            comm.Recv([recv, MPI.INT], source=source, tag=2)
            a[blockmin:blockmin+data_z] += recv

        # send block
        dest = rank-1
        tmp = a[:blockmin]
        if np.any(tmp):
            sendbuf.fill(1)
            comm.Send([sendbuf, MPI.INT], dest=dest, tag=3)
            send = tmp.copy(order='C')
            comm.send([send.shape[0], send.shape[1], send.shape[2]], dest=dest, tag=4)
            comm.Send([send, MPI.INT], dest=dest, tag=5)
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
            comm.Send([send, MPI.INT], dest=dest, tag=8)
        else:
            sendbuf.fill(0)
            comm.Send([sendbuf, MPI.INT], dest=dest, tag=6)

        # recv block
        source = rank+1
        comm.Recv([recvbuf, MPI.INT], source=source, tag=9)
        if recvbuf:
            data_z, data_y, data_x = comm.recv(source=source, tag=10)
            recv = np.empty((data_z, data_y, data_x), dtype=np.int32)
            comm.Recv([recv, MPI.INT], source=source, tag=11)
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
            comm.Send([send, MPI.INT], dest=dest, tag=2)
        else:
            sendbuf.fill(0)
            comm.Send([sendbuf, MPI.INT], dest=dest, tag=0)

        # recv block
        source = rank+1
        comm.Recv([recvbuf, MPI.INT], source=source, tag=3)
        if recvbuf:
            data_z, data_y, data_x = comm.recv(source=source, tag=4)
            recv = np.empty((data_z, data_y, data_x), dtype=np.int32)
            comm.Recv([recv, MPI.INT], source=source, tag=5)
            a[blockmax-data_z:blockmax] += recv

        # recv block
        source = rank-1
        comm.Recv([recvbuf, MPI.INT], source=source, tag=6)
        if recvbuf:
            data_z, data_y, data_x = comm.recv(source=source, tag=7)
            recv = np.empty((data_z, data_y, data_x), dtype=np.int32)
            comm.Recv([recv, MPI.INT], source=source, tag=8)
            a[blockmin:blockmin+data_z] += recv

        # send block
        dest = rank-1
        tmp = a[:blockmin]
        if np.any(tmp):
            sendbuf.fill(1)
            comm.Send([sendbuf, MPI.INT], dest=dest, tag=9)
            send = tmp.copy(order='C')
            comm.send([send.shape[0], send.shape[1], send.shape[2]], dest=dest, tag=10)
            comm.Send([send, MPI.INT], dest=dest, tag=11)
        else:
            sendbuf.fill(0)
            comm.Send([sendbuf, MPI.INT], dest=dest, tag=9)

    return a

@numba.jit(nopython=True)
def max_to_label(a, walkmap, final, blockmin, blockmax, segment):
    zsh, ysh, xsh = a.shape
    zmin, zmax = 0, zsh
    ymin, ymax = 0, ysh
    xmin, xmax = 0, xsh
    zmin, zmax = blockmin, blockmax
    for k in range(zmin, zmax):
        for l in range(ymin, ymax):
            for m in range(xmin, xmax):
                if a[k,l,m] > walkmap[k,l,m]:
                    walkmap[k,l,m] = a[k,l,m]
                    final[k-zmin,l-ymin,m-xmin] = segment
    return walkmap, final

def walk(comm, raw, extracted_slices, indices, nbrw, sorw, blockmin, blockmax,
         name, allLabels, smooth, uncertainty, ctx, queue, platform, allx):

    # get rank and size of mpi process
    rank = comm.Get_rank()
    size = comm.Get_size()

    # build kernels
    if raw.dtype == 'uint8':
        kernel = cl.Program(ctx, _build_kernel_int8()).build().randomWalk
        raw = (raw-128).astype('int8')
    else:   
        kernel = cl.Program(ctx, _build_kernel_float32()).build().randomWalk
        raw = raw.astype(np.float32)

    # image size
    zsh, ysh, xsh = raw.shape

    # block and grid size
    global_size = (xsh, ysh, zsh)
    local_size = None

    # rebuild slices
    slices = np.zeros(raw.shape, dtype=np.int32) - 1
    if allx:
        slices[indices[0]] = extracted_slices[0]
        slices[:,indices[1]] = extracted_slices[1].transpose(1, 0, 2)
        slices[:,:,indices[2]] = extracted_slices[2].transpose(1, 2, 0)
    else:
        slices[indices] = extracted_slices

    # crop to region of interest
    slices = reduceBlocksize(slices)

    # allocate host memory
    hits = np.empty(raw.shape, dtype=np.int32)
    final = np.zeros((blockmax-blockmin, ysh, xsh), dtype=np.uint8)

    # allocate memory on the device
    mf = cl.mem_flags

    # allocate device memory or use subdomains
    memory_error = False
    subdomains = False
    if zsh * ysh * xsh > 42e8:
        print('Warning: Volume indexes exceed unsigned long int range. The volume is splitted into subdomains.')
        subdomains = True
    else:
        try:
            slices_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=slices)
            raw_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=raw)
            hits_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=hits)
        except Exception as e:
            print('Warning: Device ran out of memory. The volume is splitted into subdomains.')
            subdomains = True
            try:
                raw_cl.release()
                hits_cl.release()
                slices_cl.release()
            except:
                pass

    # disable smoothing and uncertainty for subdomains
    if subdomains:
        sendbuf = np.zeros(1, dtype=np.int32) + 1
        recvbuf = np.zeros(1, dtype=np.int32)
    else:
        sendbuf = np.zeros(1, dtype=np.int32)
        recvbuf = np.zeros(1, dtype=np.int32)
    comm.Barrier()
    comm.Allreduce([sendbuf, MPI.INT], [recvbuf, MPI.INT], op=MPI.MAX)
    if recvbuf > 0:
        smooth, uncertainty = 0, False

    if smooth:
        try:
            update_gpu = _build_update_gpu(ctx)
            curvature_gpu = _build_curvature_gpu(ctx)
            b_npy = np.zeros(raw.shape, dtype=np.float32)
            b_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=b_npy)
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
                b_cl.release()
            except:
                pass

    if uncertainty:
        try:
            kernel_uncertainty = _build_kernel_uncertainty(ctx)
            kernel_max = _build_kernel_max(ctx)
            max_npy = np.zeros((3,)+raw.shape, dtype=np.float32)
            max_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=max_npy)
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
                max_cl.release()
            except:
                pass

    for label_counter, segment in enumerate(allLabels):
        print('%s:' %(name) + ' ' + str(label_counter+1) + '/' + str(len(allLabels)))

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

                    # allocate memory and compute random walks on subdomain
                    sub_slices = slices[data_block_min:data_block_max].copy()

                    sub_slices[:sub_block_min-data_block_min] = -1
                    sub_slices[sub_block_max-data_block_min:] = -1
                    sub_slices_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sub_slices)

                    sub_zsh = data_block_max - data_block_min

                    sub_raw = raw[data_block_min:data_block_max].copy()
                    sub_raw_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sub_raw)

                    sub_hits = np.empty(sub_raw.shape, dtype=np.int32)
                    sub_hits_cl = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=sub_hits)
                    cl.enqueue_fill_buffer(queue, sub_hits_cl, np.int32(0), offset=0, size=sub_hits.nbytes)

                    block = None
                    grid = (sub_zsh, ysh, xsh)
                    kernel(queue, grid, block, np.int32(segment), sub_raw_cl, sub_slices_cl, sub_hits_cl,
                        np.int32(xsh), np.int32(ysh), np.int32(sub_zsh), np.int32(sorw), np.int32(nbrw))
                    cl.enqueue_copy(queue, sub_hits, sub_hits_cl)
                    hits[data_block_min:data_block_max] += sub_hits

                    sub_slices_cl.release()
                    sub_hits_cl.release()
                    sub_raw_cl.release()
            except Exception as e:
                print('Error: Device out of memory. Data too large.')
                memory_error = True
                try:
                    sub_slices_cl.release()
                    sub_hits_cl.release()
                    sub_raw_cl.release()
                except:
                    pass

        else:
            # compute random walks on the entire volume
            block = None
            grid = (zsh, ysh, xsh)
            cl.enqueue_fill_buffer(queue, hits_cl, np.int32(0), offset=0, size=hits.nbytes)
            kernel(queue, grid, block, np.int32(segment), raw_cl, slices_cl, hits_cl,
                np.int32(xsh), np.int32(ysh), np.int32(zsh), np.int32(sorw), np.int32(nbrw))
            cl.enqueue_copy(queue, hits, hits_cl)

        # memory error
        if memory_error:
            sendbuf = np.zeros(1, dtype=np.int32) + 1
            recvbuf = np.zeros(1, dtype=np.int32)
        else:
            sendbuf = np.zeros(1, dtype=np.int32)
            recvbuf = np.zeros(1, dtype=np.int32)
        comm.Barrier()
        comm.Allreduce([sendbuf, MPI.INT], [recvbuf, MPI.INT], op=MPI.MAX)
        if recvbuf > 0:
            memory_error = True
            try:
                slices_cl.release()
                hits_cl.release()
                raw_cl.release()
            except:
                pass
            return memory_error, None, None, None, None

        # communicate hits
        if size > 1:
            hits = sendrecv(hits, blockmin, blockmax, comm, rank, size)

        # convert hits buffer to float32 and back to int32
        if uncertainty or smooth:
            hits = hits.astype(np.float32)
            cl.enqueue_copy(queue, hits_cl, hits)
            hits = hits.astype(np.int32)

        # save the three most occuring hits
        if uncertainty:
            kernel_max(queue, global_size, local_size, max_cl, hits_cl, np.int32(xsh), np.int32(ysh))

        # smooth manifold
        if smooth:
            for _ in range(smooth):
              curvature_gpu(queue, global_size, local_size, hits_cl, b_cl, np.int32(xsh), np.int32(ysh))
              update_gpu(queue, global_size, local_size, hits_cl, b_cl, np.int32(xsh), np.int32(ysh))
            hits_smooth =  np.empty((zsh, ysh, xsh), dtype=np.float32)
            cl.enqueue_copy(queue, hits_smooth, hits_cl)
            if label_counter == 0:
                hits_smooth[hits_smooth<0] = 0
                walkmap_smooth = hits_smooth.copy()
            else:
                walkmap_smooth, final_smooth = max_to_label(hits_smooth, walkmap_smooth, final_smooth, blockmin, blockmax, segment)

        # get the label with the most hits
        if label_counter == 0:
            walkmap = hits.copy()
        else:
            walkmap, final = max_to_label(hits, walkmap, final, blockmin, blockmax, segment)

    # compute uncertainty
    if uncertainty:
        kernel_uncertainty(queue, global_size, local_size, max_cl, hits_cl, np.int32(xsh), np.int32(ysh))
        final_uncertainty =  np.empty((zsh, ysh, xsh), dtype=np.float32)
        cl.enqueue_copy(queue, final_uncertainty, hits_cl)
        final_uncertainty = final_uncertainty[blockmin:blockmax]
    else:
        final_uncertainty = None

    if not smooth:
        final_smooth = None

    try:
        slices_cl.release()
        hits_cl.release()
        raw_cl.release()
    except:
        pass

    return memory_error, final, final_uncertainty, final_smooth, None

def _build_kernel_int8():
    src = '''

    float _calc_var(unsigned int index, int B, __global char *raw, const int segment, __global int *labels, const int xsh, const int ysh) {
        float dev = 0;
        float summe = 0;
        for (int m = -1; m < 2; m++) {
          for (int n = -1; n < 2; n++) {
            for (int o = -1; o < 2; o++) {
                if (labels[index + m*xsh*ysh + n*xsh + o] == segment) {
                    float tmp = B - raw[index + m*xsh*ysh + n*xsh + o];
                    dev += tmp * tmp;
                    summe += 1;
                    }
                }
              }
            }
        float var = dev / summe;
        if (var < 1.0) {
            var = 1.0;
            }
        return var;
        }

    float weight(int B, int A, float div1) {
        int tmp = B - A;
        return exp( - tmp * tmp * div1 );
        }

    __kernel void randomWalk(const int segment, __global char *raw, __global int *slices, __global int *hits, const int xsh, const int ysh, const int zsh, const int sorw, const int nbrw) {

        // get_global_id(0)         // blockIdx.z * blockDim.z + threadIdx.z
        // get_local_id(0)          // threadIdx.z
        // get_global_size(0)       // gridDim.z * blockDim.z
        // get_local_size(0)        // blockDim.z

        int flat   = xsh * ysh;
        int column = get_global_id(2);
        int row    = get_global_id(1);
        int plane  = get_global_id(0);
        unsigned int index  = plane * flat + row * xsh + column;

        if (index < get_global_size(0)*flat && plane>0 && row>0 && column>0 && plane<zsh-1 && row<ysh-1 && column<xsh-1) {

            if (slices[index]==segment) {

                /* Adaptive random walks */
                int found = 0;
                if ((column + plane) % 2 == 0 && (row + plane) % 2 == 0) {
                    found = 1;
                    }
                else {
                  for (int z = -100; z < 101; z=z+5) {
                    for (int y = -100; y < 101; y=y+5) {
                        for (int x = -100; x < 101; x=x+5) {
                            if (plane+z>0 && row+y>0 && column+x>0 && plane+z<zsh-1 && row+y<ysh-1 && column+x<xsh-1) {
                                unsigned int tmp_idx = (plane+z) * flat + (row+y) * xsh + column+x;
                                if (slices[tmp_idx] != segment && slices[tmp_idx] != -1) {
                                    found = 1;
                                    }
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
                    int B = raw[index];
                    float var = _calc_var(index, B, raw, segment, slices, xsh, ysh);
                    float div1 = 1 / (2 * var);

                    int k = plane;
                    int l = row;
                    int m = column;

                    int step = 0;
                    int n_rw = 0;

                    /* Compute random walks */
                    while (n_rw < nbrw) {

                        /* Compute weights */
                        W0 = weight(B, raw[index + flat], div1);
                        W1 = weight(B, raw[index - flat], div1);
                        W2 = weight(B, raw[index + xsh], div1);
                        W3 = weight(B, raw[index - xsh], div1);
                        W4 = weight(B, raw[index + 1], div1);
                        W5 = weight(B, raw[index - 1], div1);

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
                            index = k*flat + l*xsh + m;
                            atomic_add(&hits[index], 1);
                            }

                        step += 1;

                        if (step==sorw) {
                            k = plane;
                            l = row;
                            m = column;
                            index = k*flat + l*xsh + m;
                            n_rw += 1;
                            step = 0;
                            }
                        }
                    }
                }
            }
        }
    '''
    return src

def _build_kernel_float32():
    src = '''

    float _calc_var(unsigned int index, float B, __global float *raw, const int segment, __global int *labels, const int xsh, const int ysh) {
        float dev = 0;
        float summe = 0;
        for (int m = -1; m < 2; m++) {
          for (int n = -1; n < 2; n++) {
            for (int o = -1; o < 2; o++) {
                if (labels[index + m*ysh*xsh + n*xsh + o] == segment) {
                    float tmp = B - raw[index + m*ysh*xsh + n*xsh + o];
                    dev += tmp * tmp;
                    summe += 1;
                    }
                }
              }
            }
        float var = dev / summe;
        if (var < 1.0) {
            var = 1.0;
            }
        return var;
        }

    float weight(float B, float A, float div1) {
        float tmp = B - A;
        return exp( - tmp * tmp * div1 );
        }

    __kernel void randomWalk(const int segment, __global float *raw, __global int *slices, __global int *hits, const int xsh, const int ysh, const int zsh, const int sorw, const int nbrw) {

        // get_global_id(0)         // blockIdx.z * blockDim.z + threadIdx.z
        // get_local_id(0)          // threadIdx.z
        // get_global_size(0)       // gridDim.z * blockDim.z
        // get_local_size(0)        // blockDim.z

        int flat   = xsh * ysh;
        int column = get_global_id(2);
        int row    = get_global_id(1);
        int plane  = get_global_id(0);
        unsigned int index  = plane * flat + row * xsh + column;

        if (index < get_global_size(0)*flat && plane>0 && row>0 && column>0 && plane<zsh-1 && row<ysh-1 && column<xsh-1) {

            if (slices[index]==segment) {

                /* Adaptive random walks */
                int found = 0;
                if ((column + plane) % 2 == 0 && (row + plane) % 2 == 0) {
                    found = 1;
                    }
                else {
                  for (int z = -100; z < 101; z=z+5) {
                    for (int y = -100; y < 101; y=y+5) {
                        for (int x = -100; x < 101; x=x+5) {
                            if (plane+z>0 && row+y>0 && column+x>0 && plane+z<zsh-1 && row+y<ysh-1 && column+x<xsh-1) {
                                unsigned int tmp_idx = (plane+z) * flat + (row+y) * xsh + column+x;
                                if (slices[tmp_idx] != segment && slices[tmp_idx] != -1) {
                                    found = 1;
                                    }
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
                    float B = raw[index];
                    float var = _calc_var(index, B, raw, segment, slices, xsh, ysh);
                    float div1 = 1 / (2 * var);

                    int k = plane;
                    int l = row;
                    int m = column;

                    int step = 0;
                    int n_rw = 0;

                    /* Compute random walks */
                    while (n_rw < nbrw) {

                        /* Compute weights */
                        W0 = weight(B, raw[index + flat], div1);
                        W1 = weight(B, raw[index - flat], div1);
                        W2 = weight(B, raw[index + xsh], div1);
                        W3 = weight(B, raw[index - xsh], div1);
                        W4 = weight(B, raw[index + 1], div1);
                        W5 = weight(B, raw[index - 1], div1);

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
                            index = k*flat + l*xsh + m;
                            atomic_add(&hits[index], 1);
                            }

                        step += 1;

                        if (step==sorw) {
                            k = plane;
                            l = row;
                            m = column;
                            index = k*flat + l*xsh + m;
                            n_rw += 1;
                            step = 0;
                            }
                        }
                    }
                }
            }
        }
    '''
    return src

