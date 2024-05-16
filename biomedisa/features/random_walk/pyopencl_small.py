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

import numpy as np
import pyopencl as cl
import os

def walk(data, slices, indices, indices_child, nbrw, sorw, name, ctx, queue):

    labels = np.unique(slices)
    slicesChunk = _extract_slices(slices, indices, indices_child)
    labelsChunk = np.unique(slicesChunk)

    # remove negative labels from list
    index = np.argwhere(labels<0)
    labels = np.delete(labels, index)
    index = np.argwhere(labelsChunk<0)
    labelsChunk = np.delete(labelsChunk, index)

    walkmapChunk = _walk_on_current_gpu(data, slicesChunk, labelsChunk, indices_child, nbrw, sorw, name, ctx, queue)

    if walkmapChunk.shape[0] != len(labels):
        walkmap = np.zeros((len(labels),)+data.shape, dtype=np.float32)
        chunk2Walkmap = np.nonzero(np.in1d(labels, labelsChunk))[0]
        for chunkIndex, walkmapIndex in enumerate(chunk2Walkmap):
            walkmap[walkmapIndex] += walkmapChunk[chunkIndex]
    else:
        walkmap = walkmapChunk

    return walkmap
 
def _extract_slices(slices, indices, indicesChunk):
    extracted = np.zeros((0, slices.shape[1], slices.shape[2]), dtype=np.int32)
    slicesIndicesToExtract = np.nonzero(np.in1d(indices, indicesChunk))[0]
    for arraySliceIndex in slicesIndicesToExtract:
        extracted = np.append(extracted, [slices[arraySliceIndex]], axis=0)
    return extracted

def _walk_on_current_gpu(raw, slices, allLabels, indices, nbrw, sorw, name, ctx, queue):

    # build kernels
    if raw.dtype == 'uint8':
        src = _build_kernel_int8()
        raw = (raw-128).astype('int8')
    else:
        src = _build_kernel_float32()
        raw = raw.astype(np.float32)

    indices = np.array(indices, dtype=np.int32)
    slices = np.array(slices, dtype=np.int32)
    walkmap = np.zeros((len(allLabels),)+raw.shape, dtype=np.float32)
    a = np.empty(raw.shape, dtype=np.int32)

    # image size
    zsh, ysh, xsh = raw.shape
    slshape = slices.shape[0]

    # kernel function instantiation
    mf = cl.mem_flags
    prg = cl.Program(ctx, src).build()

    # allocate memory for variables on the device
    indices_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=indices)
    slices_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=slices)
    raw_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=raw)
    a_cl = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=a)

    xsh_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(xsh))
    ysh_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(ysh))
    zsh_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(zsh))
    sorw_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(sorw))
    nbrw_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(nbrw))
    segment_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(0))

    # block and grid size
    #block = (1, 32, 32)
    #x_grid = ((xsh // 32) + 1)*32
    #y_grid = ((ysh // 32) + 1)*32
    #grid = (slshape, y_grid, x_grid)

    block = None
    grid = (slshape, ysh, xsh)

    # call Kernel
    for label_counter, segment in enumerate(allLabels):
        print('%s:' %(name) + ' ' + str(label_counter+1) + '/' + str(len(allLabels)))
        cl.enqueue_fill_buffer(queue, a_cl, np.int32(0), offset=0, size=a.nbytes)
        cl.enqueue_fill_buffer(queue, segment_cl, np.int32(segment), offset=0, size=4, wait_for=None)
        prg.randomWalk(queue, grid, block, segment_cl, raw_cl, slices_cl, a_cl, xsh_cl, ysh_cl, zsh_cl, indices_cl, sorw_cl, nbrw_cl)
        cl.enqueue_copy(queue, a, a_cl)
        walkmap[label_counter] += a
    return walkmap

def _build_kernel_int8():
    src = '''

    float _calc_var(int position, int index, int B, __global char *raw, int segment, __global int *labels, int xsh) {
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

    float weight(int B, int A, float div1) {
        int tmp = B - A;
        return exp( - tmp * tmp * div1 );
        }

    __kernel void randomWalk(__global int *Segment, __global char *raw, __global int *slices, __global int *a, __global int *Xsh, __global int *Ysh, __global int *Zsh, __global int *indices, __global int *Sorw, __global int *Nbrw) {

        int sorw = *Sorw;
        int nbrw = *Nbrw;
        int xsh = *Xsh;
        int ysh = *Ysh;
        int zsh = *Zsh;
        int segment = *Segment;

        // get_global_id(0)         // blockIdx.z * blockDim.z + threadIdx.z
        // get_local_id(0)          // threadIdx.z
        // get_global_size(0)       // gridDim.z * blockDim.z
        // get_local_size(0)        // blockDim.z

        int flat   = xsh * ysh;
        int column = get_global_id(2);
        int row    = get_global_id(1);
        int slice  = get_global_id(0);
        int plane  = indices[slice];
        int index  = slice * flat + row * xsh + column;
        int position = plane*flat + row*xsh + column;

        if (index < get_global_size(0)*flat && plane>0 && row>0 && column>0 && plane<zsh-1 && row<ysh-1 && column<xsh-1) {

            if (slices[index]==segment) {

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
                int B = raw[position];
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
                        atomic_add(&a[position], 1);
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
    '''
    return src

def _build_kernel_float32():
    src = '''

    float _calc_var(int position, int index, float B, __global float *raw, int segment, __global int *labels, int xsh) {
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

    float weight(float B, float A, float div1) {
        float tmp = B - A;
        return exp( - tmp * tmp * div1 );
        }

    __kernel void randomWalk(__global int *Segment, __global float *raw, __global int *slices, __global int *a, __global int *Xsh, __global int *Ysh, __global int *Zsh, __global int *indices, __global int *Sorw, __global int *Nbrw)
    {
        int sorw = *Sorw;
        int nbrw = *Nbrw;
        int xsh = *Xsh;
        int ysh = *Ysh;
        int zsh = *Zsh;
        int segment = *Segment;

        // get_global_id(0)         // blockIdx.z * blockDim.z + threadIdx.z
        // get_local_id(0)          // threadIdx.z
        // get_global_size(0)       // gridDim.z * blockDim.z
        // get_local_size(0)        // blockDim.z

        int flat   = xsh * ysh;
        int column = get_global_id(2);
        int row    = get_global_id(1);
        int slice  = get_global_id(0);
        int plane  = indices[slice];
        int index  = slice * flat + row * xsh + column;
        int position = plane*flat + row*xsh + column;

        if (index < get_global_size(0)*flat && plane>0 && row>0 && column>0 && plane<zsh-1 && row<ysh-1 && column<xsh-1) {

            if (slices[index]==segment) {

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
                        atomic_add(&a[position], 1);
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
    '''
    return src

