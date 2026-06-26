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

import numpy as np
import pyopencl as cl
import os

def walk(data, slices, indices, indices_child, nbrw, sorw, name, ctx, queue, allx):
    if allx:
        labels = np.zeros(0)
        for k in range(3):
            labels = np.append(labels, np.unique(slices[k]))
        labels = np.unique(labels)
        slicesChunk, indicesChunk = [], []
        labelsChunk = np.zeros(0)
        for k in range(3):
            slices_tmp, indices_tmp = _extract_slices(slices[k], indices, indices_child, allx, k)
            slicesChunk.append(slices_tmp)
            indicesChunk.append(indices_tmp)
            labelsChunk = np.append(labelsChunk, np.unique(slices_tmp))
        labelsChunk = np.unique(labelsChunk)
    else:
        labels = np.unique(slices)
        slicesChunk, indicesChunk = _extract_slices(slices, indices, indices_child, allx, 0)
        labelsChunk = np.unique(slicesChunk)

    # remove negative labels from list
    index = np.argwhere(labels<0)
    labels = np.delete(labels, index)
    index = np.argwhere(labelsChunk<0)
    labelsChunk = np.delete(labelsChunk, index)

    walkmapChunk = _walk_on_current_gpu(data, slicesChunk, labelsChunk, indicesChunk, nbrw, sorw, name, ctx, queue, allx)

    if walkmapChunk.shape[0] != len(labels):
        walkmap = np.zeros((len(labels),)+data.shape, dtype=np.float32)
        chunk2Walkmap = np.nonzero(np.isin(labels, labelsChunk))[0]
        for chunkIndex, walkmapIndex in enumerate(chunk2Walkmap):
            walkmap[walkmapIndex] += walkmapChunk[chunkIndex]
    else:
        walkmap = walkmapChunk

    return walkmap

def _extract_slices(slices, indices, indicesChunk, allx, k):
    if allx:
        indices = [x for (x, y) in indices if y == k]
        indicesChunk = [x for (x, y) in indicesChunk if y == k]
    extracted = np.zeros((0, slices.shape[1], slices.shape[2]), dtype=np.int32)
    slicesIndicesToExtract = np.nonzero(np.isin(indices, indicesChunk))[0]
    for arraySliceIndex in slicesIndicesToExtract:
        extracted = np.append(extracted, [slices[arraySliceIndex]], axis=0)
    return extracted, indicesChunk

def _walk_on_current_gpu(raw, extracted_slices, allLabels, indices, nbrw, sorw, name, ctx, queue, allx):

    # allocate host memory
    walkmap = np.zeros((len(allLabels),)+raw.shape, dtype=np.float32)
    a = np.empty(raw.shape, dtype=np.int32)

    # image size
    zsh, ysh, xsh = raw.shape

    # determine label dtype
    if len(allLabels) < 256:
        labels_dtype = [np.uint8, 'uchar']
        ignore_value = next((i for i in range(256) if i not in set(allLabels)), None)
    else:
        labels_dtype = [np.int16, 'short']
        ignore_value = -1

    # rebuild slices
    slices = np.zeros(raw.shape, dtype=labels_dtype[0]) + ignore_value
    if allx:
        for x in range(3):
            extracted_slices[x][extracted_slices[x]<0] = ignore_value
        slices[indices[0]] = extracted_slices[0]
        slices[:,indices[1]] = extracted_slices[1].transpose(1, 0, 2)
        slices[:,:,indices[2]] = extracted_slices[2].transpose(1, 2, 0)
    else:
        extracted_slices[extracted_slices<0] = ignore_value
        slices[indices] = extracted_slices

    # build kernels
    if raw.dtype == 'uint8':
        kernel = cl.Program(ctx, _build_kernel('char', labels_dtype[1])).build().randomWalk
        raw = (raw-128).astype('int8')
    else:
        kernel = cl.Program(ctx, _build_kernel('float', labels_dtype[1])).build().randomWalk
        raw = raw.astype(np.float32)

    # allocate memory for variables on the device
    mf = cl.mem_flags
    slices_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=slices)
    raw_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=raw)
    a_cl = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=a)

    # block and grid size
    block = None
    grid = (zsh, ysh, xsh)

    # call Kernel
    for label_counter, segment in enumerate(allLabels):
        print('%s:' %(name) + ' ' + str(label_counter+1) + '/' + str(len(allLabels)))
        cl.enqueue_fill_buffer(queue, a_cl, np.int32(0), offset=0, size=a.nbytes)
        kernel(queue, grid, block, np.int32(segment), raw_cl, slices_cl, a_cl,
            np.int32(xsh), np.int32(ysh), np.int32(zsh), np.int32(sorw), np.int32(nbrw))
        cl.enqueue_copy(queue, a, a_cl)
        walkmap[label_counter] += a
    return walkmap

def _build_kernel(data_dtype, labels_dtype):
    src = '''

    float _calc_var(int index, float B, __global DATA_DTYPE *raw, const int segment, __global LABELS_DTYPE *labels, const int xsh, const int ysh) {
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

    float weight(float B, float A, float div1) {
        float tmp = B - A;
        return exp( - tmp * tmp * div1 );
        }

    __kernel void randomWalk(const int segment, __global DATA_DTYPE *raw, __global LABELS_DTYPE *slices, __global int *a, const int xsh, const int ysh, const int zsh, const int sorw, const int nbrw) {

        // get_global_id(0)         // blockIdx.z * blockDim.z + threadIdx.z
        // get_local_id(0)          // threadIdx.z
        // get_global_size(0)       // gridDim.z * blockDim.z
        // get_local_size(0)        // blockDim.z

        int flat   = xsh * ysh;
        int column = get_global_id(2);
        int row    = get_global_id(1);
        int plane  = get_global_id(0);
        int index  = plane * flat + row * xsh + column;

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
                        atomic_add(&a[index], 1);
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
    '''
    return src.replace("DATA_DTYPE", data_dtype).replace("LABELS_DTYPE", labels_dtype)

