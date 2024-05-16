##########################################################################
##                                                                      ##
##  Copyright (c) 20019-2024 Philipp Lösel. All rights reserved.        ##
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
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from biomedisa.features.random_walk.gpu_kernels import _build_kernel_fill

def walk(data, slices, indices, indices_child, nbrw, sorw, name, ctx, queue):

    labels = np.unique(slices)
    slicesChunk = _extract_slices(slices, indices, indices_child)
    labelsChunk = np.unique(slicesChunk)

    # remove negative labels from list
    index = np.argwhere(labels<0)
    labels = np.delete(labels, index)
    index = np.argwhere(labelsChunk<0)
    labelsChunk = np.delete(labelsChunk, index)

    walkmapChunk = _walk_on_current_gpu(data, slicesChunk, labelsChunk, indices_child, nbrw, sorw, name)

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

def _walk_on_current_gpu(raw, slices, allLabels, indices, nbrw, sorw, name):

    if raw.dtype == 'uint8':
        kernel = _build_kernel_int8()
        raw = (raw-128).astype('int8')
    else:
        kernel = _build_kernel_float32()
        raw = raw.astype(np.float32)

    fill_gpu = _build_kernel_fill()

    walkmap = np.zeros((len(allLabels),)+raw.shape, dtype=np.float32)
    zsh, ysh, xsh = raw.shape
    slshape = slices.shape[0]

    xsh_gpu = np.int32(xsh)
    ysh_gpu = np.int32(ysh)
    zsh_gpu = np.int32(zsh)
    sorw = np.int32(sorw)
    nbrw = np.int32(nbrw)
    indices = np.array(indices, dtype=np.int32)
    slices = np.array(slices, dtype=np.int32)

    raw_gpu = gpuarray.to_gpu(raw)
    indices_gpu = gpuarray.to_gpu(indices)
    slices_gpu = gpuarray.to_gpu(slices)

    a = np.empty(raw.shape, dtype=np.float32)
    a_gpu = cuda.mem_alloc(a.nbytes)

    block = (32, 32, 1)
    x_grid = (xsh // 32) + 1
    y_grid = (ysh // 32) + 1
    grid = (int(x_grid), int(y_grid), int(slshape))
    grid2 = (int(x_grid), int(y_grid), int(zsh))

    for label_counter, segment in enumerate(allLabels):
        print('%s:' %(name) + ' ' + str(label_counter+1) + '/' + str(len(allLabels)))
        segment_gpu = np.int32(segment)
        fill_gpu(a_gpu, xsh_gpu, ysh_gpu, block=block, grid=grid2)
        kernel(segment_gpu, raw_gpu, slices_gpu, a_gpu, xsh_gpu, ysh_gpu, zsh_gpu, indices_gpu, sorw, nbrw, block=block, grid=grid)
        cuda.memcpy_dtoh(a, a_gpu)
        walkmap[label_counter] += a
    return walkmap

def _build_kernel_int8():
    code = """

    __device__ float _calc_var(int position, int index, float B, float *raw, int segment, int *labels, int xsh) {
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

    __device__ float weight(float B, float *raw, float div1, int position) {
        float tmp = B - (float)(*((char*)(raw) + position));
        return exp( - tmp * tmp * div1 );
        }

    __global__ void Funktion(int segment, float *raw, int *slices, float *a, int xsh, int ysh, int zsh, int *indices, int sorw, int nbrw) {

        int flat   = xsh * ysh;
        int column = blockIdx.x * blockDim.x + threadIdx.x;
        int row    = blockIdx.y * blockDim.y + threadIdx.y;
        int slice  = blockIdx.z;
        int plane  = indices[slice];
        int index  = slice * flat + row * xsh + column;

        if (index < gridDim.z*flat && plane>0 && row>0 && column>0 && plane<zsh-1 && row<ysh-1 && column<xsh-1) {

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
                int position = plane*flat + row*xsh + column;
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
    """
    mod = SourceModule(code)
    kernel = mod.get_function("Funktion")
    return kernel

def _build_kernel_float32():
    code = """

    __device__ float _calc_var(int position, int index, float B, float *raw, int segment, int *labels, int xsh) {
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

    __global__ void Funktion(int segment, float *raw, int *slices, float *a, int xsh, int ysh, int zsh, int *indices, int sorw, int nbrw) {

        int flat   = xsh * ysh;
        int column = blockIdx.x * blockDim.x + threadIdx.x;
        int row    = blockIdx.y * blockDim.y + threadIdx.y;
        int slice  = blockIdx.z;
        int plane  = indices[slice];
        int index  = slice * flat + row * xsh + column;

        if (index < gridDim.z*flat && plane>0 && row>0 && column>0 && plane<zsh-1 && row<ysh-1 && column<xsh-1) {

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
                int position = plane*flat + row*xsh + column;
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
    """
    mod = SourceModule(code)
    kernel = mod.get_function("Funktion")
    return kernel

