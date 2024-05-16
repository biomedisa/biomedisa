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
import numba
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

def min_distances(a, b, xsh, ysh):

    code = """
    __global__ void Funktion(int *a, int *b, int *distance, int a_shape, int b_shape, int xsh, int ysh) {

        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index < a_shape) {

            int i = a[index];
            int z_a = i / (xsh * ysh);
            int y_a = (i % (xsh * ysh)) / xsh;
            int x_a = (i % (xsh * ysh)) % xsh;

            i = b[0];
            int z_b = i / (xsh * ysh);
            int y_b = (i % (xsh * ysh)) / xsh;
            int x_b = (i % (xsh * ysh)) % xsh;

            int min_dist = (z_b - z_a)*(z_b - z_a) + (y_b - y_a)*(y_b - y_a) + (x_b - x_a)*(x_b - x_a);

            for (int k = 1; k < b_shape; k++) {

                i = b[k];
                z_b = i / (xsh * ysh);
                y_b = (i % (xsh * ysh)) / xsh;
                x_b = (i % (xsh * ysh)) % xsh;

                int tmp = (z_b - z_a)*(z_b - z_a) + (y_b - y_a)*(y_b - y_a) + (x_b - x_a)*(x_b - x_a);
                if (tmp < min_dist) {
                    min_dist = tmp;
                    }
                }

            distance[index] = min_dist;

            }
        }
    """
    mod = SourceModule(code)
    func = mod.get_function("Funktion")

    x_grid = (a.size // 256) + 1

    a_gpu = gpuarray.to_gpu(a.astype(np.int32))
    b_gpu = gpuarray.to_gpu(b.astype(np.int32))

    distance_gpu = gpuarray.zeros(a.size, dtype=np.int32)

    a_shape_gpu = np.int32(a.size)
    b_shape_gpu = np.int32(b.size)

    xsh_gpu = np.int32(xsh)
    ysh_gpu = np.int32(ysh)

    func(a_gpu, b_gpu, distance_gpu, a_shape_gpu, b_shape_gpu, xsh_gpu, ysh_gpu, block = (256,1,1), grid = (x_grid,1,1))

    return distance_gpu.get()

@numba.jit(nopython=True)
def nonzero(a, indices, zsh, ysh, xsh):
    s = 0
    for k in range(zsh):
        for l in range(ysh):
            for m in range(xsh):
                if a[k,l,m] == 1:
                    indices[s] = k*ysh*xsh + l*xsh + m
                    s += 1
    return indices

def ASSD_one_label(a, b, label):

    # consider label of interest only
    tmp = np.zeros_like(a)
    tmp[a==label] = 1
    a = np.copy(tmp, order='C')
    tmp = np.zeros_like(b)
    tmp[b==label] = 1
    b = np.copy(tmp, order='C')

    # get gradients
    zsh, ysh, xsh = a.shape
    a_gradient = np.sum(np.abs(np.gradient(a)), axis=0)
    b_gradient = np.sum(np.abs(np.gradient(b)), axis=0)
    a_gradient = a_gradient.astype(np.float32)
    b_gradient = b_gradient.astype(np.float32)

    # get surfaces
    a_surface = np.zeros_like(a)
    b_surface = np.zeros_like(b)
    a_surface[np.logical_and(a_gradient>0, a>0)] = 1
    b_surface[np.logical_and(b_gradient>0, b>0)] = 1

    # size of surfaces
    a_size = np.sum(a_surface)
    b_size = np.sum(b_surface)

    # min distances from a_to_b
    a_save = np.copy(a_surface, order='C')
    a_surface[b_surface==1] = 0
    a_surface = np.copy(a_surface, order='C')
    b_surface = np.copy(b_surface, order='C')
    if np.sum(a_surface) == 0:
        distances_a_to_b = 0
    else:
        #a_indices = np.nonzero(a_surface.flatten())[0]
        #b_indices = np.nonzero(b_surface.flatten())[0]
        a_indices = nonzero(a_surface, np.zeros(np.sum(a_surface), dtype=np.int32), zsh, ysh, xsh)
        b_indices = nonzero(b_surface, np.zeros(np.sum(b_surface), dtype=np.int32), zsh, ysh, xsh)
        distances_a_to_b = min_distances(a_indices, b_indices, xsh, ysh)
        distances_a_to_b = np.sqrt(distances_a_to_b)

    # min distances from b_to_a
    b_surface[a_save==1] = 0
    a_surface = np.copy(a_save, order='C')
    b_surface = np.copy(b_surface, order='C')
    if np.sum(b_surface) == 0:
        distances_b_to_a = 0
    else:
        #a_indices = np.nonzero(a_save.flatten())[0]
        #b_indices = np.nonzero(b_surface.flatten())[0]
        a_indices = nonzero(a_surface, np.zeros(np.sum(a_surface), dtype=np.int32), zsh, ysh, xsh)
        b_indices = nonzero(b_surface, np.zeros(np.sum(b_surface), dtype=np.int32), zsh, ysh, xsh)
        distances_b_to_a = min_distances(b_indices, a_indices, xsh, ysh)
        distances_b_to_a = np.sqrt(distances_b_to_a)

    # hausdorff
    hausdorff = max(np.amax(distances_a_to_b), np.amax(distances_b_to_a))

    # return distances
    return np.sum(distances_a_to_b) + np.sum(distances_b_to_a), a_size + b_size, hausdorff

