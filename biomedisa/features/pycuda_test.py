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
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

if __name__ == "__main__":

    cuda.init()
    dev = cuda.Device(0)
    ctx = dev.make_context()

    code = """
    __global__ void Funktion(int *a) {

            int xsh = gridDim.x * 10;
            int ysh = gridDim.y * 10;
            int zsh = gridDim.z;

            int column = blockIdx.x * 10 + threadIdx.x;
            int row    = blockIdx.y * 10 + threadIdx.y;
            int plane  = blockIdx.z;

            int index = plane * ysh * xsh + row * xsh + column;

            if ( index < xsh*ysh*zsh ) {
                a[index] = index;
                }

        }
    """
    mod = SourceModule(code)
    kernel = mod.get_function("Funktion")

    xsh = 100
    ysh = 100
    zsh = 100

    a = np.arange(xsh*ysh*zsh, dtype=np.int32)
    a = a.reshape(zsh, ysh, xsh)

    a_gpu = gpuarray.zeros((zsh, ysh, xsh), np.int32)

    block = (10, 10, 1)
    grid = (xsh//10, ysh//10, zsh)

    kernel(a_gpu, block = block, grid = grid)

    test = np.abs(a_gpu.get() - a)

    if np.sum(test) == 0:
        print("PyCUDA test okay!")
    else:
        print("Something went wrong!")

    ctx.pop()
    del ctx

