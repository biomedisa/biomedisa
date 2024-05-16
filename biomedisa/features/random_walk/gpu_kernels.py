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

from pycuda.compiler import SourceModule

def _build_kernel_fill():
    code = """
    __global__ void Funktion(float *a, int xsh, int ysh) {

        int zsh    = gridDim.z;
        int flat   = xsh * ysh;

        int column = blockIdx.x * blockDim.x + threadIdx.x;
        int row    = blockIdx.y * blockDim.y + threadIdx.y;
        int plane  = blockIdx.z;

        unsigned int vol = zsh * flat;
        unsigned int index = plane * flat + row * xsh + column;

        if (index < vol && column < xsh && row < ysh) {
            a[index] = 0;
            }
        }
    """
    mod = SourceModule(code)
    kernel = mod.get_function("Funktion")
    return kernel

def _build_kernel_uncertainty():
    code = """
    __global__ void Funktion(float *max, float *a, int xsh, int ysh) {

        int zsh    = gridDim.z;
        int flat   = xsh * ysh;

        int column = blockIdx.x * blockDim.x + threadIdx.x;
        int row    = blockIdx.y * blockDim.y + threadIdx.y;
        int plane  = blockIdx.z;

        unsigned int vol = zsh * flat;
        unsigned int index = plane * flat + row * xsh + column;

        if (index < vol && plane>0 && row>0 && column>0 && plane<zsh-1 && row<ysh-1 && column<xsh-1) {
            float max2 = max[2 * vol + index];
            if (max2 == 0) {max2 = 1;}
            float tmp  = max[index] / max2;
            float tmp1 = max[vol + index] / max2;
            a[index]   = 1 - (1 - tmp) * (1 - tmp1);
            }
        }
    """
    mod = SourceModule(code)
    kernel = mod.get_function("Funktion")
    return kernel

def _build_kernel_max():
    code = """
    __global__ void Funktion(float *max, float *a, int xsh, int ysh) {

        int zsh    = gridDim.z;
        int flat   = xsh * ysh;

        int column = blockIdx.x * blockDim.x + threadIdx.x;
        int row    = blockIdx.y * blockDim.y + threadIdx.y;
        int plane  = blockIdx.z;

        unsigned int vol = zsh * flat;
        unsigned int index = plane * flat + row * xsh + column;

        if (index < vol && plane>0 && row>0 && column>0 && plane<zsh-1 && row<ysh-1 && column<xsh-1) {
            float tmp  = a[index];
            float max2 = max[2 * vol + index];
            float max1 = max[vol + index];
            if (tmp > max2) {
                max[index] = max1;
                max[vol + index] = max2;
                max[2 * vol + index] = tmp;
                }
            else if (tmp > max1) {
                max[index] = max1;
                max[vol + index] = tmp;
                }
            else if (tmp > max[index]) {
                max[index] = tmp;
                }
            }
        }
    """
    mod = SourceModule(code)
    kernel = mod.get_function("Funktion")
    return kernel

def _build_update_gpu():
    code = """
    __global__ void Funktion(float *phi, float *curvature, int xsh, int ysh) {

        int zsh    = gridDim.z;
        int flat   = xsh * ysh;

        int column = blockIdx.x * blockDim.x + threadIdx.x;
        int row    = blockIdx.y * blockDim.y + threadIdx.y;
        int plane  = blockIdx.z;

        unsigned int vol = zsh * flat;
        unsigned int index = plane * flat + row * xsh + column;

        if (index < vol && plane>0 && row>0 && column>0 && plane<zsh-1 && row<ysh-1 && column<xsh-1) {
            phi[index] += 0.01 * curvature[index];
            }
        }
    """
    mod = SourceModule(code)
    kernel = mod.get_function("Funktion")
    return kernel

def _build_curvature_gpu():
    code = """
    __global__ void Funktion(float *phi, float *curvature, int xsh, int ysh) {

        int zsh    = gridDim.z;
        int flat   = xsh * ysh;

        int column = blockIdx.x * blockDim.x + threadIdx.x;
        int row    = blockIdx.y * blockDim.y + threadIdx.y;
        int plane  = blockIdx.z;

        unsigned int vol = zsh * flat;
        unsigned int index = plane * flat + row * xsh + column;

        if (index < vol && plane>0 && row>0 && column>0 && plane<zsh-1 && row<ysh-1 && column<xsh-1) {

            float eps = 0.0000000001;
            float dx, dxx, dx2, dy, dyy, dy2, dz, dzz, dz2, dxy, dxz, dyz;

            dx  = (phi[index-1] - phi[index+1]) / 2.0;
            dxx = (phi[index-1] - 2*phi[index] + phi[index+1]);
            dx2 = dx * dx;

            dy  = (phi[index-xsh] - phi[index+xsh]) / 2.0;
            dyy = (phi[index-xsh] - 2*phi[index] + phi[index+xsh]);
            dy2 = dy * dy;

            dz  = (phi[index-flat] - phi[index+flat]) / 2.0;
            dzz = (phi[index-flat] - 2*phi[index] + phi[index+flat]);
            dz2 = dz * dz;

            dxy = (phi[index-xsh-1] + phi[index+xsh+1] - phi[index-xsh+1] - phi[index+xsh-1]) / 4.0;
            dxz = (phi[index+flat-1] + phi[index-flat+1] - phi[index+flat+1] - phi[index-flat-1]) / 4.0;
            dyz = (phi[index+flat-xsh] + phi[index-flat+xsh] - phi[index+flat+xsh] - phi[index-flat-xsh]) / 4.0;

            curvature[index] = (dxx*(dy2+dz2)+dyy*(dx2+dz2)+dzz*(dx2+dy2)-2*dx*dy*dxy-2*dx*dz*dxz-2*dy*dz*dyz) / (dx2+dy2+dz2+eps);

            }
        }
    """
    mod = SourceModule(code)
    kernel = mod.get_function("Funktion")
    return kernel

