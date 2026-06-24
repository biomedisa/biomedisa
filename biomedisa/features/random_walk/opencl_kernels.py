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

import pyopencl as cl

FILL_SRC = r"""
__kernel void Funktion(__global float *a,
                       const int xsh,
                       const int ysh)
{
    int column = get_global_id(0);
    int row    = get_global_id(1);
    int plane  = get_global_id(2);

    int zsh  = get_global_size(2);
    int flat = xsh * ysh;

    uint vol   = zsh * flat;
    uint index = plane * flat + row * xsh + column;

    if (index < vol && column < xsh && row < ysh)
    {
        a[index] = 0.0f;
    }
}
"""


UNCERTAINTY_SRC = r"""
__kernel void Funktion(__global float *maxbuf,
                       __global float *a,
                       const int xsh,
                       const int ysh)
{
    int column = get_global_id(0);
    int row    = get_global_id(1);
    int plane  = get_global_id(2);

    int zsh  = get_global_size(2);
    int flat = xsh * ysh;

    uint vol   = zsh * flat;
    uint index = plane * flat + row * xsh + column;

    if (index < vol &&
        plane > 0 && row > 0 && column > 0 &&
        plane < zsh - 1 &&
        row   < ysh - 1 &&
        column < xsh - 1)
    {
        float max2 = maxbuf[2 * vol + index];

        if (max2 == 0.0f)
            max2 = 1.0f;

        float tmp  = maxbuf[index] / max2;
        float tmp1 = maxbuf[vol + index] / max2;

        a[index] = 1.0f - (1.0f - tmp) * (1.0f - tmp1);
    }
}
"""


MAX_SRC = r"""
__kernel void Funktion(__global float *maxbuf,
                       __global float *a,
                       const int xsh,
                       const int ysh)
{
    int column = get_global_id(0);
    int row    = get_global_id(1);
    int plane  = get_global_id(2);

    int zsh  = get_global_size(2);
    int flat = xsh * ysh;

    uint vol   = zsh * flat;
    uint index = plane * flat + row * xsh + column;

    if (index < vol &&
        plane > 0 && row > 0 && column > 0 &&
        plane < zsh - 1 &&
        row   < ysh - 1 &&
        column < xsh - 1)
    {
        float tmp  = a[index];
        float max2 = maxbuf[2 * vol + index];
        float max1 = maxbuf[vol + index];

        if (tmp > max2)
        {
            maxbuf[index]           = max1;
            maxbuf[vol + index]     = max2;
            maxbuf[2 * vol + index] = tmp;
        }
        else if (tmp > max1)
        {
            maxbuf[index]       = max1;
            maxbuf[vol + index] = tmp;
        }
        else if (tmp > maxbuf[index])
        {
            maxbuf[index] = tmp;
        }
    }
}
"""


UPDATE_SRC = r"""
__kernel void Funktion(__global float *phi,
                       __global float *curvature,
                       const int xsh,
                       const int ysh)
{
    int column = get_global_id(0);
    int row    = get_global_id(1);
    int plane  = get_global_id(2);

    int zsh  = get_global_size(2);
    int flat = xsh * ysh;

    uint vol   = zsh * flat;
    uint index = plane * flat + row * xsh + column;

    if (index < vol &&
        plane > 0 && row > 0 && column > 0 &&
        plane < zsh - 1 &&
        row   < ysh - 1 &&
        column < xsh - 1)
    {
        phi[index] += 0.01f * curvature[index];
    }
}
"""


CURVATURE_SRC = r"""
__kernel void Funktion(__global float *phi,
                       __global float *curvature,
                       const int xsh,
                       const int ysh)
{
    int column = get_global_id(0);
    int row    = get_global_id(1);
    int plane  = get_global_id(2);

    int zsh  = get_global_size(2);
    int flat = xsh * ysh;

    uint vol   = zsh * flat;
    uint index = plane * flat + row * xsh + column;

    if (index < vol &&
        plane > 0 && row > 0 && column > 0 &&
        plane < zsh - 1 &&
        row   < ysh - 1 &&
        column < xsh - 1)
    {
        const float eps = 1e-10f;

        float dx, dxx, dx2;
        float dy, dyy, dy2;
        float dz, dzz, dz2;
        float dxy, dxz, dyz;

        dx  = (phi[index-1] - phi[index+1]) * 0.5f;
        dxx = (phi[index-1] - 2.0f*phi[index] + phi[index+1]);
        dx2 = dx * dx;

        dy  = (phi[index-xsh] - phi[index+xsh]) * 0.5f;
        dyy = (phi[index-xsh] - 2.0f*phi[index] + phi[index+xsh]);
        dy2 = dy * dy;

        dz  = (phi[index-flat] - phi[index+flat]) * 0.5f;
        dzz = (phi[index-flat] - 2.0f*phi[index] + phi[index+flat]);
        dz2 = dz * dz;

        dxy = (phi[index-xsh-1] + phi[index+xsh+1]
             - phi[index-xsh+1] - phi[index+xsh-1]) * 0.25f;

        dxz = (phi[index+flat-1] + phi[index-flat+1]
             - phi[index+flat+1] - phi[index-flat-1]) * 0.25f;

        dyz = (phi[index+flat-xsh] + phi[index-flat+xsh]
             - phi[index+flat+xsh] - phi[index-flat-xsh]) * 0.25f;

        curvature[index] =
            (dxx*(dy2+dz2)
            +dyy*(dx2+dz2)
            +dzz*(dx2+dy2)
            -2.0f*dx*dy*dxy
            -2.0f*dx*dz*dxz
            -2.0f*dy*dz*dyz)
            /(dx2+dy2+dz2+eps);
    }
}
"""

def build_kernel(ctx, source):
    return cl.Program(ctx, source).build().Funktion

def _build_kernel_fill(ctx):
    return build_kernel(ctx, FILL_SRC)

def _build_kernel_uncertainty(ctx):
    return build_kernel(ctx, UNCERTAINTY_SRC)

def _build_kernel_max(ctx):
    return build_kernel(ctx, MAX_SRC)

def _build_update_gpu(ctx):
    return build_kernel(ctx, UPDATE_SRC)

def _build_curvature_gpu(ctx):
    return build_kernel(ctx, CURVATURE_SRC)

