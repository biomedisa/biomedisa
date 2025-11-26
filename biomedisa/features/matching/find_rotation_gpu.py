#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

def best_rotation_gpu(verts1, verts2, arr1, arr2, centroid1, centroid2):

    code = """
    __global__ void Funktion(int *match, float *verts1, float *verts2, int *arr1, int *arr2, int xsh1, int ysh1, int zsh1, int xsh2, int ysh2, int zsh2, float *centroid1, float *centroid2, int n_verts1, int n_verts2) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        int idz = blockIdx.z;

        uint index = idz*360*180 + idy*360 + idx;

        if (index < 360*180*360) {

            /* convert to bogenmass */
            float gamma = (float)(idx) / 180.0 * 3.14159;
            float beta = ((float)(idy)-90.0) / 180.0 * 3.14159;
            float alpha = (float)(idz) / 180.0 * 3.14159;

            float r11 = cosf(alpha)*cosf(beta);
            float r12 = cosf(alpha)*sinf(beta)*sinf(gamma);
            float r13 = -sinf(alpha)*cosf(gamma);
            float r14 = cosf(alpha)*sinf(beta)*cosf(gamma);
            float r15 = sinf(alpha)*sinf(gamma);
            float r21 = sinf(alpha)*cosf(beta);
            float r22 = sinf(alpha)*sinf(beta)*sinf(gamma);
            float r23 = cosf(alpha)*cosf(gamma);
            float r24 = sinf(alpha)*sinf(beta)*cosf(gamma);
            float r25 = -cosf(alpha)*sinf(gamma);
            float r31 = -sinf(beta);
            float r32 = cosf(beta)*sinf(gamma);
            float r33 = cosf(beta)*cosf(gamma);

            /* rotate vertices1 */
            for (int k=0; k<n_verts1; k++) {
                float z = verts1[k*3+0];
                float y = verts1[k*3+1];
                float x = verts1[k*3+2];

                /* rotate vertice */
                float nx = r11*x + (r12 + r13)*y + (r14 + r15)*z;
                float ny = r21*x + (r22 + r23)*y + (r24 + r25)*z;
                float nz = r31*x + r32*y + r33*z;

                /* vertice to voxel data */
                int Nx = (int)(roundf(nx + centroid2[2]));
                int Ny = (int)(roundf(ny + centroid2[1]));
                int Nz = (int)(roundf(nz + centroid2[0]));
                Ny = min(max(0,Ny),xsh2-1);
                Nx = min(max(0,Nx),ysh2-1);
                Nz = min(max(0,Nz),zsh2-1);

                int position = Nz*xsh2*ysh2 + Nx*xsh2 + Ny;
                match[index] += arr2[position];
                }

            /* inverse rotation */
            alpha *= -1;
            beta *= -1;
            gamma *= -1;

            r11 = cosf(alpha)*cosf(beta);
            r12 = -sinf(alpha)*cosf(beta);
            r13 = sinf(beta);
            r21 = sinf(alpha)*cosf(gamma);
            r22 = cosf(alpha)*sinf(beta)*sinf(gamma);
            r23 = cosf(alpha)*cosf(gamma);
            r24 = -sinf(alpha)*sinf(beta)*sinf(gamma);
            r25 = -cosf(beta)*sinf(gamma);
            r31 = sinf(alpha)*sinf(gamma);
            r32 = -cosf(alpha)*sinf(beta)*cosf(gamma);
            r33 = cosf(alpha)*sinf(gamma);
            float r34 = sinf(alpha)*sinf(beta)*cosf(gamma);
            float r35 = cosf(beta)*cosf(gamma);

            /* rotate vertices2 */
            for (int k=0; k<n_verts2; k++) {
                float z = verts2[k*3];
                float y = verts2[k*3 + 1];
                float x = verts2[k*3 + 2];

                /* rotate vertice */
                float nx = r11*x + r12*y + r13*z;
                float ny = (r21 + r22)*x + (r23 + r24)*y + r25*z;
                float nz = (r31 + r32)*x + (r33 + r34)*y + r35*z;

                /* vertice to voxel data */
                int Nx = (int)(roundf(nx + centroid1[2]));
                int Ny = (int)(roundf(ny + centroid1[1]));
                int Nz = (int)(roundf(nz + centroid1[0]));
                Ny = min(max(0,Ny),xsh1-1);
                Nx = min(max(0,Nx),ysh1-1);
                Nz = min(max(0,Nz),zsh1-1);

                int position = Nz*xsh1*ysh1 + Nx*xsh1 + Ny;
                match[index] += arr1[position];
                }

            }
        }
    """
    mod = SourceModule(code)
    kernel = mod.get_function("Funktion")

    zsh1, ysh1, xsh1 = arr1.shape
    zsh2, ysh2, xsh2 = arr2.shape

    centroid1_gpu = gpuarray.to_gpu(np.array(centroid1, dtype=np.float32))
    centroid2_gpu = gpuarray.to_gpu(np.array(centroid2, dtype=np.float32))
    arr1_gpu = gpuarray.to_gpu(arr1.astype(np.int32))
    arr2_gpu = gpuarray.to_gpu(arr2.astype(np.int32))
    verts1_gpu = gpuarray.to_gpu(np.array(verts1, dtype=np.float32))
    verts2_gpu = gpuarray.to_gpu(np.array(verts2, dtype=np.float32))
    match_gpu = gpuarray.zeros((360, 180, 360), np.int32)

    block = (20, 20, 1)
    grid = (360//20, 180//20, 360)

    xsh1 = np.int32(xsh1)
    ysh1 = np.int32(ysh1)
    zsh1 = np.int32(zsh1)
    xsh2 = np.int32(xsh2)
    ysh2 = np.int32(ysh2)
    zsh2 = np.int32(zsh2)
    n_verts1 = np.int32(verts1.shape[0])
    n_verts2 = np.int32(verts2.shape[0])

    kernel(match_gpu, verts1_gpu, verts2_gpu, arr1_gpu, arr2_gpu,
        xsh1, ysh1, zsh1, xsh2, ysh2, zsh2, centroid1_gpu, centroid2_gpu,
        n_verts1, n_verts2, block=block, grid=grid)

    match_cpu = match_gpu.get()
    max_coords = np.unravel_index(np.argmax(match_cpu), match_cpu.shape)
    best_alpha = max_coords[0]
    best_beta = max_coords[1]-90
    best_gamma = max_coords[2]
    return np.amax(match_cpu), best_alpha, best_beta, best_gamma

