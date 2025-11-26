#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

def _build_kernel():
    code = """
    __global__ void Funktion(short *verts1, char *arr1, char *arr2, int xsh1, int ysh1, int zsh1, int xsh2, int ysh2, int zsh2, float *centroid1, float *centroid2, int n_verts1, float alpha, float beta, float gamma) {

        uint index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index < n_verts1) {

            /* convert to bogenmass */
            gamma = gamma / 180.0 * 3.14159;
            beta = beta / 180.0 * 3.14159;
            alpha = alpha / 180.0 * 3.14159;

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

            /* get vertice */
            float z = (float)(verts1[index*3+0]) - centroid1[0];
            float y = (float)(verts1[index*3+1]) - centroid1[1];
            float x = (float)(verts1[index*3+2]) - centroid1[2];

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

            int Z = (int)(verts1[index*3+0]);
            int Y = (int)(verts1[index*3+1]);
            int X = (int)(verts1[index*3+2]);

            int position1 = Z*xsh1*ysh1 + X*xsh1 + Y;
            int position2 = Nz*xsh2*ysh2 + Nx*xsh2 + Ny;

            /* correct particle */
            if (arr2[position2]>0) {
                arr1[position1] = 1;}

            }
        }
    """
    mod = SourceModule(code)
    kernel = mod.get_function("Funktion")
    return kernel

def _build_inverse_kernel():
    code = """
    __global__ void Funktion(short *verts1, char *arr1, char *arr2, int xsh1, int ysh1, int zsh1, int xsh2, int ysh2, int zsh2, float *centroid1, float *centroid2, int n_verts1, float alpha, float beta, float gamma) {

        uint index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index < n_verts1) {

            /* convert to bogenmass */
            gamma = gamma / 180.0 * 3.14159;
            beta = beta / 180.0 * 3.14159;
            alpha = alpha / 180.0 * 3.14159;

            /* inverse rotation */
            alpha *= -1;
            beta *= -1;
            gamma *= -1;

            float r11 = cosf(alpha)*cosf(beta);
            float r12 = -sinf(alpha)*cosf(beta);
            float r13 = sinf(beta);
            float r21 = sinf(alpha)*cosf(gamma);
            float r22 = cosf(alpha)*sinf(beta)*sinf(gamma);
            float r23 = cosf(alpha)*cosf(gamma);
            float r24 = -sinf(alpha)*sinf(beta)*sinf(gamma);
            float r25 = -cosf(beta)*sinf(gamma);
            float r31 = sinf(alpha)*sinf(gamma);
            float r32 = -cosf(alpha)*sinf(beta)*cosf(gamma);
            float r33 = cosf(alpha)*sinf(gamma);
            float r34 = sinf(alpha)*sinf(beta)*cosf(gamma);
            float r35 = cosf(beta)*cosf(gamma);

            /* get vertice */
            float z = (float)(verts1[index*3+0]) - centroid1[0];
            float y = (float)(verts1[index*3+1]) - centroid1[1];
            float x = (float)(verts1[index*3+2]) - centroid1[2];

            /* rotate vertice */
            float nx = r11*x + r12*y + r13*z;
            float ny = (r21 + r22)*x + (r23 + r24)*y + r25*z;
            float nz = (r31 + r32)*x + (r33 + r34)*y + r35*z;

            /* vertice to voxel data */
            int Nx = (int)(roundf(nx + centroid2[2]));
            int Ny = (int)(roundf(ny + centroid2[1]));
            int Nz = (int)(roundf(nz + centroid2[0]));
            Ny = min(max(0,Ny),xsh2-1);
            Nx = min(max(0,Nx),ysh2-1);
            Nz = min(max(0,Nz),zsh2-1);

            int Z = (int)(verts1[index*3+0]);
            int Y = (int)(verts1[index*3+1]);
            int X = (int)(verts1[index*3+2]);

            int position1 = Z*xsh1*ysh1 + X*xsh1 + Y;
            int position2 = Nz*xsh2*ysh2 + Nx*xsh2 + Ny;

            /* correct particle */
            if (arr2[position2]>0) {
                arr1[position1] = 1;}

            }
        }
    """
    mod = SourceModule(code)
    kernel = mod.get_function("Funktion")
    return kernel

def correct_particle(verts1, centroid1, centroid2, alpha, beta, gamma, arr1, arr2, inverse):

    zsh1, ysh1, xsh1 = arr1.shape
    zsh2, ysh2, xsh2 = arr2.shape

    xsh1 = np.int32(xsh1)
    ysh1 = np.int32(ysh1)
    zsh1 = np.int32(zsh1)
    xsh2 = np.int32(xsh2)
    ysh2 = np.int32(ysh2)
    zsh2 = np.int32(zsh2)
    alpha = np.float32(alpha)
    beta = np.float32(beta)
    gamma = np.float32(gamma)
    n_verts1 = np.int32(verts1.shape[0])

    centroid1_gpu = gpuarray.to_gpu(np.array(centroid1, dtype=np.float32))
    centroid2_gpu = gpuarray.to_gpu(np.array(centroid2, dtype=np.float32))
    arr1_gpu = gpuarray.zeros(arr1.shape, np.int8)
    arr2_gpu = gpuarray.to_gpu(arr2.astype(np.int8))
    verts1_gpu = gpuarray.to_gpu(np.array(verts1, dtype=np.int16))

    block = (128, 1, 1)
    grid = ((verts1.shape[0]//block[0])+1, 1, 1)

    if inverse:
        kernel = _build_inverse_kernel()
    else:
        kernel = _build_kernel()

    kernel(verts1_gpu, arr1_gpu, arr2_gpu,
        xsh1, ysh1, zsh1, xsh2, ysh2, zsh2, centroid1_gpu, centroid2_gpu,
        n_verts1, alpha, beta, gamma, block=block, grid=grid)

    return arr1_gpu.get()

