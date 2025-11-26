#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

def _build_kernel():
    code = """
    __global__ void Funktion(int *match, float *verts1, float *verts2, int *arr1, int *arr2, int xsh1, int ysh1, int zsh1, int xsh2, int ysh2, int zsh2, float *centroid1, float *centroid2, int n_verts1, int n_verts2, float alpha_s, float beta_s, float gamma_s) {

        uint index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index < n_verts1) {

            for (int gamma_i=0; gamma_i<10; gamma_i++) {
            for (int beta_i=0; beta_i<10; beta_i++) {
            for (int alpha_i=0; alpha_i<10; alpha_i++) {

                float gamma = gamma_s - 1 + 2*(float)(gamma_i)/9.0;
                float beta = beta_s - 1 + 2*(float)(beta_i)/9.0;
                float alpha = alpha_s - 1 + 2*(float)(alpha_i)/9.0;

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

                /* rotate vertice1 */
                float z = verts1[index*3+0];
                float y = verts1[index*3+1];
                float x = verts1[index*3+2];

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
                atomicAdd(&match[alpha_i*100+beta_i*10+gamma_i], arr2[position]);
                }
                }
                }

            }
        }
    """
    mod = SourceModule(code)
    kernel = mod.get_function("Funktion")
    return kernel

def _build_inverse_kernel():
    code = """
    __global__ void Funktion(int *match, float *verts1, float *verts2, int *arr1, int *arr2, int xsh1, int ysh1, int zsh1, int xsh2, int ysh2, int zsh2, float *centroid1, float *centroid2, int n_verts1, int n_verts2, float alpha_s, float beta_s, float gamma_s) {

        uint index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index < n_verts2) {

            for (int gamma_i=0; gamma_i<10; gamma_i++) {
            for (int beta_i=0; beta_i<10; beta_i++) {
            for (int alpha_i=0; alpha_i<10; alpha_i++) {

                float gamma = gamma_s - 1 + 2*(float)(gamma_i)/9.0;
                float beta = beta_s - 1 + 2*(float)(beta_i)/9.0;
                float alpha = alpha_s - 1 + 2*(float)(alpha_i)/9.0;

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

                /* rotate vertice2 */
                float z = verts2[index*3+0];
                float y = verts2[index*3+1];
                float x = verts2[index*3+2];

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
                atomicAdd(&match[alpha_i*100+beta_i*10+gamma_i], arr1[position]);
                }
                }
                }

            }
        }
    """
    mod = SourceModule(code)
    kernel = mod.get_function("Funktion")
    return kernel

def best_rotation_gpu(verts1, verts2, arr1, arr2, centroid1, centroid2, alpha, beta, gamma):

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
    n_verts2 = np.int32(verts2.shape[0])

    centroid1_gpu = gpuarray.to_gpu(np.array(centroid1, dtype=np.float32))
    centroid2_gpu = gpuarray.to_gpu(np.array(centroid2, dtype=np.float32))
    arr1_gpu = gpuarray.to_gpu(arr1.astype(np.int32))
    arr2_gpu = gpuarray.to_gpu(arr2.astype(np.int32))
    verts1_gpu = gpuarray.to_gpu(np.array(verts1, dtype=np.float32))
    verts2_gpu = gpuarray.to_gpu(np.array(verts2, dtype=np.float32))
    match_gpu = gpuarray.zeros((10, 10, 10), np.int32)

    block = (128, 1, 1)
    grid = ((verts1.shape[0]//block[0])+1, 1, 1)
    kernel = _build_kernel()
    kernel(match_gpu, verts1_gpu, verts2_gpu, arr1_gpu, arr2_gpu,
        xsh1, ysh1, zsh1, xsh2, ysh2, zsh2, centroid1_gpu, centroid2_gpu,
        n_verts1, n_verts2, alpha, beta, gamma, block=block, grid=grid)

    grid = ((verts2.shape[0]//block[0])+1, 1, 1)
    kernel = _build_inverse_kernel()
    kernel(match_gpu, verts1_gpu, verts2_gpu, arr1_gpu, arr2_gpu,
        xsh1, ysh1, zsh1, xsh2, ysh2, zsh2, centroid1_gpu, centroid2_gpu,
        n_verts1, n_verts2, alpha, beta, gamma, block=block, grid=grid)

    match_cpu = match_gpu.get()
    max_coords = np.unravel_index(np.argmax(match_cpu), match_cpu.shape)
    best_alpha = alpha - 1 + 2*max_coords[0]/9.0
    best_beta = beta - 1 + 2*max_coords[1]/9.0
    best_gamma = gamma - 1 + 2*max_coords[2]/9.0
    return np.amax(match_cpu), best_alpha, best_beta, best_gamma

