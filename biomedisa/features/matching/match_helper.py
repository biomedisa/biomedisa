import os
import sys
import numpy as np
from tifffile import imread, imwrite
from biomedisa.features.biomedisa_helper import *
from scipy import ndimage
import numba
from numba.typed import List

@numba.jit(nopython=True)
def find_best_rotation(verts1, verts2, arr1, arr2, centroid1, centroid2, total,
    best_match, best_alpha, best_beta, best_gamma, alpha_range, beta_range, gamma_range):
    zsh1, xsh1, ysh1 = arr1.shape
    zsh2, xsh2, ysh2 = arr2.shape
    for alpha_i in alpha_range: # around z
        for beta_i in beta_range: # around y
            for gamma_i in gamma_range: # around x
                match = 0
                # convert to bogenmass
                alpha = alpha_i/180*np.pi
                beta = beta_i/180*np.pi
                gamma = gamma_i/180*np.pi

                # rotate vertices1
                for k in range(verts1.shape[0]):
                    z, y, x = verts1[k]

                    # rotate vertice
                    nx = np.cos(alpha)*np.cos(beta)*x + np.cos(alpha)*np.sin(beta)*np.sin(gamma)*y - np.sin(alpha)*np.cos(gamma)*y + np.cos(alpha)*np.sin(beta)*np.cos(gamma)*z + np.sin(alpha)*np.sin(gamma)*z
                    ny = np.sin(alpha)*np.cos(beta)*x + np.sin(alpha)*np.sin(beta)*np.sin(gamma)*y + np.cos(alpha)*np.cos(gamma)*y + np.sin(alpha)*np.sin(beta)*np.cos(gamma)*z - np.cos(alpha)*np.sin(gamma)*z
                    nz = -np.sin(beta)*x + np.cos(beta)*np.sin(gamma)*y + np.cos(beta)*np.cos(gamma)*z

                    # vertice to voxel data
                    nx = int(round(nx + centroid2[2]))
                    ny = int(round(ny + centroid2[1]))
                    nz = int(round(nz + centroid2[0]))
                    nx = min(max(0,nx),xsh2-1)
                    ny = min(max(0,ny),ysh2-1)
                    nz = min(max(0,nz),zsh2-1)
                    match += arr2[nz, nx, ny]

                # inverse rotation
                alpha *= -1
                beta *= -1
                gamma *= -1

                # rotate vertices2
                for k in range(verts2.shape[0]):
                    z, y, x = verts2[k]

                    # rotate vertice
                    nx = np.cos(alpha)*np.cos(beta)*x - np.sin(alpha)*np.cos(beta)*y + np.sin(beta)*z
                    ny = np.sin(alpha)*np.cos(gamma)*x + np.cos(alpha)*np.sin(beta)*np.sin(gamma)*x + np.cos(alpha)*np.cos(gamma)*y - np.sin(alpha)*np.sin(beta)*np.sin(gamma)*y - np.cos(beta)*np.sin(gamma)*z
                    nz = np.sin(alpha)*np.sin(gamma)*x - np.cos(alpha)*np.sin(beta)*np.cos(gamma)*x + np.cos(alpha)*np.sin(gamma)*y + np.sin(alpha)*np.sin(beta)*np.cos(gamma)*y + np.cos(beta)*np.cos(gamma)*z

                    # vertice to voxel data
                    nx = int(round(nx + centroid1[2]))
                    ny = int(round(ny + centroid1[1]))
                    nz = int(round(nz + centroid1[0]))
                    nx = min(max(0,nx),xsh1-1)
                    ny = min(max(0,ny),ysh1-1)
                    nz = min(max(0,nz),zsh1-1)
                    match += arr1[nz, nx, ny]

                # match result
                if match > best_match:
                    best_match = match
                    best_alpha = alpha_i
                    best_beta = beta_i
                    best_gamma = gamma_i
                    #print(round(best_match/total,4), best_alpha, best_beta, best_gamma)

    return best_match, best_alpha, best_beta, best_gamma


def rotation_dice(a, b, best_alpha, best_beta, best_gamma, rank, gpu=True):
    # total number of points
    total1 = np.sum(a)
    total2 = np.sum(b)
    total = total1 + total2

    # get vertices
    verts1 = where(a)
    verts2 = where(b)
    '''z, x, y = np.where(a)
    verts1 = np.zeros((z.size, 3))
    for k in range(z.size):
        verts1[k] = (z[k],y[k],x[k])
    z, x, y = np.where(b)
    verts2 = np.zeros((z.size, 3))
    for k in range(z.size):
        verts2[k] = (z[k],y[k],x[k])'''

    # get centroids
    centroid1 = (np.mean(verts1[:,0]), np.mean(verts1[:,1]), np.mean(verts1[:,2]))
    centroid2 = (np.mean(verts2[:,0]), np.mean(verts2[:,1]), np.mean(verts2[:,2]))

    # centralize
    verts1 = verts1 - centroid1
    verts2 = verts2 - centroid2

    # find best rotation
    best_match = 0
    if gpu and best_alpha==None:
        import pycuda.driver as cuda
        cuda.init()
        dev = cuda.Device(rank)
        ctx = dev.make_context()
        from biomedisa.features.matching.find_rotation_gpu import best_rotation_gpu
        best_match, best_alpha, best_beta, best_gamma = best_rotation_gpu(
            verts1, verts2, a, b, centroid1, centroid2)
        ctx.pop()
        ctx.detach()
        del ctx
    if gpu:
        import pycuda.driver as cuda
        cuda.init()
        dev = cuda.Device(rank)
        ctx = dev.make_context()
        from biomedisa.features.matching.refine_rotation_gpu import best_rotation_gpu
        best_match, best_alpha, best_beta, best_gamma = best_rotation_gpu(
            verts1, verts2, a, b, centroid1, centroid2, best_alpha, best_beta, best_gamma)
        ctx.pop()
        ctx.detach()
        del ctx
    else:
        start = 1 if best_alpha==None else 3
        for step in range(start,4):
            if step==1:
                alpha_range = range(0,360,5)
                beta_range = range(-90,91,5)
                gamma_range = range(0,360,5)
            elif step==2:
                alpha_range = range(best_alpha-10,best_alpha+11)
                beta_range = range(best_beta-10,best_beta+11)
                gamma_range = range(best_gamma-10,best_gamma+11)
            elif step==3:
                alpha_range = np.linspace(best_alpha-1,best_alpha+1,10)
                beta_range = np.linspace(best_beta-1,best_beta+1,10)
                gamma_range = np.linspace(best_gamma-1,best_gamma+1,10)
            # convert to numba list
            typed_a = List()
            [typed_a.append(x) for x in alpha_range]
            alpha_range = typed_a
            typed_b = List()
            [typed_b.append(x) for x in beta_range]
            beta_range = typed_b
            typed_c = List()
            [typed_c.append(x) for x in gamma_range]
            gamma_range = typed_c
            # iterate over possibilities
            best_match, best_alpha, best_beta, best_gamma = find_best_rotation(
                verts1, verts2, a, b, centroid1, centroid2, total,
                best_match, best_alpha, best_beta, best_gamma, alpha_range, beta_range, gamma_range)

    return best_match/total, best_alpha, best_beta, best_gamma


@numba.jit(nopython=True)
def rotate_image(verts1, centroid1, centroid2, alpha, beta, gamma, arr2):
    zsh, xsh, ysh = arr2.shape
    # convert to bogenmass
    alpha = alpha/180*np.pi
    beta = beta/180*np.pi
    gamma = gamma/180*np.pi
    # rotate vertices
    for k in range(verts1.shape[0]):
        z = verts1[k,0] - centroid1[0]
        y = verts1[k,1] - centroid1[1]
        x = verts1[k,2] - centroid1[2]
        # rotate vertice
        nx = np.cos(alpha)*np.cos(beta)*x + np.cos(alpha)*np.sin(beta)*np.sin(gamma)*y - np.sin(alpha)*np.cos(gamma)*y + np.cos(alpha)*np.sin(beta)*np.cos(gamma)*z + np.sin(alpha)*np.sin(gamma)*z
        ny = np.sin(alpha)*np.cos(beta)*x + np.sin(alpha)*np.sin(beta)*np.sin(gamma)*y + np.cos(alpha)*np.cos(gamma)*y + np.sin(alpha)*np.sin(beta)*np.cos(gamma)*z - np.cos(alpha)*np.sin(gamma)*z
        nz = -np.sin(beta)*x + np.cos(beta)*np.sin(gamma)*y + np.cos(beta)*np.cos(gamma)*z
        # vertice to voxel data
        nx = int(round(nx + centroid2[2]))
        ny = int(round(ny + centroid2[1]))
        nz = int(round(nz + centroid2[0]))
        nx = min(max(0,nx),xsh-1)
        ny = min(max(0,ny),ysh-1)
        nz = min(max(0,nz),zsh-1)
        arr2[nz, nx, ny] = 1
    return arr2


@numba.jit(nopython=True)
def rotate_image_inv(verts2, centroid1, centroid2, alpha, beta, gamma, arr1):
    zsh, xsh, ysh = arr1.shape
    # convert to bogenmass
    alpha = -alpha/180*np.pi
    beta = -beta/180*np.pi
    gamma = -gamma/180*np.pi
    # rotate vertices
    for k in range(verts2.shape[0]):
        z = verts2[k,0] - centroid2[0]
        y = verts2[k,1] - centroid2[1]
        x = verts2[k,2] - centroid2[2]
        # rotate vertice
        nx = np.cos(alpha)*np.cos(beta)*x - np.sin(alpha)*np.cos(beta)*y + np.sin(beta)*z
        ny = np.sin(alpha)*np.cos(gamma)*x + np.cos(alpha)*np.sin(beta)*np.sin(gamma)*x + np.cos(alpha)*np.cos(gamma)*y - np.sin(alpha)*np.sin(beta)*np.sin(gamma)*y - np.cos(beta)*np.sin(gamma)*z
        nz = np.sin(alpha)*np.sin(gamma)*x - np.cos(alpha)*np.sin(beta)*np.cos(gamma)*x + np.cos(alpha)*np.sin(gamma)*y + np.sin(alpha)*np.sin(beta)*np.cos(gamma)*y + np.cos(beta)*np.cos(gamma)*z
        # vertice to voxel data
        nx = int(round(nx + centroid1[2]))
        ny = int(round(ny + centroid1[1]))
        nz = int(round(nz + centroid1[0]))
        nx = min(max(0,nx),xsh-1)
        ny = min(max(0,ny),ysh-1)
        nz = min(max(0,nz),zsh-1)
        arr1[nz, nx, ny] = 1
    return arr1


@numba.jit(nopython=True)
def true_rotate_image(verts1, centroid1, centroid2, alpha, beta, gamma, arr2, arr1):
    zsh, xsh, ysh = arr2.shape
    # convert to bogenmass
    alpha = alpha/180*np.pi
    beta = beta/180*np.pi
    gamma = gamma/180*np.pi
    # rotate vertices
    for k in range(verts1.shape[0]):
        z = verts1[k,0] - centroid1[0]
        y = verts1[k,1] - centroid1[1]
        x = verts1[k,2] - centroid1[2]
        # rotate vertice
        nx = np.cos(alpha)*np.cos(beta)*x + np.cos(alpha)*np.sin(beta)*np.sin(gamma)*y - np.sin(alpha)*np.cos(gamma)*y + np.cos(alpha)*np.sin(beta)*np.cos(gamma)*z + np.sin(alpha)*np.sin(gamma)*z
        ny = np.sin(alpha)*np.cos(beta)*x + np.sin(alpha)*np.sin(beta)*np.sin(gamma)*y + np.cos(alpha)*np.cos(gamma)*y + np.sin(alpha)*np.sin(beta)*np.cos(gamma)*z - np.cos(alpha)*np.sin(gamma)*z
        nz = -np.sin(beta)*x + np.cos(beta)*np.sin(gamma)*y + np.cos(beta)*np.cos(gamma)*z
        # vertice to voxel data
        nx = int(round(nx + centroid2[2]))
        ny = int(round(ny + centroid2[1]))
        nz = int(round(nz + centroid2[0]))
        nx = min(max(0,nx),xsh-1)
        ny = min(max(0,ny),ysh-1)
        nz = min(max(0,nz),zsh-1)
        arr1[int(verts1[k,0]), int(verts1[k,2]), int(verts1[k,1])] = arr2[nz, nx, ny]
    return arr1


@numba.jit(nopython=True)
def true_rotate_image_inv(verts2, centroid1, centroid2, alpha, beta, gamma, arr1, arr2):
    zsh, xsh, ysh = arr1.shape
    # convert to bogenmass
    alpha = -alpha/180*np.pi
    beta = -beta/180*np.pi
    gamma = -gamma/180*np.pi
    # rotate vertices
    for k in range(verts2.shape[0]):
        z = verts2[k,0] - centroid2[0]
        y = verts2[k,1] - centroid2[1]
        x = verts2[k,2] - centroid2[2]
        # rotate vertice
        nx = np.cos(alpha)*np.cos(beta)*x - np.sin(alpha)*np.cos(beta)*y + np.sin(beta)*z
        ny = np.sin(alpha)*np.cos(gamma)*x + np.cos(alpha)*np.sin(beta)*np.sin(gamma)*x + np.cos(alpha)*np.cos(gamma)*y - np.sin(alpha)*np.sin(beta)*np.sin(gamma)*y - np.cos(beta)*np.sin(gamma)*z
        nz = np.sin(alpha)*np.sin(gamma)*x - np.cos(alpha)*np.sin(beta)*np.cos(gamma)*x + np.cos(alpha)*np.sin(gamma)*y + np.sin(alpha)*np.sin(beta)*np.cos(gamma)*y + np.cos(beta)*np.cos(gamma)*z
        # vertice to voxel data
        nx = int(round(nx + centroid1[2]))
        ny = int(round(ny + centroid1[1]))
        nz = int(round(nz + centroid1[0]))
        nx = min(max(0,nx),xsh-1)
        ny = min(max(0,ny),ysh-1)
        nz = min(max(0,nz),zsh-1)
        arr2[int(verts2[k,0]), int(verts2[k,2]), int(verts2[k,1])] = arr1[nz, nx, ny]
    return arr2


@numba.jit(nopython=True)
def where(mask, xmin=0, xmax=np.inf, ymin=0, ymax=np.inf, zmin=0, zmax=np.inf):
    zsh, ysh, xsh = mask.shape
    xmax = min(xmax, xsh)
    ymax = min(ymax, ysh)
    zmax = min(zmax, zsh)
    size = 0
    for k in range(zmin,zmax):
        for l in range(ymin,ymax):
            for m in range(xmin,xmax):
                if mask[k,l,m]>0:
                    size += 1
    verts = np.zeros((size, 3))
    i = 0
    for k in range(zmin,zmax):
        for l in range(ymin,ymax):
            for m in range(xmin,xmax):
                if mask[k,l,m]>0:
                    verts[i,0] = k
                    verts[i,1] = m
                    verts[i,2] = l
                    i += 1
    return verts


@numba.jit(nopython=True)
def correct_particle(verts1, centroid1, centroid2, alpha, beta, gamma, arr1, arr2, mask2, lock1, lock2):
    zsh, xsh, ysh = arr2.shape
    # convert to bogenmass
    alpha = alpha/180*np.pi
    beta = beta/180*np.pi
    gamma = gamma/180*np.pi
    # rotate vertices
    for k in range(verts1.shape[0]):
        z = verts1[k,0] - centroid1[0]
        y = verts1[k,1] - centroid1[1]
        x = verts1[k,2] - centroid1[2]
        # rotate vertice
        nx = np.cos(alpha)*np.cos(beta)*x + np.cos(alpha)*np.sin(beta)*np.sin(gamma)*y - np.sin(alpha)*np.cos(gamma)*y + np.cos(alpha)*np.sin(beta)*np.cos(gamma)*z + np.sin(alpha)*np.sin(gamma)*z
        ny = np.sin(alpha)*np.cos(beta)*x + np.sin(alpha)*np.sin(beta)*np.sin(gamma)*y + np.cos(alpha)*np.cos(gamma)*y + np.sin(alpha)*np.sin(beta)*np.cos(gamma)*z - np.cos(alpha)*np.sin(gamma)*z
        nz = -np.sin(beta)*x + np.cos(beta)*np.sin(gamma)*y + np.cos(beta)*np.cos(gamma)*z
        # vertice to voxel data
        nx = int(round(nx + centroid2[2]))
        ny = int(round(ny + centroid2[1]))
        nz = int(round(nz + centroid2[0]))
        #nx = min(max(0,nx),xsh-1)
        #ny = min(max(0,ny),ysh-1)
        #nz = min(max(0,nz),zsh-1)
        # correct particle
        if 0<=nx<xsh and 0<=ny<ysh and 0<=nz<zsh:
            z, y, x = verts1[k]
            if arr1[z,x,y]>0 and mask2[nz,nx,ny]==0:
                arr1[z,x,y] = 0
            elif arr1[z,x,y]>0 and lock2[nz,nx,ny]==1:
                arr1[z,x,y] = 0
            elif arr1[z,x,y]>0 and arr2[nz,nx,ny]>0:
                lock1[z,x,y] = 1
            elif arr1[z,x,y]==0 and lock1[z,x,y]==0:
                arr1[z,x,y] = arr2[nz,nx,ny]
    return arr1, lock1


@numba.jit(nopython=True)
def correct_particle_inv(verts2, centroid1, centroid2, alpha, beta, gamma, arr1, arr2, mask1, lock1, lock2):
    zsh, xsh, ysh = arr1.shape
    # convert to bogenmass
    alpha = -alpha/180*np.pi
    beta = -beta/180*np.pi
    gamma = -gamma/180*np.pi
    # rotate vertices
    for k in range(verts2.shape[0]):
        z = verts2[k,0] - centroid2[0]
        y = verts2[k,1] - centroid2[1]
        x = verts2[k,2] - centroid2[2]
        # rotate vertice
        nx = np.cos(alpha)*np.cos(beta)*x - np.sin(alpha)*np.cos(beta)*y + np.sin(beta)*z
        ny = np.sin(alpha)*np.cos(gamma)*x + np.cos(alpha)*np.sin(beta)*np.sin(gamma)*x + np.cos(alpha)*np.cos(gamma)*y - np.sin(alpha)*np.sin(beta)*np.sin(gamma)*y - np.cos(beta)*np.sin(gamma)*z
        nz = np.sin(alpha)*np.sin(gamma)*x - np.cos(alpha)*np.sin(beta)*np.cos(gamma)*x + np.cos(alpha)*np.sin(gamma)*y + np.sin(alpha)*np.sin(beta)*np.cos(gamma)*y + np.cos(beta)*np.cos(gamma)*z
        # vertice to voxel data
        nx = int(round(nx + centroid1[2]))
        ny = int(round(ny + centroid1[1]))
        nz = int(round(nz + centroid1[0]))
        #nx = min(max(0,nx),xsh-1)
        #ny = min(max(0,ny),ysh-1)
        #nz = min(max(0,nz),zsh-1)
        # correct particle
        if 0<=nx<xsh and 0<=ny<ysh and 0<=nz<zsh:
            z, y, x = verts2[k]
            if arr2[z,x,y]>0 and mask1[nz,nx,ny]==0:
                arr2[z,x,y] = 0
            elif arr2[z,x,y]>0 and lock1[nz,nx,ny]==1:
                arr2[z,x,y] = 0
            elif arr2[z,x,y]>0 and arr1[nz,nx,ny]>0:
                lock2[z,x,y] = 1
            elif arr2[z,x,y]==0 and lock2[z,x,y]==0:
                arr2[z,x,y] = arr1[nz,nx,ny]
    return arr2, lock2


def correct_match(p1, p2, m1, m2, l1, l2, alpha, beta, gamma, rank=0, gpu=True):
    # get vertices of particles for centroid
    verts1 = where(p1)
    verts2 = where(p2)
    '''z, x, y = np.where(p1)
    verts1 = np.zeros((z.size, 3))
    for k in range(z.size):
        verts1[k] = (z[k],y[k],x[k])
    z, x, y = np.where(p2)
    verts2 = np.zeros((z.size, 3))
    for k in range(z.size):
        verts2[k] = (z[k],y[k],x[k])'''

    # get centroids
    centroid1 = (np.mean(verts1[:,0]), np.mean(verts1[:,1]), np.mean(verts1[:,2]))
    centroid2 = (np.mean(verts2[:,0]), np.mean(verts2[:,1]), np.mean(verts2[:,2]))

    # get vertices of masks
    verts1 = where(m1).astype(int)
    verts2 = where(m2).astype(int)
    '''z, x, y = np.where(m1)
    verts1 = np.zeros((z.size, 3), dtype=int)
    for k in range(z.size):
        verts1[k] = (z[k],y[k],x[k])
    z, x, y = np.where(m2)
    verts2 = np.zeros((z.size, 3), dtype=int)
    for k in range(z.size):
        verts2[k] = (z[k],y[k],x[k])'''

    # correct particles
    if gpu:
        import pycuda.driver as cuda
        cuda.init()
        dev = cuda.Device(rank)
        ctx = dev.make_context()
        from biomedisa.features.matching.correct_particle_gpu import correct_particle
        p1, p2 = correct_particle(verts1, verts2, centroid1, centroid2, alpha, beta, gamma, p1, p2, m1, m2, l1, l2)
        ctx.pop()
        ctx.detach()
        del ctx
    else:
        l0 = l1.copy()
        p1, l1 = correct_particle(verts1, centroid1, centroid2, alpha, beta, gamma, p1, p2, m2, l1, l2)
        p2, l2 = correct_particle_inv(verts2, centroid1, centroid2, alpha, beta, gamma, p1, p2, m1, l0, l2)

    return p1, p2


def correct_match_majority(p1, p2, m1, alpha, beta, gamma, rank=0, inverse=False):
    # get vertices of particles for centroid
    verts1 = where(p1)
    verts2 = where(p2)
    '''z, x, y = np.where(p1)
    verts1 = np.zeros((z.size, 3))
    for k in range(z.size):
        verts1[k] = (z[k],y[k],x[k])
    z, x, y = np.where(p2)
    verts2 = np.zeros((z.size, 3))
    for k in range(z.size):
        verts2[k] = (z[k],y[k],x[k])'''

    # get centroids
    centroid1 = (np.mean(verts1[:,0]), np.mean(verts1[:,1]), np.mean(verts1[:,2]))
    centroid2 = (np.mean(verts2[:,0]), np.mean(verts2[:,1]), np.mean(verts2[:,2]))

    # get vertices of masks
    verts1 = where(m1).astype(int)
    '''z, x, y = np.where(m1)
    verts1 = np.zeros((z.size, 3), dtype=int)
    for k in range(z.size):
        verts1[k] = (z[k],y[k],x[k])'''

    # correct particles
    import pycuda.driver as cuda
    cuda.init()
    dev = cuda.Device(rank)
    ctx = dev.make_context()
    from biomedisa.features.matching.correct_particle_majority_gpu import correct_particle
    p12 = correct_particle(verts1, centroid1, centroid2, alpha, beta, gamma, p1, p2, inverse)
    ctx.pop()
    ctx.detach()
    del ctx

    return p12.astype(np.int8)


def register_particles(img1_path, img2_path, result1_path, result2_path, mappings_path):
    from biomedisa.features.matching.match_particles import reduce_blocksize_fast, fill_fast
    import xarray as xr
    # load data
    img1,_=load_data(img1_path)
    img2,_=load_data(img2_path)
    result1,_=load_data(result1_path)
    result2,_=load_data(result2_path)
    mappings = np.load(mappings_path)
    result_dir = os.path.dirname(result1)

    # loop over particles
    nop = mappings.shape[0]
    for k in range(nop):
        result_val1 = mappings[k,0]
        result_val2 = mappings[k,1]
        print(f'{k+1}/{nop}', int(result_val1), '->', int(result_val2))

        # crop data
        argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = reduce_blocksize_fast(result1, result_val1)
        p1 = np.zeros((argmax_z-argmin_z,argmax_y-argmin_y,argmax_x-argmin_x), dtype=np.uint8)
        p1[result1[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x]==result_val1]=1
        i1 = img1[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x].copy(order='C')*p1
        da = xr.DataArray(i1, dims=['time', 'lat', 'lon'])
        da.to_netcdf(result_dir+f'/img_{int(result_val1)}.nc')
        #p1 = fill_fast(p1)
        p1_full = p1.copy()

        argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = reduce_blocksize_fast(result2, result_val2)
        p2 = np.zeros((argmax_z-argmin_z,argmax_y-argmin_y,argmax_x-argmin_x), dtype=np.uint8)
        p2[result2[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x]==result_val2]=1
        i2 = img2[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x].copy(order='C')*p2
        da = xr.DataArray(i2, dims=['time', 'lat', 'lon'])
        da.to_netcdf(result_dir+f'/img_{int(result_val1)}_red.nc')
        #p2 = fill_fast(p2)
        p2_full = p2.copy()

        # scale particles
        p1_size = np.sum(p1)
        p2_size = np.sum(p2)
        max_size = 10000
        zoom_factor = (max_size / p1_size)**(1/3)
        p1 = ndimage.zoom(p1, zoom_factor, order=0)
        zoom_factor = (max_size / p2_size)**(1/3)
        p2 = ndimage.zoom(p2, zoom_factor, order=0)

        # find best rotation 12
        TIC = time.time()
        rot_dice, alpha12, beta12, gamma12 = rotation_dice(p1, p2, None, None, None, 0, True)
        #print(round(rot_dice,4), alpha12, beta12, gamma12)
        #print('Calculation time:', int(time.time() - TIC), 'sec')

        # scale particles to same size
        zoom_factor = (p1_size / p2_size)**(1/3)
        p2_full = ndimage.zoom(p2_full, zoom_factor, order=0)
        i2 = ndimage.zoom(i2, zoom_factor, order=0)
        print(np.sum(p1_full), np.sum(p2_full))

        # refine rotation 12
        TIC = time.time()
        rot_dice, alpha12, beta12, gamma12 = rotation_dice(p1_full, p2_full, alpha12, beta12, gamma12, 0, True)
        #print(round(rot_dice,4), alpha12, beta12, gamma12)
        #print('Calculation time:', int(time.time() - TIC), 'sec')

        # align image and label data
        verts1 = where(p1_full)
        verts2 = where(p2_full)

        # get centroids
        centroid1 = (np.mean(verts1[:,0]), np.mean(verts1[:,1]), np.mean(verts1[:,2]))
        centroid2 = (np.mean(verts2[:,0]), np.mean(verts2[:,1]), np.mean(verts2[:,2]))

        verts1 = where(np.ones_like(p1_full))

        # centralize
        #p1_rot = true_rotate_image_inv(verts2, centroid1, centroid2, alpha12, beta12, gamma12, p1_full, np.zeros_like(p2_full))
        i2_rot = true_rotate_image(verts1, centroid1, centroid2, alpha12, beta12, gamma12, i2, np.zeros_like(i1))

        # save data
        #imwrite(result_dir+f'/img_{int(result_val1)}_red_scaled={round(p1_size / p2_size,3)}_rot.tif', i2_rot)
        #save_data(result_dir+f'/mask_{int(result_val1)}.nrrd', p1_rot)
        #save_data(result_dir+f'/mask_{int(result_val1)}_red_scale={round(p1_size / p2_size,3)}.nrrd', p2_full)

        # Save to NetCDF
        da = xr.DataArray(i2_rot, dims=['time', 'lat', 'lon'])
        da.to_netcdf(result_dir+f'/img_{int(result_val1)}_red_scaled={round(p1_size / p2_size,3)}_rot.nc')


if __name__ == "__main__":
    from biomedisa.matching import reduce_blocksize_fast, fill_fast

    # values to match
    result_val1 = 461
    result_val2 = 426

    # optional particle correction
    result_val3 = None

    # load and crop data
    result1,_ = load_data(sys.argv[1])
    argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = reduce_blocksize_fast(result1, result_val1)
    p1 = np.zeros((argmax_z-argmin_z,argmax_y-argmin_y,argmax_x-argmin_x), dtype=np.int8)
    p1[result1[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x]==result_val1]=1
    p1 = fill_fast(p1)
    p1_full = p1.copy()
    del result1

    if result_val3:
        mask1,_ = load_data(sys.argv[4])
        m1 = np.zeros_like(p1_full)
        m1[mask1[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x]>0]=1
        del mask1

    result2,_ = load_data(sys.argv[2])
    argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = reduce_blocksize_fast(result2, result_val2)
    p2 = np.zeros((argmax_z-argmin_z,argmax_y-argmin_y,argmax_x-argmin_x), dtype=np.int8)
    p2[result2[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x]==result_val2]=1
    p2 = fill_fast(p2)
    p2_full = p2.copy()
    del result2

    if result_val3:
        result3,_ = load_data(sys.argv[3])
        argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = reduce_blocksize_fast(result3, result_val3)
        p3 = np.zeros((argmax_z-argmin_z,argmax_y-argmin_y,argmax_x-argmin_x), dtype=np.int8)
        p3[result3[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x]==result_val3]=1
        p3 = fill_fast(p3)
        p3_full = p3.copy()
        del result3

    # down scale large particles
    p1_size = np.sum(p1)
    max_size = 10000
    if p1_size > max_size:
        zoom_factor = (max_size / p1_size)**(1/3)
        p1 = ndimage.zoom(p1, zoom_factor, order=0)
        p2 = ndimage.zoom(p2, zoom_factor, order=0)
        if result_val3:
            p3 = ndimage.zoom(p3, zoom_factor, order=0)

    # print particle sizes
    if result_val3:
        print('Sizes:', np.sum(p1), np.sum(p2), np.sum(p3))
    print('Sizes:', np.sum(p1), np.sum(p2))

    # find best rotation 12
    TIC = time.time()
    rot_dice, alpha12, beta12, gamma12 = rotation_dice(p1, p2, None, None, None, 0, True)
    print(round(rot_dice,4), alpha12, beta12, gamma12)
    print('Calculation time:', int(time.time() - TIC), 'sec')

    # refine rotation 12
    if p1_size > max_size:
        TIC = time.time()
        rot_dice, alpha12, beta12, gamma12 = rotation_dice(p1_full, p2_full, alpha12, beta12, gamma12, 0, True)
        print(round(rot_dice,4), alpha12, beta12, gamma12)
        print('Calculation time:', int(time.time() - TIC), 'sec')

    if result_val3:
        # find best rotation 13
        TIC = time.time()
        rot_dice, alpha13, beta13, gamma13 = rotation_dice(p1, p3, None, None, None, 0, True)
        print(round(rot_dice,4), alpha13, beta13, gamma13)
        print('Calculation time:', int(time.time() - TIC), 'sec')

        # refine rotation 13
        TIC = time.time()
        rot_dice, alpha13, beta13, gamma13 = rotation_dice(p1_full, p3_full, alpha13, beta13, gamma13, 0, True)
        print(round(rot_dice,4), alpha13, beta13, gamma13)
        print('Calculation time:', int(time.time() - TIC), 'sec')

        # correct match
        print(np.sum(p1_full))
        p12 = correct_match_majority(p1_full, p2_full, m1, alpha12, beta12, gamma12, rank=0)
        p13 = correct_match_majority(p1_full, p3_full, m1, alpha13, beta13, gamma13, rank=0)
        p1_full += p12 + p13
        p1_full[p1_full<2]=0
        p1_full[p1_full>0]=1
        print(np.sum(p1_full))

