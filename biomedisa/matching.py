#!/usr/bin/python3
##########################################################################
##                                                                      ##
##  Copyright (c) 2019-2025 Philipp Lösel. All rights reserved.         ##
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
from biomedisa.features.biomedisa_helper import *
from biomedisa.features.matching.match_helper import rotation_dice, correct_match, correct_match_majority
from tifffile import imread
from pathlib import Path
from scipy import ndimage
import argparse
import numba
import sys
import time
import shutil
import glob
import zarr

@numba.jit(nopython=True)
def assign_labels(ref, result, labels_matrix):
    zsh, ysh, xsh = ref.shape
    for z in range(zsh):
        for y in range(ysh):
            for x in range(xsh):
                ref_value = ref[z,y,x]
                result_value = result[z,y,x]
                labels_matrix[ref_value,result_value] += 1
    return labels_matrix

@numba.jit(nopython=True)
def dice_score_(ref, result, ref_val, result_val):
    ref = ref.ravel()
    result = result.ravel()
    numerator = 0
    denominator = 0
    for k in range(ref.size):
        if ref[k]==ref_val and result[k]==result_val:
            numerator += 1
        if ref[k]==ref_val:
            denominator += 1
        if result[k]==result_val:
            denominator += 1
    dice = 2 * numerator / denominator
    return dice

@numba.jit(nopython=True)
def reduce_blocksize_fast(data, value=1, buff=10):
    zsh, ysh, xsh = data.shape
    argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = zsh, 0, ysh, 0, xsh, 0
    for k in range(zsh):
        for l in range(ysh):
            for m in range(xsh):
                if data[k,l,m]==value:
                    argmin_x = min(argmin_x, m)
                    argmax_x = max(argmax_x, m)
                    argmin_y = min(argmin_y, l)
                    argmax_y = max(argmax_y, l)
                    argmin_z = min(argmin_z, k)
                    argmax_z = max(argmax_z, k)
    argmin_x = max(argmin_x - buff, 0)
    argmax_x = min(argmax_x + buff, xsh)
    argmin_y = max(argmin_y - buff, 0)
    argmax_y = min(argmax_y + buff, ysh)
    argmin_z = max(argmin_z - buff, 0)
    argmax_z = min(argmax_z + buff, zsh)
    return argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x

@numba.jit(nopython=True)
def get_bounding_boxes(data, labels=None, buff=10):
    zsh, ysh, xsh = data.shape
    max_label = np.int32(np.amax(data))
    boxes = np.zeros((max_label, 6), dtype=np.uint32)
    for k in range(max_label):
        boxes[k] = zsh, 0, ysh, 0, xsh, 0
    for k in range(zsh):
        for l in range(ysh):
            for m in range(xsh):
                label = data[k,l,m]
                if label>0:
                    index = label - 1
                    boxes[index,0] = min(boxes[index,0], k)
                    boxes[index,1] = max(boxes[index,1], k)
                    boxes[index,2] = min(boxes[index,2], l)
                    boxes[index,3] = max(boxes[index,3], l)
                    boxes[index,4] = min(boxes[index,4], m)
                    boxes[index,5] = max(boxes[index,5], m)
    for k in range(max_label):
        boxes[k,0] = max(boxes[k,0] - buff, 0)
        boxes[k,1] = min(boxes[k,1] + buff, zsh)
        boxes[k,2] = max(boxes[k,2] - buff, 0)
        boxes[k,3] = min(boxes[k,3] + buff, ysh)
        boxes[k,4] = max(boxes[k,4] - buff, 0)
        boxes[k,5] = min(boxes[k,5] + buff, xsh)
    return boxes

def fill_fast(particle):
    '''particle must be binary segmentation'''
    # invert
    particle = 1 - particle

    # get clusters
    s = [[[0,0,0], [0,1,0], [0,0,0]], [[0,1,0], [1,1,1], [0,1,0]], [[0,0,0], [0,1,0], [0,0,0]]]
    labeled_array, _ = ndimage.label(particle, structure=s)
    size = np.bincount(labeled_array.ravel())
    biggest_label = np.argmax(size)

    # get label with all holes filled
    particle.fill(1)
    particle[labeled_array==biggest_label] = 0

    return particle

def clean_fast(particle):
    '''particle must be binary segmentation'''
    s = [[[0,0,0], [0,1,0], [0,0,0]], [[0,1,0], [1,1,1], [0,1,0]], [[0,0,0], [0,1,0], [0,0,0]]]
    labeled_array, _ = ndimage.label(particle, structure=s)
    labeled_array += 1
    labeled_array *= particle
    labels, counts = unique(labeled_array, return_counts=True)
    arg = np.argmax(counts[1:])+1
    particle[labeled_array!=labels[arg]] = 0
    return particle

@numba.jit(nopython=True)
def remove_small_particles2(labeled_array, label_vals):
    zsh, ysh, xsh = labeled_array.shape
    for z in range(zsh):
        for y in range(ysh):
            for x in range(xsh):
                labeled_array[z,y,x] *= label_vals[labeled_array[z,y,x]]
    return labeled_array

@numba.jit(nopython=True)
def get_centroid(img):
    zsh, ysh, xsh = img.shape
    centroid_x = 0
    centroid_y = 0
    centroid_z = 0
    counter = 0
    for z in range(zsh):
        for y in range(ysh):
            for x in range(xsh):
                if img[z,y,x]:
                    centroid_z += z
                    centroid_y += y
                    centroid_x += x
                    counter += 1
    centroid_z = centroid_z / counter
    centroid_y = centroid_y / counter
    centroid_x = centroid_x / counter
    return (centroid_x, centroid_y, centroid_z)

@numba.jit(nopython=True)
def get_distances(img, centroid):
    zsh, ysh, xsh = img.shape
    counter = 0
    for z in range(1,zsh-1):
        for y in range(1,ysh-1):
            for x in range(1,xsh-1):
                if img[z,y,x]:
                    if img[z-1,y,x]==0 or img[z+1,y,x]==0 or img[z,y-1,x]==0 or img[z,y+1,x]==0 or img[z,y,x-1]==0 or img[z,y,x+1]==0:
                        counter += 1
    points = np.zeros((3,counter))
    counter = 0
    for z in range(1,zsh-1):
        for y in range(1,ysh-1):
            for x in range(1,xsh-1):
                if img[z,y,x]:
                    if img[z-1,y,x]==0 or img[z+1,y,x]==0 or img[z,y-1,x]==0 or img[z,y+1,x]==0 or img[z,y,x-1]==0 or img[z,y,x+1]==0:
                        points[0,counter] = z
                        points[1,counter] = y
                        points[2,counter] = x
                        counter += 1
    distances = np.zeros((counter))
    for k in range(counter):
        distances[k] = np.sqrt((points[0,k]-centroid[2])**2 + (points[1,k]-centroid[1])**2 + (points[2,k]-centroid[0])**2)
    return distances

@numba.jit(nopython=True)
def change_label_values(data, ref):
    zsh, ysh, xsh = data.shape
    #c = np.zeros_like(data)
    for z in range(zsh):
        for y in range(ysh):
            for x in range(xsh):
                data[z,y,x] = ref[data[z,y,x]]
    return data

@numba.jit(nopython=True)
def matched_particles(result, labels_array):
    zsh, ysh, xsh = result.shape
    #unmatched = result.copy()
    for z in range(zsh):
        for y in range(ysh):
            for x in range(xsh):
                result[z,y,x] *= labels_array[result[z,y,x]]
                #unmatched[z,y,x] *= 1 - labels_array[result[z,y,x]]
    return result#, unmatched

def get_gpu_count():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,nounits,noheader'], encoding='utf-8')
        gpu_list = result.strip().split('\n')
        return len(gpu_list)
    except:
        print('Warning: No NVIDIA GPU detected. Defaulting to 1 device.')
        return 1

@numba.jit(nopython=True)
def nearest_neighbour(labeled_array, mask, nearest_indices):
    zsh, ysh, xsh = labeled_array.shape
    for k in range(zsh):
        for l in range(ysh):
            for m in range(xsh):
                if mask[k,l,m]==1 and labeled_array[k,l,m]==0:
                    z = nearest_indices[0,k,l,m]
                    y = nearest_indices[1,k,l,m]
                    x = nearest_indices[2,k,l,m]
                    labeled_array[k,l,m] = labeled_array[z,y,x]
    return labeled_array

def convert_to_zarr(file_path, compress=False):
    path_to_zarr = file_path + '.zarr'
    print(os.path.basename(file_path))
    if os.path.exists(path_to_zarr):
        shutil.rmtree(path_to_zarr)
    # load data
    data,_ = load_data(file_path)
    # Create a Zarr array with the specified chunk size
    zarr_array = zarr.create(
        shape=data.shape,
        chunks=(100, 100, 100),
        dtype=data.dtype,
        store=zarr.DirectoryStore(path_to_zarr),
        compressor=zarr.get_codec({'id': 'zlib', 'level': 5}) if compress else None
        )
    # Write the data into the Zarr array
    zarr_array[:] = data

@numba.jit(nopython=True)
def init_indices(nearest_indices):
    zsh, ysh, xsh = nearest_indices.shape[1:]
    for z in range(zsh):
        for y in range(ysh):
            for x in range(xsh):
                nearest_indices[0,z,y,x] = z
                nearest_indices[1,z,y,x] = y
                nearest_indices[2,z,y,x] = x
    return nearest_indices

@numba.jit(nopython=True)
def nearest_neighbour_indices(distances, mask, nearest_indices, iterations=3):
    sqrt2, sqrt3 = np.sqrt(2), np.sqrt(3)
    zsh, ysh, xsh = distances.shape
    for i in range(iterations):
        for z in range(1,zsh):
            for y in range(1,ysh-1):
                for x in range(1,xsh-1):
                  if mask[z,y,x]==1 and distances[z,y,x]>0:
                    a1 = distances[z-1,y-1,x-1] + sqrt3
                    a2 = distances[z-1,y-1,x] + sqrt2
                    a3 = distances[z-1,y-1,x+1] + sqrt3
                    a4 = distances[z-1,y,x-1] + sqrt2
                    a5 = distances[z-1,y,x] + 1
                    a6 = distances[z-1,y,x+1] + sqrt2
                    a7 = distances[z-1,y+1,x-1] + sqrt3
                    a8 = distances[z-1,y+1,x] + sqrt2
                    a9 = distances[z-1,y+1,x+1] + sqrt3
                    a10 = distances[z,y-1,x-1] + sqrt2
                    a11 = distances[z,y-1,x] + 1
                    a12 = distances[z,y-1,x+1] + sqrt2
                    a13 = distances[z,y,x-1] + 1
                    a14 = distances[z,y,x]
                    a = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14]
                    b = [(z-1,y-1,x-1),
                        (z-1,y-1,x),
                        (z-1,y-1,x+1),
                        (z-1,y,x-1),
                        (z-1,y,x),
                        (z-1,y,x+1),
                        (z-1,y+1,x-1),
                        (z-1,y+1,x),
                        (z-1,y+1,x+1),
                        (z,y-1,x-1),
                        (z,y-1,x),
                        (z,y-1,x+1),
                        (z,y,x-1),
                        (z,y,x)]
                    amin = a[0]
                    zmin,ymin,xmin = b[0]
                    for k in range(1,14):
                        if a[k] < amin:
                            amin = a[k]
                            zmin,ymin,xmin = b[k]
                    if amin < np.inf:
                        distances[z,y,x] = amin
                        nearest_indices[0,z,y,x] = nearest_indices[0,zmin,ymin,xmin]
                        nearest_indices[1,z,y,x] = nearest_indices[1,zmin,ymin,xmin]
                        nearest_indices[2,z,y,x] = nearest_indices[2,zmin,ymin,xmin]
        for z in range(zsh-2,-1,-1):
            for y in range(ysh-2,0,-1):
                for x in range(xsh-2,0,-1):
                  if mask[z,y,x]==1 and distances[z,y,x]>0:
                    a1 = distances[z+1,y-1,x-1] + sqrt3
                    a2 = distances[z+1,y-1,x] + sqrt2
                    a3 = distances[z+1,y-1,x+1] + sqrt3
                    a4 = distances[z+1,y,x-1] + sqrt2
                    a5 = distances[z+1,y,x] + 1
                    a6 = distances[z+1,y,x+1] + sqrt2
                    a7 = distances[z+1,y+1,x-1] + sqrt3
                    a8 = distances[z+1,y+1,x] + sqrt2
                    a9 = distances[z+1,y+1,x+1] + sqrt3
                    a10 = distances[z,y+1,x+1] + sqrt2
                    a11 = distances[z,y+1,x] + 1
                    a12 = distances[z,y+1,x-1] + sqrt2
                    a13 = distances[z,y,x+1] + 1
                    a14 = distances[z,y,x]
                    a = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14]
                    b = [(z+1,y-1,x-1),
                        (z+1,y-1,x),
                        (z+1,y-1,x+1),
                        (z+1,y,x-1),
                        (z+1,y,x),
                        (z+1,y,x+1),
                        (z+1,y+1,x-1),
                        (z+1,y+1,x),
                        (z+1,y+1,x+1),
                        (z,y+1,x+1),
                        (z,y+1,x),
                        (z,y+1,x-1),
                        (z,y,x+1),
                        (z,y,x)]
                    amin = a[0]
                    zmin,ymin,xmin = b[0]
                    for k in range(1,14):
                        if a[k] < amin:
                            amin = a[k]
                            zmin,ymin,xmin = b[k]
                    if amin < np.inf:
                        distances[z,y,x] = amin
                        nearest_indices[0,z,y,x] = nearest_indices[0,zmin,ymin,xmin]
                        nearest_indices[1,z,y,x] = nearest_indices[1,zmin,ymin,xmin]
                        nearest_indices[2,z,y,x] = nearest_indices[2,zmin,ymin,xmin]
    return nearest_indices

def get_data_size(path):
    suffix = '.nrrd'
    for s in ['_small.nrrd', '_half.nrrd']:
        if s in path:
            suffix = s
    if os.path.exists(path.replace(suffix,'_small.nrrd')):
        old_size = '_small'
        old_path = path.replace(suffix,'_small.nrrd')
    elif os.path.exists(path.replace(suffix,'_half.nrrd')):
        old_size = '_half'
        old_path = path.replace(suffix,'_half.nrrd')
    elif os.path.exists(path.replace(suffix,'.nrrd')):
        old_size = ''
        old_path = path.replace(suffix,'.nrrd')
    else:
        old_size, old_path = None, None
    return old_size, old_path

@numba.jit(nopython=True)
def upscale_label(downscaled, upscaled):
    downscaled_size = downscaled.shape
    # Calculate scaling factors
    scale_z = upscaled.shape[0] / downscaled_size[0]
    scale_y = upscaled.shape[1] / downscaled_size[1]
    scale_x = upscaled.shape[2] / downscaled_size[2]
    # Iterate over each pixel in the downscaled array
    for z in range(downscaled_size[0]):
        for y in range(downscaled_size[1]):
            for x in range(downscaled_size[2]):
                value = downscaled[z,y,x]
                if value > 0:
                    x_start = int(x * scale_x)
                    x_end = int((x + 1) * scale_x)
                    y_start = int(y * scale_y)
                    y_end = int((y + 1) * scale_y)
                    z_start = int(z * scale_z)
                    z_end = int((z + 1) * scale_z)
                    upscaled[z_start:z_end, y_start:y_end, x_start:x_end] = value
    return upscaled

@numba.jit(nopython=True)
def boundary(arr):
    boundaries = np.zeros(arr.shape, np.uint8)
    zsh, ysh, xsh = arr.shape
    for k in range(1,zsh-1):
        for l in range(1,ysh-1):
            for m in range(1,xsh-1):
                value = arr[k,l,m]
                g = abs(arr[k-1,l,m]-value) + abs(arr[k+1,l,m]-value) + abs(arr[k,l-1,m]-value) + abs(arr[k,l+1,m]-value) + abs(arr[k,l,m-1]-value) + abs(arr[k,l,m+1]-value)
                if g > 0:
                    boundaries[k,l,m] = 1
    return boundaries

def generate_border(image, sigma=1, threshold=0.25):
    from scipy.ndimage import gaussian_filter
    boundaries = boundary(image)
    boundaries = boundaries.astype(np.float32)
    boundary_image = gaussian_filter(boundaries, sigma)
    boundary_image[boundary_image>threshold]=1
    boundary_image[boundary_image<1]=0
    return boundary_image.astype(np.uint8)

def create_boundary_labels(result, mask):
    boundary_image = generate_border(result, sigma=1, threshold=0.25)

    # enlarge mask
    boundary_mask = generate_border(mask, sigma=1, threshold=0.25)
    mask[boundary_mask>0]=1

    # result and result boundaries are known areas
    mask[result>0]=0
    mask[boundary_image>0]=0

    # remove outliers
    s = [[[0,0,0], [0,1,0], [0,0,0]], [[0,1,0], [1,1,1], [0,1,0]], [[0,0,0], [0,1,0], [0,0,0]]]
    mask = mask.astype(np.uint32)
    ndimage.label(mask, structure=s, output=mask)
    lv, ln = unique(mask, return_counts=True)
    label_vals = np.zeros(np.amax(lv)+1, dtype=np.uint32)
    for i,l in enumerate(lv):
        if ln[i]>=100:
            label_vals[l] = 1
    mask = remove_small_particles2(mask, label_vals)
    mask[mask>0]=1
    mask = mask.astype(np.uint8)

    # invert mask
    ignore_mask = 1 - mask

    # save as multi channel label
    volume = np.stack([boundary_image, ignore_mask], axis=-1)
    return volume

@numba.jit(nopython=True)
def remove_container(a,xs,ys,ds):
    res = 15
    size = 2
    ysh,xsh = a.shape
    min_counter = np.inf
    for d in np.linspace(-size,size,res):
        for yi in np.linspace(-size,size,res):
            for xi in np.linspace(-size,size,res):
                r = (ds+d)/2
                cx = xs + xi
                cy = ys + yi
                # find smallest ring with no zeros
                counter = 0
                for y in range(ysh):
                    for x in range(xsh):
                        if r**2 <= (y-cy)**2 + (x-cx)**2 < (r+10)**2 and a[y,x]==0:
                            counter += 1
                if counter < min_counter:
                    min_counter = counter
                    min_d = (ds+d)
                    min_x = cx
                    min_y = cy
    # remove container
    for y in range(ysh):
        for x in range(xsh):
            if ((min_d-3)/2)**2 <= (y-min_y)**2 + (x-min_x)**2:
                a[y,x]=0
    return a,min_x,min_y,min_d

def label_in_ascending_order(label):
    lv = unique(label)
    print('Labels:', len(lv))
    ref = np.zeros(np.amax(lv)+1, dtype=np.int32)
    for i, v in enumerate(lv):
        ref[v] = i
    label = change_label_values(label, ref)
    if np.amax(label) <= 65535:
        label = label.astype(np.uint16)
    return label

@numba.jit(nopython=True)
def get_boundary_sizes(arr, total_boundary, embedded_boundary):
    zsh, ysh, xsh = arr.shape
    for k in range(1,zsh-1):
        for l in range(1,ysh-1):
            for m in range(1,xsh-1):
                value = arr[k,l,m]
                if value>0:
                    g = abs(arr[k-1,l,m]-value) \
                        + abs(arr[k+1,l,m]-value) \
                        + abs(arr[k,l-1,m]-value) \
                        + abs(arr[k,l+1,m]-value) \
                        + abs(arr[k,l,m-1]-value) \
                        + abs(arr[k,l,m+1]-value)
                    if g > 0:
                        total_boundary[value] += 1
                    g = min(1,float(arr[k-1,l,m]))*abs(arr[k-1,l,m]-value) \
                        + min(1,float(arr[k+1,l,m]))*abs(arr[k+1,l,m]-value) \
                        + min(1,float(arr[k,l-1,m]))*abs(arr[k,l-1,m]-value) \
                        + min(1,float(arr[k,l+1,m]))*abs(arr[k,l+1,m]-value) \
                        + min(1,float(arr[k,l,m-1]))*abs(arr[k,l,m-1]-value) \
                        + min(1,float(arr[k,l,m+1]))*abs(arr[k,l,m+1]-value)
                    if g > 0:
                        embedded_boundary[value] += 1
    return total_boundary, embedded_boundary

@numba.jit(nopython=True)
def is_mostly_inside(arr, value):
    zsh, ysh, xsh = arr.shape
    labels = np.zeros(np.amax(arr)+1, dtype=np.int32)
    for k in range(1,zsh-1):
        for l in range(1,ysh-1):
            for m in range(1,xsh-1):
                if arr[k,l,m]==value:
                    if arr[k-1,l,m]!=value:
                        labels[arr[k-1,l,m]] += 1
                    elif arr[k+1,l,m]!=value:
                        labels[arr[k+1,l,m]] += 1
                    elif arr[k,l-1,m]!=value:
                        labels[arr[k,l-1,m]] += 1
                    elif arr[k,l+1,m]!=value:
                        labels[arr[k,l+1,m]] += 1
                    elif arr[k,l,m-1]!=value:
                        labels[arr[k,l,m-1]] += 1
                    elif arr[k,l,m+1]!=value:
                        labels[arr[k,l,m+1]] += 1
    labels[0]=0
    return labels

if __name__ == "__main__":

    # initialize arguments
    parser = argparse.ArgumentParser(description='Biomedisa matching.',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # arguments
    parser.add_argument('-m','--model', type=str, default='implicit',
                        help='Model used (sam, explicit, implicit)')
    parser.add_argument('-pr','--project', type=str,
                        help='Project name')
    parser.add_argument('-d','--datasets', type=str, nargs='+',
                        help='List of dataset paths')
    parser.add_argument('-ld','--labelDatasets', type=str, nargs='+', default=None,
                        help='List of pre-calculated label dataset paths')
    parser.add_argument('--sample', type=int, default=None,
                        help='If not given, calculate all')
    parser.add_argument('-s','--step', type=int, default=None,
                        help='If not given, use last available')
    parser.add_argument('-mp','--model_path', type=str, metavar='PATH', default=None,
                        help='Location of model for prediction (.h5)')
    parser.add_argument('-cm','--create_mask', nargs='?', type=int, const=13000, default=None,
                        help='binarize volume using a threshold')
    parser.add_argument('-rc','--remove_container', action='store_true', default=False,
                        help='remove_container')
    parser.add_argument('-pm','--positive_mask', action='store_true', default=False,
                        help='use unmatched areas to create a new positive mask for prediction')
    parser.add_argument('-p','--predict_boundaries', action='store_true', default=False,
                        help='predict boundaries')
    parser.add_argument('-l','--label_particles', action='store_true', default=False,
                        help='label particles individually')
    parser.add_argument('-dc','--distances', action='store_true', default=False,
                        help='distances to centroid')
    parser.add_argument('-mp2','--match_particles', action='store_true', default=False,
                        help='determine best particle based on distances to centroid')
    parser.add_argument('-r','--rot_dice', action='store_true', default=False,
                        help='determine best rotation dice for matched particles')
    parser.add_argument('-lmp','--label_matched_particles', action='store_true', default=False,
                        help='determine best particle based on distances to centroid')
    parser.add_argument('-cp','--correct_particles', action='store_true', default=False,
                        help='determine best particle based on distances to centroid')
    parser.add_argument('-ma','--matched_area', action='store_true', default=False,
                        help='determine size of matched area')
    parser.add_argument('-f','--fill_labels', action='store_true', default=False,
                        help='fill labels')
    parser.add_argument('-t','--train', action='store_true', default=False,
                        help='train boundary detection')
    parser.add_argument('-ps','--patch_size', type=int, default=16,
                        help='patch size')
    parser.add_argument('-ss','--stride_size', type=int, default=8,
                        help='stride size')
    parser.add_argument('-sl','--scale_labels', action='store_true', default=False,
                        help='Downsize full resolution data for matching process')
    bm = parser.parse_args()

    #=======================================================================================
    # datasets
    #=======================================================================================
    min_particle_size = 1000

    # datasets
    BASE = os.path.dirname(bm.datasets[0])
    for k, dataset in enumerate(bm.datasets):
        bm.datasets[k] = os.path.basename(dataset).replace('.tif','')
    n_datasets = len(bm.datasets)

    # step (define step manually or use maximum existing)
    if bm.step==None:
        bm.step = 0
        liste = glob.glob(BASE+f'/{bm.project}/step=*')
        for l in liste:
            bm.step = max(bm.step, int(l.split('/')[-1].split('=')[-1]))
    print('Step:', bm.step)

    # sample (process all if None)
    if bm.sample!=None:
        print('Sample:', bm.sample)

    # make directories
    path_to_dir = BASE+f'/{bm.project}/step={bm.step}'
    path_to_meta = f'{path_to_dir}/meta'
    Path(path_to_meta).mkdir(parents=True, exist_ok=True)

    #=======================================================================================
    # create mask
    #=======================================================================================
    if bm.create_mask:
        print(f'Binarize image data using {bm.create_mask}. Please use "--create_mask=" to specify.')
        for dataset in bm.datasets:
            path_to_mask = BASE+f'/mask.{dataset}.tif'
            if not os.path.exists(path_to_mask):
                print(os.path.basename(path_to_mask))
                # threshold image data to create mask (does not influence separation since this is done by edge removal)
                path_to_img = BASE+f'/{dataset}.tif'
                img = imread(path_to_img)
                #img[img==np.amax(img)]=np.amin(img)
                print(img.shape)
                if '63microns' in dataset:
                    img[img<10200]=0
                else:
                    img[img<bm.create_mask]=0
                img[img>0]=1
                save_data(path_to_mask, img.astype(np.uint8))

    #=======================================================================================
    # remove container
    #=======================================================================================
    if bm.remove_container:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if bm.sample==0:
            path = BASE+f'/mask.Quartz_sphere_15_25_110mins_2_greaterthan63microns_uint16_half.tif'
            xs,ys,ds=1370/2,1371/2,2088/2
        if bm.sample==1:
            path = BASE+f'/mask.Quartz_sphere_15_25_110mins_2_greaterthan63microns_rescan_A_uint16_half.tif'
            xs,ys,ds=1372/2,1360/2,2090/2
        if bm.sample==2:
            path = BASE+f'/mask.Quartz_sphere_15_25_110mins_2_morethan63microns_rescan_3_uint16_half.tif'
            xs,ys,ds=1370/2,1364/2,2090/2

        zsh,ysh,xsh = TiffInfo(path).shape
        print(zsh,ysh,xsh)

        if rank==0:
            result = np.zeros((zsh, ysh, xsh), dtype=np.uint8)
        for k in range(rank,zsh,size):
            a = imread(path, key=range(k,k+1))
            a,xs,ys,ds = remove_container(a,xs,ys,ds)
            print(rank, k, round(xs,2),round(ys,2),round(ds,2))
            # gather results
            if rank==0:
                result[k] = a
                for source in range(1, size):
                    if k+source < zsh:
                        comm.Recv([result[k+source], MPI.SIGNED_CHAR], source=source, tag=k+source)
            else:
                comm.Send([a.copy(), MPI.SIGNED_CHAR], dest=0, tag=k)
        if rank==0:
            save_data(path.replace('.tif','_clean.tif'), result)

    #=======================================================================================
    # use unmatched areas to create a new positive mask for prediction
    #=======================================================================================
    if bm.positive_mask:
        for i, dataset in enumerate(bm.datasets):
          if bm.sample==None or i==bm.sample:
            new_mask_path = f'{path_to_dir}/mask.{dataset}.tif'
            if not os.path.exists(new_mask_path):
                print(os.path.basename(new_mask_path))

                # get previous data
                _, old_path = get_data_size(BASE+f'/{bm.project}/step={bm.step-1}/corr.{dataset}.nrrd')
                if not old_path:
                    raise RuntimeError("No previous result available.")
                matched,_ = load_data(old_path)
                mask = imread(BASE +'/'+ os.path.basename(old_path).replace('corr.','mask.').replace('.nrrd','.tif'))

                # unmatched areas
                unmatched = (mask > matched).astype(np.uint8)

                # remove outliers
                s = [[[0,0,0], [0,1,0], [0,0,0]], [[0,1,0], [1,1,1], [0,1,0]], [[0,0,0], [0,1,0], [0,0,0]]]
                unmatched = unmatched.astype(np.uint32)
                ndimage.label(unmatched, structure=s, output=unmatched)
                lv, ln = unique(unmatched, return_counts=True)
                label_vals = np.zeros(np.amax(lv)+1, dtype=np.uint32)
                for i,l in enumerate(lv):
                    if ln[i]>=100:
                        label_vals[l] = 1
                unmatched = remove_small_particles2(unmatched, label_vals)
                unmatched[unmatched>0]=1
                unmatched = unmatched.astype(np.uint8)

                # scale data
                upscale, downscale = False, False
                new_shape = TiffInfo(BASE+f'/mask.{dataset}.tif').shape
                if unmatched.shape != new_shape:
                    if all(a > b for a, b in zip(unmatched.shape, new_shape)):
                        downscale = True
                    elif all(a < b for a, b in zip(unmatched.shape, new_shape)):
                        upscale = True
                print(f'Upscale: {upscale}, Downscale: {downscale}')

                if upscale:
                    # Initialize the mapped highres mask with zeros
                    original_mask = imread(BASE+f'/mask.{dataset}.tif')
                    mapped_mask = np.zeros_like(original_mask)

                    # Map the region in the original mask corresponding to this pixel
                    mapped_mask = upscale_label(unmatched, mapped_mask)

                    # Filter the original mask using the mapped mask
                    filtered_mask = original_mask * mapped_mask
                    save_data(new_mask_path, filtered_mask.astype(np.uint8))

                elif downscale:
                    zoom_factors = [t / s for t, s in zip(TiffInfo(BASE+f'/mask.{dataset}.tif').shape, unmatched.shape)]
                    unmatched = ndimage.zoom(unmatched, zoom_factors, order=0)
                    save_data(new_mask_path, unmatched.astype(np.uint8))

                else:
                    save_data(new_mask_path, unmatched)

    #=======================================================================================
    # predict boundaries
    #=======================================================================================
    if bm.predict_boundaries:
        ngpus = get_gpu_count()
        print('Number of GPUs:', ngpus)

        # batch size
        if bm.patch_size==64:
            batch_size = 24
        elif bm.patch_size==16:
            batch_size = 1024

        for i, dataset in enumerate(bm.datasets):
          if bm.sample==None or i==bm.sample:
            path_to_img = BASE+f'/{dataset}.tif'
            if os.path.exists(f'{path_to_dir}/mask.{dataset}.tif'):
                path_to_mask = f'{path_to_dir}/mask.{dataset}.tif'
            else:
                path_to_mask = BASE+f'/mask.{dataset}.tif'
            print('Mask:', path_to_mask)
            path_to_boundaries = f'{path_to_dir}/final.{dataset}.tif'
            if not os.path.exists(path_to_boundaries):
                print(os.path.basename(path_to_img))
                if bm.model == 'sam':
                    from biomedisa.features.matching.sam_helper import sam_boundaries
                    sam_boundaries(volume_path=path_to_img, boundaries_path=path_to_boundaries,
                        sam_checkpoint=bm.model_path, mask_path=path_to_mask)
                elif bm.model == 'explicit':
                    cmd = [sys.executable, '-m', 'biomedisa.deeplearning',
                        path_to_img, bm.model_path, '-rf', '-im', f'-m={path_to_mask}', f'-ss={bm.patch_size//2}',
                        f'-xp={bm.patch_size}', f'-yp={bm.patch_size}', f'-zp={bm.patch_size}', f'-bs={batch_size}']
                    if ngpus>1:
                        cmd = ['mpirun', '-n', f'{ngpus}'] + cmd
                    subprocess.Popen(cmd).wait()
                    shutil.move(BASE+f'/final.{dataset}.refined.tif', path_to_boundaries)
                elif bm.model == 'implicit':
                    cmd = [sys.executable, '-m', 'biomedisa.deeplearning',
                        path_to_img, bm.model_path, f'-ss={bm.patch_size//4}', '-s', f'-m={path_to_mask}',
                        f'-xp={bm.patch_size}', f'-yp={bm.patch_size}', f'-zp={bm.patch_size}', f'-bs={batch_size}']
                    if ngpus>1:
                        cmd = ['mpirun', '-n', f'{ngpus}'] + cmd
                    subprocess.Popen(cmd).wait()
                    shutil.move(BASE+f'/final.{dataset}.tif', path_to_boundaries)

    #=======================================================================================
    # label particles
    #=======================================================================================
    if bm.label_particles and not bm.labelDatasets:
        for sample_i, dataset in enumerate(bm.datasets):
          if bm.sample==None or bm.sample==sample_i:

            # path to data
            path_to_boundaries = f'{path_to_dir}/final.{dataset}.tif'
            path_to_result = f'{path_to_dir}/result.{dataset}.nrrd'
            if bm.scale_labels:
                path_to_result = path_to_result.replace('.nrrd','_half.nrrd')

            if not os.path.exists(path_to_result):
                print(os.path.basename(path_to_result))

                # load mask
                TIC = time.time()
                if os.path.exists(f'{path_to_dir}/mask.{dataset}.tif'):
                    path_to_mask = f'{path_to_dir}/mask.{dataset}.tif'
                else:
                    path_to_mask = BASE+f'/mask.{dataset}.tif'
                print('Mask:', path_to_mask)
                mask = imread(path_to_mask).astype(np.uint8)

                # remove boundary
                labeled_array = mask.copy()
                edge = imread(path_to_boundaries)
                labeled_array[edge>0]=0
                print(edge.shape)
                del edge
                print('Data loaded:', time.time() - TIC)

                # downsize mask
                if bm.scale_labels:
                    zsh, ysh, xsh = mask.shape
                    zoom_factors = [t / s for t, s in zip((zsh//2,ysh//2,xsh//2), mask.shape)]
                    mask = ndimage.zoom(mask, zoom_factors, order=0)

                # label particles individually
                TIC = time.time()
                s = [[[0,0,0], [0,1,0], [0,0,0]], [[0,1,0], [1,1,1], [0,1,0]], [[0,0,0], [0,1,0], [0,0,0]]]
                labeled_array = labeled_array.astype(np.uint32)
                ndimage.label(labeled_array, structure=s, output=labeled_array)
                print('Label particles:', time.time() - TIC)

                # downsize labels
                if bm.scale_labels:
                    labeled_array = ndimage.zoom(labeled_array, zoom_factors, order=0)
                print(labeled_array.shape)

                # fill segments up to mask
                TIC = time.time()
                nearest_indices = np.zeros(((3,) + labeled_array.shape), dtype=np.uint16)
                #ndimage.distance_transform_edt(labeled_array==0, return_distances=False, return_indices=True, indices=nearest_indices)
                distances = np.zeros(labeled_array.shape, dtype=np.float32)
                distances[labeled_array==0] = np.inf
                nearest_indices = init_indices(nearest_indices) # TODO: use global index and calculate z,y,x on the fly
                nearest_indices = nearest_neighbour_indices(distances, mask, nearest_indices)
                labeled_array = nearest_neighbour(labeled_array, mask, nearest_indices)
                print('Segments refilled:', time.time() - TIC)

                # sort according to size, label in ascending order, remove small particles & save as 16bit if possible
                TIC = time.time()
                lv, ln = unique(labeled_array, return_counts=True)
                t = []
                for v,n in zip(lv, ln):
                    t.append((n,v))
                t = sorted(t, key=lambda x: x[0])[::-1]
                ref = np.zeros(np.amax(lv)+1, dtype=np.int32)
                for i,nv in enumerate(t):
                    if nv[0]>=min_particle_size:
                        ref[int(nv[1])] = i
                labeled_array = change_label_values(labeled_array, ref)
                if np.amax(labeled_array) <= 65535:
                    labeled_array = labeled_array.astype(np.uint16)
                print('Sorting and removing of small particles done:', time.time() - TIC)

                # fill inclusions and pores
                TIC = time.time()
                bounding_boxes = get_bounding_boxes(labeled_array)
                lv, ln = unique(labeled_array, return_counts=True)
                for value in lv[1:]:
                    argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = bounding_boxes[value-1]
                    p1 = np.zeros((argmax_z-argmin_z,argmax_y-argmin_y,argmax_x-argmin_x), dtype=np.uint8)
                    p1[labeled_array[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x]==value]=1
                    if np.any(p1):
                        p1 = fill_fast(p1)
                        labeled_array[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x][p1==1] = value
                print('Inclusions filled:', time.time() - TIC)

                # remove embeddings
                TIC = time.time()
                total_boundary = np.zeros(np.amax(labeled_array)+1, dtype=np.uint32)
                embedded_boundary = np.zeros(np.amax(labeled_array)+1, dtype=np.uint32)
                total_boundary, embedded_boundary = get_boundary_sizes(labeled_array, total_boundary, embedded_boundary)
                #label_vals = np.ones(np.amax(lv)+1, dtype=np.uint32)
                embeddings = []
                for value, total_size in enumerate(total_boundary):
                    if total_size and embedded_boundary[value] > 0.5*total_size:
                        print(value, total_size, embedded_boundary[value] / total_size)
                        #label_vals[value] = 0
                        embeddings.append(value)
                for value in embeddings:
                    argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = bounding_boxes[value-1]
                    p1 = labeled_array[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x]
                    new_labels = is_mostly_inside(p1, value)
                    if np.argmax(new_labels)>0 and np.amax(new_labels) > 0.5*total_size:
                        labeled_array[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x][p1==value]=np.argmax(new_labels)
                #labeled_array = remove_small_particles2(labeled_array, label_vals)
                print('Embeddings removed:', time.time() - TIC)

                # label in ascending order
                labeled_array = label_in_ascending_order(labeled_array)

                # combine with corrected particles from previous step
                old_size, old_path = get_data_size(BASE+f'/{bm.project}/step={bm.step-1}/corr.{dataset}.nrrd')
                if old_size != None:
                    matched,_ = load_data(old_path)
                    m1_max = np.amax(matched)
                    m2_max = np.amax(labeled_array)
                    print(f'Oldsize: {matched.shape}, Newsize: {labeled_array.shape}')
                    if matched.shape != labeled_array.shape:
                        zoom_factors = [t / s for t, s in zip(labeled_array.shape, matched.shape)]
                        matched = ndimage.zoom(matched, zoom_factors, order=0)
                    if int(m1_max) + int(m2_max) > 65535:
                        matched = matched.astype(np.uint32)
                        labeled_array = labeled_array.astype(np.uint32)
                        m1_max = np.uint32(m1_max)
                    labeled_array[labeled_array>0] += m1_max
                    labeled_array[matched>0] = matched[matched>0]
                else:
                    print('No previous result merged.')

                # save results
                save_data(path_to_result, labeled_array)
                print('Saving done.')

                # save labels, sizes and bounding boxes
                bounding_boxes = get_bounding_boxes(labeled_array)
                np.save(f'{path_to_meta}/bounding_boxes{sample_i+1}.npy', bounding_boxes)
                lv, ln = unique(labeled_array, return_counts=True)
                np.save(f'{path_to_meta}/labels{sample_i+1}.npy', lv[1:])
                np.save( f'{path_to_meta}/sizes{sample_i+1}.npy', ln[1:])
                print('Labels and sizes done.')

    #=======================================================================================
    # pre-calculated particles
    #=======================================================================================
    if bm.label_particles and bm.labelDatasets:
        for sample_i, dataset in enumerate(bm.datasets):
          if bm.sample==None or bm.sample==sample_i:

            # path to data
            path_to_boundaries = f'{path_to_dir}/final.{dataset}.tif'
            path_to_result = f'{path_to_dir}/result.{dataset}.nrrd'
            if bm.scale_labels:
                path_to_result = path_to_result.replace('.nrrd','_half.nrrd')

            if not os.path.exists(path_to_result):
                print(os.path.basename(path_to_result))

                # load pre-calculated labels
                labeled_array,_ = load_data(bm.labelDatasets[sample_i])

                # downsize labels
                if bm.scale_labels:
                    zsh, ysh, xsh = TiffInfo(BASE+f'/{dataset}.tif').shape
                    zoom_factors = [t / s for t, s in zip((zsh//2,ysh//2,xsh//2), labeled_array.shape)]
                    labeled_array = ndimage.zoom(labeled_array, zoom_factors, order=0)
                    #tmp = np.zeros_like(labeled_array)
                    #max_z, max_y, max_x = 0, 70, 112
                    #tmp[:, max_y:, max_x:]=labeled_array[:,:-max_y,:-max_x]
                    #labeled_array = tmp.copy()
                print(labeled_array.shape)

                # sort according to size, label in ascending order, remove small particles & save as 16bit if possible
                TIC = time.time()
                lv, ln = unique(labeled_array, return_counts=True)
                t = []
                for v,n in zip(lv, ln):
                    t.append((n,v))
                t = sorted(t, key=lambda x: x[0])[::-1]
                ref = np.zeros(np.amax(lv)+1, dtype=np.int32)
                for i,nv in enumerate(t):
                    if nv[0]>=0:#min_particle_size:
                        ref[int(nv[1])] = i
                labeled_array = change_label_values(labeled_array, ref)
                if np.amax(labeled_array) <= 65535:
                    labeled_array = labeled_array.astype(np.uint16)
                print('Sorting and removing of small particles done:', time.time() - TIC)

                # fill inclusions and pores
                TIC = time.time()
                bounding_boxes = get_bounding_boxes(labeled_array)
                lv, ln = unique(labeled_array, return_counts=True)
                for value in lv[1:]:
                    argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = bounding_boxes[value-1]
                    p1 = np.zeros((argmax_z-argmin_z,argmax_y-argmin_y,argmax_x-argmin_x), dtype=np.uint8)
                    p1[labeled_array[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x]==value]=1
                    if np.any(p1):
                        p1 = fill_fast(p1)
                        labeled_array[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x][p1==1] = value
                print('Inclusions filled:', time.time() - TIC)

                # label in ascending order TODO: merge with sorting above
                labeled_array = label_in_ascending_order(labeled_array)

                # save results
                save_data(path_to_result, labeled_array)
                print('Saving done.')

                # save labels, sizes and bounding boxes
                bounding_boxes = get_bounding_boxes(labeled_array)
                np.save(f'{path_to_meta}/bounding_boxes{sample_i+1}.npy', bounding_boxes)
                lv, ln = unique(labeled_array, return_counts=True)
                np.save(f'{path_to_meta}/labels{sample_i+1}.npy', lv[1:])
                np.save( f'{path_to_meta}/sizes{sample_i+1}.npy', ln[1:])
                print('Labels and sizes done.')

    #=======================================================================================
    # calculate distances to centroid
    #=======================================================================================
    if bm.distances:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

        # iterate over datasets
        for sample_i, dataset in enumerate(bm.datasets):
          if bm.sample==None or bm.sample==sample_i:
            # load particles
            if rank==0:
                convert_to_zarr(f'{path_to_dir}/result.{dataset}.nrrd')
            comm.Barrier()
            zarr_store = zarr.DirectoryStore(f'{path_to_dir}/result.{dataset}.nrrd.zarr')
            particles = zarr.open(zarr_store, mode='r')

            # load label values
            labels = np.load(f'{path_to_meta}/labels{sample_i+1}.npy')

            # load bounding boxes
            bounding_boxes = np.load(f'{path_to_meta}/bounding_boxes{sample_i+1}.npy')

            # allocate distances array
            dists = [0]*(np.amax(labels)+1)

            # calculate distances
            print('Numbers:', len(labels))
            TIC = time.time()
            for i in range(labels.size):
                if i % nprocs == rank:
                    value = labels[i]

                    # extract particle & fill inclusions
                    argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = bounding_boxes[value-1]
                    p1 = np.zeros((argmax_z-argmin_z,argmax_y-argmin_y,argmax_x-argmin_x), dtype=np.uint8)
                    p1[particles[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x]==value]=1
                    #p1 = fill_fast(p1)

                    # distances to centroid
                    centroid = get_centroid(p1)
                    dists[value] = get_distances(p1, centroid)

            # gather results
            comm.Barrier()
            if rank==0:
                for source in range(1, nprocs):
                    for i in range(labels.size):
                        if i % nprocs == source:
                            value = labels[i]
                            dists[value] = comm.recv(source=source, tag=i)
            else:
                for i in range(labels.size):
                    if i % nprocs == rank:
                        value = labels[i]
                        comm.send(dists[value], dest=0, tag=i)

            # save distances
            if rank==0:
                np.save(f'{path_to_meta}/dists{sample_i+1}.npy', np.array(dists, dtype=object))

            # remove zarr files
            comm.Barrier()
            if rank==0:
                shutil.rmtree(f'{path_to_dir}/result.{dataset}.nrrd.zarr')

            # print calculation time
            print('Distances:', int(time.time() - TIC), 'sec')

    #=======================================================================================
    # match particles using distances to centroid
    #=======================================================================================
    if bm.match_particles:
        TIC = time.time()
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

        # check if pre-computed distances exist
        if os.path.exists(f'{path_to_meta}/dists1.npy'):
            dists = True
        else:
            dists = False

        # create zarr files
        if not dists and rank==0:
            for dataset in bm.datasets:
                convert_to_zarr(f'{path_to_dir}/result.{dataset}.nrrd')
        comm.Barrier()

        for i1 in range(1,n_datasets):
            for i2 in range(i1+1,n_datasets+1):
              if bm.sample==None or bm.sample==2*i2-i1:

                # load histograms
                if dists:
                    dists1 = np.load(f'{path_to_meta}/dists{i1}.npy', allow_pickle=True)
                    dists2 = np.load(f'{path_to_meta}/dists{i2}.npy', allow_pickle=True)
                else:
                    # open particles
                    zarr_store = zarr.DirectoryStore(f'{path_to_dir}/result.{bm.datasets[i1-1]}.nrrd.zarr')
                    result1 = zarr.open(zarr_store, mode='r')
                    zarr_store = zarr.DirectoryStore(f'{path_to_dir}/result.{bm.datasets[i2-1]}.nrrd.zarr')
                    result2 = zarr.open(zarr_store, mode='r')

                # load label values
                l1 = np.load(f'{path_to_meta}/labels{i1}.npy')
                l2 = np.load(f'{path_to_meta}/labels{i2}.npy')

                # load label sizes
                n1 = np.load(f'{path_to_meta}/sizes{i1}.npy')
                n2 = np.load(f'{path_to_meta}/sizes{i2}.npy')

                # load bounding boxes
                bb1 = np.load(f'{path_to_meta}/bounding_boxes{i1}.npy')
                bb2 = np.load(f'{path_to_meta}/bounding_boxes{i2}.npy')

                # distance matrix
                #mse = np.zeros((len(l1),len(l2)), dtype=np.float32)
                #mse.fill(np.inf)
                best_mse = -np.ones(len(l1), dtype=np.int32)

                # loop over particles
                for k in range(l1.size):
                  if k % nprocs == rank:
                    size1 = n1[k]
                    if dists:
                        dist1 = dists1[l1[k]]
                    else:
                        # extract particle & fill inclusions
                        value = l1[k]
                        argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = bb1[value-1]
                        p1 = np.zeros((argmax_z-argmin_z,argmax_y-argmin_y,argmax_x-argmin_x), dtype=np.uint8)
                        p1[result1[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x]==value]=1

                        # distances to centroid
                        centroid = get_centroid(p1)
                        dist1 = get_distances(p1, centroid)

                    # reference
                    eps = 1e-6
                    num_bins = int(round(dist1.max() - dist1.min())) // 2
                    r_bins = np.linspace(dist1.min(), dist1.max()+eps, num_bins)
                    r_digitized = np.digitize(dist1, r_bins)
                    bins1, counts1 = np.unique(r_digitized, return_counts=True)
                    nbins1 = np.arange(r_bins.size + 1)
                    ncounts1 = np.zeros_like(nbins1)
                    for i in range(len(bins1)):
                        ncounts1[bins1[i]] = counts1[i]

                    min_error = np.inf
                    for l in range(l2.size):
                        size2 = n2[l]
                        if dists:
                            dist2 = dists2[l2[l]]
                        else:
                            # extract particle & fill inclusions
                            value = l2[l]
                            argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = bb2[value-1]
                            p2 = np.zeros((argmax_z-argmin_z,argmax_y-argmin_y,argmax_x-argmin_x), dtype=np.uint8)
                            p2[result2[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x]==value]=1

                            # zoom to same size
                            zoom_factor = (np.sum(p1) / np.sum(p2))**(1/3)
                            p2 = ndimage.zoom(p2, zoom_factor, order=0)

                            # distances to centroid
                            centroid = get_centroid(p2)
                            dist2 = get_distances(p2, centroid)

                        # select potential particles
                        if size1-0.1*size1 < size2 < size1+0.1*size1:

                            # test sample
                            r_digitized = np.digitize(dist2, r_bins)
                            bins2, counts2 = np.unique(r_digitized, return_counts=True)
                            ncounts2 = np.zeros_like(nbins1)
                            for i in range(len(bins2)):
                                ncounts2[bins2[i]] = counts2[i]

                            # mean squared error
                            error = (ncounts1 - ncounts2)**2
                            #mse[k,l] = np.mean(error)

                            if np.mean(error) < min_error:
                                min_ncounts2 = ncounts2
                                min_error = np.mean(error)
                                best_mse[k] = l

                # gather results
                comm.Barrier()
                if rank==0:
                    for source in range(1, nprocs):
                        for k in range(len(l1)):
                            if k % nprocs == source:
                                #comm.Recv([mse[k], MPI.FLOAT], source=source, tag=k)
                                best_mse[k] = comm.recv(source=source, tag=k)
                else:
                    for k in range(len(l1)):
                        if k % nprocs == rank:
                            #comm.Send([mse[k].copy(), MPI.FLOAT], dest=0, tag=k)
                            comm.send(best_mse[k], dest=0, tag=k)

                # save errors
                if rank==0:
                    np.save(f'{path_to_meta}/mse{i1}{i2}.npy', best_mse)

        # remove zarr files
        comm.Barrier()
        if not dists and rank==0:
          for dataset in bm.datasets:
              shutil.rmtree(f'{path_to_dir}/result.{dataset}.nrrd.zarr')

        # print calculation time
        print(int(time.time() - TIC), 'sec')

    #=======================================================================================
    # find rotation of matching particles
    #=======================================================================================
    if bm.rot_dice:
      TIC = time.time()

      from mpi4py import MPI
      comm = MPI.COMM_WORLD
      rank = comm.Get_rank()
      nprocs = comm.Get_size()
      npgus = get_gpu_count()

      # refine all rotations (for example when using larger label data in second run)
      #refine = True if '--refine' in sys.argv else False
      refine = False

      # create zarr files
      if rank==0:
          for dataset in bm.datasets:
              convert_to_zarr(f'{path_to_dir}/result.{dataset}.nrrd')
      comm.Barrier()

      # datasets
      for i1 in range(1,n_datasets):
        for i2 in range(i1+1,n_datasets+1):
          if bm.sample==None or bm.sample==2*i2-i1:
            # open particles
            zarr_store = zarr.DirectoryStore(f'{path_to_dir}/result.{bm.datasets[i1-1]}.nrrd.zarr')
            result1 = zarr.open(zarr_store, mode='r')
            zarr_store = zarr.DirectoryStore(f'{path_to_dir}/result.{bm.datasets[i2-1]}.nrrd.zarr')
            result2 = zarr.open(zarr_store, mode='r')

            # load label values
            labels1 = np.load(f'{path_to_meta}/labels{i1}.npy')
            labels2 = np.load(f'{path_to_meta}/labels{i2}.npy')

            # load particles sizes
            sizes1 = np.load(f'{path_to_meta}/sizes{i1}.npy')
            sizes2 = np.load(f'{path_to_meta}/sizes{i2}.npy')

            # load boundig boxes
            bounding_boxes1 = np.load(f'{path_to_meta}/bounding_boxes{i1}.npy')
            bounding_boxes2 = np.load(f'{path_to_meta}/bounding_boxes{i2}.npy')

            # load errors
            #mse = np.load(f'{path_to_meta}/mse{i1}{i2}.npy')
            best_mse = np.load(f'{path_to_meta}/mse{i1}{i2}.npy')

            # allocate rotations array
            rotations = np.zeros((np.amax(labels1)+1, 7))

            # load previous rotations
            pre_rotations_path = f'{path_to_meta}/rotations{i1}{i2}.npy'.replace(f'step={bm.step}', f'step={bm.step-1}')
            if os.path.exists(pre_rotations_path):
                pre_rotations = np.load(pre_rotations_path)
                m1_max = np.amax(pre_rotations[:,0])

            # loop over particles
            for arg1 in range(labels1.size):
              if arg1 % nprocs == rank:
                  result_val1 = labels1[arg1]

                  # get argument of result label
                  #arg1 = np.argwhere(labels1==result_val1)[0][0]

                  # get best matching particle based on best squared error
                  #arg2 = np.argmin(mse[arg1])
                  arg2 = best_mse[arg1]
                  result_val2 = labels2[arg2]

                  # copy previous rotation
                  if os.path.exists(pre_rotations_path) and result_val1<=m1_max and pre_rotations[result_val1,2] >= 0.90:
                      rotations[result_val1] = pre_rotations[result_val1]

                  # find rotation match
                  else: #if refine:

                    # no match detected because all volumes were too different
                    #if best_mse[arg1]==0 or np.amin(mse[arg1])==np.inf:
                    if best_mse[arg1]<0 or sizes1[arg1]>2000000000 or sizes2[arg2]>2000000000: #TODO remove
                        rot_dice, best_alpha, best_beta, best_gamma, result_val2 = 0, 0, 0, 0, 0
                        output = np.array([result_val1, result_val2, rot_dice, 0, best_alpha, best_beta, best_gamma])
                        print(rank, f'{arg1+1}/{labels1.size}', result_val1, f'RotDice: {round(rot_dice,4)}', 'NO MATCH')

                    else:
                        # extract particle & fill inclusions
                        argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = bounding_boxes1[result_val1-1]
                        p1 = np.zeros((argmax_z-argmin_z,argmax_y-argmin_y,argmax_x-argmin_x), dtype=np.uint8)
                        p1[result1[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x]==result_val1]=1
                        #p1 = fill_fast(p1)
                        p1_size = np.sum(p1)

                        # scale large particles
                        max_size = 10000 # half of the 63h particles are smaller than 20000
                        if p1_size > max_size and not refine: #TODO optimise threshold for particle size
                            p1_full = p1.copy()
                            zoom_factor = (max_size / p1_size)**(1/3)
                            p1 = ndimage.zoom(p1, zoom_factor, order=0)

                        # initialize rotation
                        alpha = None
                        beta = None
                        gamma = None
                        rot_dice = None

                        # refine previous rotation
                        #if refine and os.path.exists(path_to_result):
                        #    data = np.load(path_to_result)
                        #    rot_dice = data[2]
                        #    if rot_dice > 0.9:
                        #        alpha, beta, gamma = data[4:]

                        # directly use previous result
                        if rot_dice is not None and rot_dice > 0.98 and not refine:
                            print(rank, f'{arg1+1}/{labels1.size}', data[0], data[1], f'RotDice: {round(rot_dice,4)}')
                            output = data

                        # skip refine for bad particles
                        elif refine and rot_dice is not None and rot_dice < 0.9:
                            print(rank, f'{arg1+1}/{labels1.size}', data[0], data[1], f'RotDice: {round(rot_dice,4)}')
                            output = data

                        # calculate best rotation dice
                        else:
                            # fill best matching particle
                            argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = bounding_boxes2[result_val2-1]
                            p2 = np.zeros((argmax_z-argmin_z,argmax_y-argmin_y,argmax_x-argmin_x), dtype=np.uint8)
                            p2[result2[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x]==result_val2]=1
                            #p2 = fill_fast(p2)

                            # scale large particles
                            if p1_size > max_size and not refine:
                                p2_full = p2.copy()
                                p2 = ndimage.zoom(p2, zoom_factor, order=0)
                            #else: # scale particles to the same size
                            #    zoom_factor = (np.sum(p1) / np.sum(p2))**(1/3)
                            #    p2 = ndimage.zoom(p2, zoom_factor, order=0)

                            # calculate best rotation dice
                            rot_dice, best_alpha, best_beta, best_gamma = rotation_dice(p1, p2, alpha, beta, gamma, rank % npgus)

                            # refine sufficiently matched large particles
                            if p1_size > max_size and not refine and rot_dice>0.9:
                                rot_dice, best_alpha, best_beta, best_gamma = rotation_dice(p1_full, p2_full, best_alpha, best_beta, best_gamma, rank % npgus) #TODO increase range

                            # prepare result
                            output = np.array([result_val1, result_val2, rot_dice, p1_size, best_alpha, best_beta, best_gamma])
                            print(rank, f'{arg1+1}/{labels1.size}', result_val1, result_val2, f'RotDice: {round(rot_dice,4)}')

                    # save results
                    rotations[result_val1] = output

            # gather results
            comm.Barrier()
            if rank==0:
                for source in range(1, nprocs):
                    for arg1 in range(labels1.size):
                        if arg1 % nprocs == source:
                            result_val1 = labels1[arg1]
                            rotations[result_val1] = comm.recv(source=source, tag=arg1)
            else:
                for arg1 in range(labels1.size):
                    if arg1 % nprocs == rank:
                        result_val1 = labels1[arg1]
                        comm.send(rotations[result_val1], dest=0, tag=arg1)

            # save rotations
            if rank==0:
                np.save(f'{path_to_meta}/rotations{i1}{i2}.npy', rotations)

      # remove zarr files
      comm.Barrier()
      if rank==0:
          for dataset in bm.datasets:
              shutil.rmtree(f'{path_to_dir}/result.{dataset}.nrrd.zarr')

      # calculation time
      print(int(round(time.time() - TIC)), 'sec')

    #=======================================================================================
    # label matched particles
    #=======================================================================================
    if bm.label_matched_particles:

        labels1 = np.load(f'{path_to_meta}/labels1.npy')
        labels2 = np.load(f'{path_to_meta}/labels2.npy')

        rotations12 = np.load(f'{path_to_meta}/rotations12.npy')
        if n_datasets==3:
            rotations13 = np.load(f'{path_to_meta}/rotations13.npy')
            rotations23 = np.load(f'{path_to_meta}/rotations23.npy')

        threshold = 0.90
        mappings = np.zeros((labels1.size, 5)) #1,12,13,2,23
        matchedAreas1 = np.zeros(labels1.size)
        corrupted_files = 0
        inconsistent_matches = 0
        for i, l in enumerate(labels1):

            # mapping 1->2
            _, result_val12, rot_dice12, p1_size,_,_,_ = rotations12[l]
            if rot_dice12 >= threshold:
                mappings[i,0]=l
                mappings[i,1]=result_val12
                matchedAreas1[i]=p1_size

            # mapping 1->3
            if n_datasets==3:
                _, result_val13, rot_dice13, p1_size,_,_,_ = rotations13[l]
                if rot_dice13 >= threshold:
                    mappings[i,0]=l
                    mappings[i,2]=result_val13
                    matchedAreas1[i]=p1_size

        # check for multiple assignments
        lv,ln = np.unique(mappings[:,1], return_counts=True)
        ln = ln[lv!=0]
        if np.any(ln>1):
            print('Inconsistent assignments on dataset 2')
        lv,ln = np.unique(mappings[:,2], return_counts=True)
        ln = ln[lv!=0]
        if np.any(ln>1):
            print('Inconsistent assignments on dataset 3')

        # mapping 2->3
        if n_datasets==3:
            counter = 0
            n_particles = mappings.shape[0]
            matchedAreas2 = np.zeros(labels2.size)
            for i, l in enumerate(labels2):
                _, result_val23, rot_dice23, p2_size,_,_,_ = rotations23[l]
                if rot_dice23 >= threshold:
                    # additional particles
                    if l not in mappings[:,1] and result_val23 not in mappings[:,2]:
                        mappings = np.append(mappings, np.array([0, 0, 0, l, result_val23]).reshape(1,5), axis=0)
                        counter += 1
                        matchedAreas2[i]=p2_size
                    # add match to existing mapping
                    else:
                        for j in range(n_particles):
                            if mappings[j,1]==l or mappings[j,2]==result_val23:
                                # warn if particles were matched inconsistently and remove all links
                                if mappings[j,1]==l and mappings[j,2]>0 and mappings[j,2]!=result_val23:
                                    print('Inconsistent match 1,3 and 2,3')
                                    print(mappings[j], l, result_val23)
                                    inconsistent_matches += 1
                                    # remove links
                                    mappings[j]=0

                                elif mappings[j,2]==result_val23 and mappings[j,1]>0 and mappings[j,1]!=l:
                                    print('Inconsistent match 1,2 and 2,3')
                                    print(mappings[j], l, result_val23)
                                    inconsistent_matches += 1
                                    # remove links
                                    mappings[j]=0

                                # add match if no inconsistencies detected
                                else:
                                    mappings[j,3]=l
                                    mappings[j,4]=result_val23

            # additional particles
            print('Additional particles 23:', counter)
            print('Additional area:', int(np.sum(matchedAreas2)))

        # monitor results
        print('Corrupted files:', corrupted_files)
        print('Inconsistent matches:', inconsistent_matches)
        print('Matched area:', int(np.sum(matchedAreas1)))

        # delete empty rows
        rows_to_delete = []
        for i,l in enumerate(labels1):
            if np.sum(mappings[i])==0:
                rows_to_delete.append(i)
        mappings = np.delete(mappings, rows_to_delete, axis=0)

        # sort mappings according to number of detections (5,4,3,2)
        m2 = np.zeros_like(mappings)
        i = 0
        for n in range(5,1,-1):
            for k in range(mappings.shape[0]):
                if np.sum(mappings[k]>0)==n:
                    m2[i] = mappings[k]
                    mappings[k]=0
                    i += 1

        # save mappings
        mappings = m2.copy()
        np.save(f'{path_to_meta}/mappings.npy', mappings)

        # label matched particles
        for i, dataset in enumerate(bm.datasets):
            labels = np.load(f'{path_to_meta}/labels{i+1}.npy')
            result,_ = load_data(f'{path_to_dir}/result.{dataset}.nrrd')
            print(os.path.basename(dataset))
            print('Shape:', result.shape)
            counter = 0
            labels_array = np.zeros(np.amax(labels)+1, np.uint64)
            for l in labels: #1,12,13,2,23
                if l in mappings[:,i] or (i==1 and l in mappings[:,3]) or (i==2 and l in mappings[:,4]):
                    labels_array[l]=1
                    counter += 1
            print('Total number of labels:', labels.size)
            print('Matched particles:', counter)
            print('Unmatched labels:', labels.size - counter)
            result = matched_particles(result, labels_array)
            save_data(f'{path_to_dir}/match.{dataset}.nrrd', result)

            # additional matches
            if n_datasets==3 and i==0: # 1->2 (3->1, 3->2)
                counter = 0
                result,_ = load_data(f'{path_to_dir}/result.{dataset}.nrrd')
                labels_array = np.zeros(np.amax(labels)+1, np.uint64)
                for k in range(mappings.shape[0]):
                    if mappings[k,0]>0 and mappings[k,1]>0 and mappings[k,2]==0 and mappings[k,3]==0 and mappings[k,4]==0:
                        labels_array[int(mappings[k,0])]=1
                        counter += 1
                print('Additional particles 12:', counter)
                result = matched_particles(result, labels_array)
                save_data(f'{path_to_dir}/additional12.{dataset}.nrrd', result)

            if n_datasets==3 and i==1: # 2->3 (1->2, 1->3)
                counter = 0
                result,_ = load_data(f'{path_to_dir}/result.{dataset}.nrrd')
                labels_array = np.zeros(np.amax(labels)+1, np.uint64)
                for k in range(mappings.shape[0]):
                    if mappings[k,3]>0 and mappings[k,4]>0 and mappings[k,0]==0 and mappings[k,1]==0 and mappings[k,2]==0:
                        labels_array[int(mappings[k,3])]=1
                        counter += 1
                print('Additional particles 23:', counter)
                result = matched_particles(result, labels_array)
                save_data(f'{path_to_dir}/additional23.{dataset}.nrrd', result)

            if n_datasets==3 and i==2: # 3->1 (2->1, 2->3)
                counter = 0
                result,_ = load_data(f'{path_to_dir}/result.{dataset}.nrrd')
                labels_array = np.zeros(np.amax(labels)+1, np.uint64)
                for k in range(mappings.shape[0]):
                    if mappings[k,0]>0 and mappings[k,2]>0 and mappings[k,1]==0 and mappings[k,3]==0 and mappings[k,4]==0:
                        labels_array[int(mappings[k,2])]=1
                        counter += 1
                print('Additional particles 31:', counter)
                result = matched_particles(result, labels_array)
                save_data(f'{path_to_dir}/additional31.{dataset}.nrrd', result)

    #=======================================================================================
    # correct particles
    #=======================================================================================
    if bm.correct_particles:
        TIC = time.time()
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

        # load matching particles
        dataset1, dataset2, dataset3 = bm.datasets
        if rank==0:
            for dataset in bm.datasets:
                convert_to_zarr(f'{path_to_dir}/match.{dataset}.nrrd')
        comm.Barrier()
        zarr_store = zarr.DirectoryStore(f'{path_to_dir}/match.{dataset1}.nrrd.zarr')
        particles1 = zarr.open(zarr_store, mode='r')
        zarr_store = zarr.DirectoryStore(f'{path_to_dir}/match.{dataset2}.nrrd.zarr')
        particles2 = zarr.open(zarr_store, mode='r')
        zarr_store = zarr.DirectoryStore(f'{path_to_dir}/match.{dataset3}.nrrd.zarr')
        particles3 = zarr.open(zarr_store, mode='r')

        # load masks
        #size = '_half' if '-d=63' in sys.argv else ''
        if rank==0:
            for dataset in bm.datasets:
                convert_to_zarr(BASE+f'/mask.{dataset}.tif')
        comm.Barrier()
        zarr_store = zarr.DirectoryStore(BASE+f'/mask.{dataset1}.tif.zarr')
        mask1 = zarr.open(zarr_store, mode='r')
        zarr_store = zarr.DirectoryStore(BASE+f'/mask.{dataset2}.tif.zarr')
        mask2 = zarr.open(zarr_store, mode='r')
        zarr_store = zarr.DirectoryStore(BASE+f'/mask.{dataset3}.tif.zarr')
        mask3 = zarr.open(zarr_store, mode='r')

        # array shapes
        zsh, ysh, xsh = particles1.shape
        zsh2, ysh2, xsh2 = particles2.shape
        zsh3, ysh3, xsh3 = particles3.shape

        # load mappings
        mappings = np.load(f'{path_to_meta}/mappings.npy')

        # load previous mappings
        m1_max = -1
        previous_maps_path = BASE+f'/{bm.project}/step={bm.step-1}/meta/mappings.npy'
        if os.path.exists(previous_maps_path):
            previous_mappings = np.load(previous_maps_path)
            #m1_max = np.amax(previous_mappings[:,0])
            for k in range(previous_mappings.shape[0]):
                if np.sum(previous_mappings[k]>0)==5:
                    m1_max = k

        # load rotations
        rotations12 = np.load(f'{path_to_meta}/rotations12.npy')
        rotations13 = np.load(f'{path_to_meta}/rotations13.npy')
        rotations23 = np.load(f'{path_to_meta}/rotations23.npy')

        # load boundig boxes (contains buff=10)
        bounding_boxes1 = np.load(f'{path_to_meta}/bounding_boxes1.npy')
        bounding_boxes2 = np.load(f'{path_to_meta}/bounding_boxes2.npy')
        bounding_boxes3 = np.load(f'{path_to_meta}/bounding_boxes3.npy')

        # initialize results
        corr1_path = f'{path_to_dir}/corr.{dataset1}.nrrd'
        corr2_path = f'{path_to_dir}/corr.{dataset2}.nrrd'
        corr3_path = f'{path_to_dir}/corr.{dataset3}.nrrd'
        if rank==0:
            # copy all previously fully matched particles to corr
            if os.path.exists(previous_maps_path):
                labels_array1 = np.zeros(int(np.amax(mappings))+1, np.uint64)
                labels_array2 = np.zeros(int(np.amax(mappings))+1, np.uint64)
                labels_array3 = np.zeros(int(np.amax(mappings))+1, np.uint64)
                for i in range(m1_max+1):
                    r1, r12, r13, r2, r23 = previous_mappings[i]
                    labels_array1[int(r1)]=1
                    labels_array2[int(r12)]=1
                    labels_array3[int(r13)]=1
                match,_ = load_data(f'{path_to_dir}/match.{dataset1}.nrrd')
                corr1 = matched_particles(match, labels_array1)
                save_data(corr1_path + '.zarr', corr1, compress=False)
                del corr1
                match,_ = load_data(f'{path_to_dir}/match.{dataset2}.nrrd')
                corr2 = matched_particles(match, labels_array2)
                save_data(corr2_path + '.zarr', corr2, compress=False)
                del corr2
                match,_ = load_data(f'{path_to_dir}/match.{dataset3}.nrrd')
                corr3 = matched_particles(match, labels_array3)
                save_data(corr3_path + '.zarr', corr3, compress=False)
                del corr3
            else:
                if os.path.exists(corr1_path + '.zarr'):
                    shutil.rmtree(corr1_path + '.zarr')
                if os.path.exists(corr2_path + '.zarr'):
                    shutil.rmtree(corr2_path + '.zarr')
                if os.path.exists(corr3_path + '.zarr'):
                    shutil.rmtree(corr3_path + '.zarr')
                zarr.create(shape=particles1.shape, chunks=(100,100,100), dtype=particles1.dtype,
                    store=zarr.DirectoryStore(corr1_path + '.zarr'), compressor=None)
                zarr.create(shape=particles2.shape, chunks=(100,100,100), dtype=particles2.dtype,
                    store=zarr.DirectoryStore(corr2_path + '.zarr'), compressor=None)
                zarr.create(shape=particles3.shape, chunks=(100,100,100), dtype=particles3.dtype,
                    store=zarr.DirectoryStore(corr3_path + '.zarr'), compressor=None)
        comm.Barrier()
        corr1 = zarr.open(zarr.DirectoryStore(corr1_path + '.zarr'), mode='r+')
        corr2 = zarr.open(zarr.DirectoryStore(corr2_path + '.zarr'), mode='r+')
        corr3 = zarr.open(zarr.DirectoryStore(corr3_path + '.zarr'), mode='r+')

        # iterate over mappings
        steps = mappings.shape[0] + (nprocs - (mappings.shape[0] % nprocs)) % nprocs
        for i in range(rank,steps,nprocs):
            print(f'{i+1}/{mappings.shape[0]}')

            # initialize values and particles
            result_val1, result_val12, result_val13, result_val2, result_val23 = None, None, None, None, None
            p1, p2, p3 = None, None, None
            result_val3 = None

            # get mappings
            if i < mappings.shape[0]:
                result_val1, result_val12, result_val13, result_val2, result_val23 = mappings[i]
                result_val2 = max(result_val12, result_val2)
                result_val3 = max(result_val13, result_val23)

                # skip correction
                if os.path.exists(previous_maps_path):
                    for k in range(previous_mappings.shape[0]):
                        if np.sum(previous_mappings[k]>0)==5 and np.all(previous_mappings[k]==mappings[i]):
                            result_val1, result_val12, result_val13, result_val2, result_val23 = None, None, None, None, None
                            result_val3 = None

            if result_val1:
                # extract particle1
                argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = bounding_boxes1[int(result_val1)-1]
                argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = max(argmin_z-15,0), min(argmax_z+15,zsh), max(argmin_y-15,0), min(argmax_y+15,ysh), max(argmin_x-15,0), min(argmax_x+15,xsh)
                p1 = np.zeros((argmax_z-argmin_z,argmax_y-argmin_y,argmax_x-argmin_x), dtype=np.int8)
                m1 = np.zeros_like(p1)
                l1 = np.zeros_like(p1)
                p1[particles1[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x]==result_val1]=1
                m1[mask1[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x]>0]=1
                l1[corr1[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x]>0]=1

            if result_val2:
                # extract particle2
                argmin_z2, argmax_z2, argmin_y2, argmax_y2, argmin_x2, argmax_x2 = bounding_boxes2[int(result_val2)-1]
                argmin_z2, argmax_z2, argmin_y2, argmax_y2, argmin_x2, argmax_x2 = max(argmin_z2-15,0), min(argmax_z2+15,zsh2), max(argmin_y2-15,0), min(argmax_y2+15,ysh2), max(argmin_x2-15,0), min(argmax_x2+15,xsh2)
                p2 = np.zeros((argmax_z2-argmin_z2,argmax_y2-argmin_y2,argmax_x2-argmin_x2), dtype=np.int8)
                m2 = np.zeros_like(p2)
                l2 = np.zeros_like(p2)
                p2[particles2[argmin_z2:argmax_z2, argmin_y2:argmax_y2, argmin_x2:argmax_x2]==result_val2]=1
                m2[mask2[argmin_z2:argmax_z2, argmin_y2:argmax_y2, argmin_x2:argmax_x2]>0]=1
                l2[corr2[argmin_z2:argmax_z2, argmin_y2:argmax_y2, argmin_x2:argmax_x2]>0]=1

            if result_val3:
                # extract particle3
                argmin_z3, argmax_z3, argmin_y3, argmax_y3, argmin_x3, argmax_x3 = bounding_boxes3[int(result_val3)-1]
                argmin_z3, argmax_z3, argmin_y3, argmax_y3, argmin_x3, argmax_x3 = max(argmin_z3-15,0), min(argmax_z3+15,zsh3), max(argmin_y3-15,0), min(argmax_y3+15,ysh3), max(argmin_x3-15,0), min(argmax_x3+15,xsh3)
                p3 = np.zeros((argmax_z3-argmin_z3,argmax_y3-argmin_y3,argmax_x3-argmin_x3), dtype=np.int8)
                m3 = np.zeros_like(p3)
                l3 = np.zeros_like(p3)
                p3[particles3[argmin_z3:argmax_z3, argmin_y3:argmax_y3, argmin_x3:argmax_x3]==result_val3]=1
                m3[mask3[argmin_z3:argmax_z3, argmin_y3:argmax_y3, argmin_x3:argmax_x3]>0]=1
                l3[corr3[argmin_z3:argmax_z3, argmin_y3:argmax_y3, argmin_x3:argmax_x3]>0]=1

            # majority voting
            if result_val1 and result_val12 and result_val13:
                alpha12, beta12, gamma12 = rotations12[int(result_val1),-3:]
                alpha13, beta13, gamma13 = rotations13[int(result_val1),-3:]
                p12 = correct_match_majority(p1, p2, m1, alpha12, beta12, gamma12, rank=rank)
                p13 = correct_match_majority(p1, p3, m1, alpha13, beta13, gamma13, rank=rank)
                p1 += p12 + p13
                p1[p1<2]=0
                p1[p1>0]=1
                p1[l1>0]=0
                if np.any(p1):
                    p1 = clean_fast(p1)

            if result_val1 and result_val12 and result_val23:
                alpha12, beta12, gamma12 = rotations12[int(result_val1),-3:]
                alpha23, beta23, gamma23 = rotations23[int(result_val2),-3:]
                if np.any(p1):
                    p21 = correct_match_majority(p2, p1, m2, alpha12, beta12, gamma12, rank=rank, inverse=True)
                else:
                    p21 = np.zeros_like(p2)
                p23 = correct_match_majority(p2, p3, m2, alpha23, beta23, gamma23, rank=rank)
                p2 += p21 + p23
                p2[p2<2]=0
                p2[p2>0]=1
                p2[l2>0]=0
                if np.any(p2):
                    p2 = clean_fast(p2)

            if result_val1 and result_val13 and result_val23:
                alpha13, beta13, gamma13 = rotations13[int(result_val1),-3:]
                alpha23, beta23, gamma23 = rotations23[int(result_val2),-3:]
                if np.any(p1):
                    p31 = correct_match_majority(p3, p1, m3, alpha13, beta13, gamma13, rank=rank, inverse=True)
                else:
                    p31 = np.zeros_like(p3)
                if np.any(p2):
                    p32 = correct_match_majority(p3, p2, m3, alpha23, beta23, gamma23, rank=rank, inverse=True)
                else:
                    p32 = np.zeros_like(p3)
                p3 += p31 + p32
                p3[p3<2]=0
                p3[p3>0]=1
                p3[l3>0]=0
                if np.any(p3):
                    p3 = clean_fast(p3)

            # correction if N=2 (N=2 correction < majority voting)
            if result_val1 and result_val12 and not (result_val13 and result_val23):
                alpha12, beta12, gamma12 = rotations12[int(result_val1),-3:]
                if np.any(p1) and np.any(p2):
                    p1t, p2t = correct_match(p1, p2, m1, m2, l1, l2, alpha12, beta12, gamma12, rank=rank)
                    if not result_val13 and np.any(p1t):
                        p1 = clean_fast(p1t)
                        p1[l1>0]=0
                    if not result_val23 and np.any(p2t):
                        p2 = clean_fast(p2t)
                        p2[l2>0]=0

            if result_val1 and result_val13 and not (result_val12 and result_val23):
                alpha13, beta13, gamma13 = rotations13[int(result_val1),-3:]
                if np.any(p1) and np.any(p3):
                    p1t, p3t = correct_match(p1, p3, m1, m3, l1, l3, alpha13, beta13, gamma13, rank=rank)
                    if not result_val12 and np.any(p1t):
                        p1 = clean_fast(p1t)
                        p1[l1>0]=0
                    if not result_val23 and np.any(p3t):
                        p3 = clean_fast(p3t)
                        p3[l3>0]=0

            if result_val23 and not (result_val12 and result_val13):
                alpha23, beta23, gamma23 = rotations23[int(result_val2),-3:]
                if np.any(p2) and np.any(p3):
                    p2t, p3t = correct_match(p2, p3, m2, m3, l2, l3, alpha23, beta23, gamma23, rank=rank)
                    if not result_val12 and np.any(p2t):
                        p2 = clean_fast(p2t)
                        p2[l2>0]=0
                    if not result_val13 and np.any(p3t):
                        p3 = clean_fast(p3t)
                        p3[l3>0]=0

            # wait until the reading process is complete
            comm.Barrier()

            # save corrections
            if rank==0:
                if np.any(p1):
                    corr1[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x] = (1-p1)*corr1[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x] + (p1*result_val1).astype(corr1.dtype)
                if np.any(p2):
                    corr2[argmin_z2:argmax_z2, argmin_y2:argmax_y2, argmin_x2:argmax_x2] = (1-p2)*corr2[argmin_z2:argmax_z2, argmin_y2:argmax_y2, argmin_x2:argmax_x2] + (p2*result_val2).astype(corr2.dtype)
                if np.any(p3):
                    corr3[argmin_z3:argmax_z3, argmin_y3:argmax_y3, argmin_x3:argmax_x3] = (1-p3)*corr3[argmin_z3:argmax_z3, argmin_y3:argmax_y3, argmin_x3:argmax_x3] + (p3*result_val3).astype(corr3.dtype)

            # communicate p1
            if rank==0:
                for source in range(1, nprocs):
                    data = np.empty(7, dtype=np.int32)
                    comm.Recv([data, MPI.INT], source=source, tag=0)
                    min_z, max_z, min_y, max_y, min_x, max_x, val = data[:]
                    if np.any(data):
                        data = np.empty((max_z-min_z, max_y-min_y, max_x-min_x), dtype=np.int8)
                        for k in range(data.shape[0]):
                            comm.Recv([data[k], MPI.SIGNED_CHAR], source=source, tag=k+1)
                        corr1[min_z:max_z, min_y:max_y, min_x:max_x] = (1-data)*corr1[min_z:max_z, min_y:max_y, min_x:max_x] + (data*val).astype(corr1.dtype)
            else:
                if np.any(p1):
                    data = np.array([argmin_z,argmax_z,argmin_y,argmax_y,argmin_x,argmax_x,result_val1], dtype=np.int32)
                else:
                    data = np.zeros(7, dtype=np.int32)
                comm.Send([data, MPI.INT], dest=0, tag=0)
                if np.any(data):
                    data = p1.copy().astype(np.int8)
                    for k in range(data.shape[0]):
                        comm.Send([data[k], MPI.SIGNED_CHAR], dest=0, tag=k+1)

            # communicate p2
            if rank==0:
                for source in range(1, nprocs):
                    data = np.empty(7, dtype=np.int32)
                    comm.Recv([data, MPI.INT], source=source, tag=0)
                    min_z, max_z, min_y, max_y, min_x, max_x, val = data[:]
                    if np.any(data):
                        data = np.empty((max_z-min_z, max_y-min_y, max_x-min_x), dtype=np.int8)
                        for k in range(data.shape[0]):
                            comm.Recv([data[k], MPI.SIGNED_CHAR], source=source, tag=k+1)
                        corr2[min_z:max_z, min_y:max_y, min_x:max_x] = (1-data)*corr2[min_z:max_z, min_y:max_y, min_x:max_x] + (data*val).astype(corr2.dtype)
            else:
                if np.any(p2):
                    data = np.array([argmin_z2,argmax_z2,argmin_y2,argmax_y2,argmin_x2,argmax_x2,result_val2], dtype=np.int32)
                else:
                    data = np.zeros(7, dtype=np.int32)
                comm.Send([data, MPI.INT], dest=0, tag=0)
                if np.any(data):
                    data = p2.copy().astype(np.int8)
                    for k in range(data.shape[0]):
                        comm.Send([data[k], MPI.SIGNED_CHAR], dest=0, tag=k+1)

            # communicate p3
            if rank==0:
                for source in range(1, nprocs):
                    data = np.empty(7, dtype=np.int32)
                    comm.Recv([data, MPI.INT], source=source, tag=0)
                    min_z, max_z, min_y, max_y, min_x, max_x, val = data[:]
                    if np.any(data):
                        data = np.empty((max_z-min_z, max_y-min_y, max_x-min_x), dtype=np.int8)
                        for k in range(data.shape[0]):
                            comm.Recv([data[k], MPI.SIGNED_CHAR], source=source, tag=k+1)
                        corr3[min_z:max_z, min_y:max_y, min_x:max_x] = (1-data)*corr3[min_z:max_z, min_y:max_y, min_x:max_x] + (data*val).astype(corr3.dtype)
            else:
                if np.any(p3):
                    data = np.array([argmin_z3,argmax_z3,argmin_y3,argmax_y3,argmin_x3,argmax_x3,result_val3], dtype=np.int32)
                else:
                    data = np.zeros(7, dtype=np.int32)
                comm.Send([data, MPI.INT], dest=0, tag=0)
                if np.any(data):
                    data = p3.copy().astype(np.int8)
                    for k in range(data.shape[0]):
                        comm.Send([data[k], MPI.SIGNED_CHAR], dest=0, tag=k+1)

            # wait until the writing process is complete
            comm.Barrier()

        if rank==0:
            # save results
            save_data(corr1_path, corr1)
            save_data(corr2_path, corr2)
            save_data(corr3_path, corr3)

            # remove zarr files
            for dataset in bm.datasets:
                shutil.rmtree(f'{path_to_dir}/match.{dataset}.nrrd.zarr')
                shutil.rmtree(BASE+f'/mask.{dataset}.tif.zarr')
            shutil.rmtree(corr1_path + '.zarr')
            shutil.rmtree(corr2_path + '.zarr')
            shutil.rmtree(corr3_path + '.zarr')

            # calculation time
            print(int(round(time.time() - TIC)), 'sec')

    #=======================================================================================
    # matched area
    #=======================================================================================
    if bm.matched_area:
        # dataset (1: 1,2,23; 250: 2,3,31; 63: 1,2,23)
        dataset1, dataset2, dataset3 = bm.datasets
        dataset_a = dataset1
        dataset_b = dataset2
        additional = '23'

        # unmatched area
        matched,_ = load_data(f'{path_to_dir}/corr.{dataset_a}.nrrd')
        #lv, ln = unique(matched, return_counts=True)
        #print(ln[-100:])
        if '-d=63h' in sys.argv or '-d=63' in sys.argv:
            mask,_=load_data(BASE+'/Quartz/63microns/clean_mask.Quartz_sphere_15_25_110mins_2_greaterthan63microns_uint16_half.nrrd')
        else:
            mask = imread(BASE+f'/mask.{dataset_a}.tif')
        unmatched = (mask > matched).astype(np.uint8)
        #matched_area = np.sum((matched * mask)>0)

        # total area
        total_area = np.sum(mask>0)

        # remove outliers
        s = [[[0,0,0], [0,1,0], [0,0,0]], [[0,1,0], [1,1,1], [0,1,0]], [[0,0,0], [0,1,0], [0,0,0]]]
        unmatched = unmatched.astype(np.uint32)
        ndimage.label(unmatched, structure=s, output=unmatched)
        lv, ln = unique(unmatched, return_counts=True)
        label_vals = np.zeros(np.amax(lv)+1, dtype=np.uint32)
        for i,l in enumerate(lv):
            if ln[i]>=100:
                label_vals[l] = 1
        unmatched = remove_small_particles2(unmatched, label_vals)
        unmatched[unmatched>0]=1
        unmatched = unmatched.astype(np.uint8)
        unmatched_area = np.sum(unmatched)

        # extract corrected additional particles
        a2,_=load_data(f'{path_to_dir}/additional{additional}.{dataset_b}.nrrd')
        c2,_=load_data(f'{path_to_dir}/corr.{dataset_b}.nrrd')
        m2,_=load_data(BASE+f'/mask.{dataset_b}.tif')
        labels = unique(a2)
        labels_array = np.zeros(np.amax(c2)+1, np.uint32)
        for l in labels[1:]:
            labels_array[l]=1
        c2 = matched_particles(c2, labels_array)
        c2 = c2 * m2

        # print matched area in percentage
        print('Total area:', total_area)
        matched_area_1 = 100-100/total_area*unmatched_area
        print('Matched Area 1:', matched_area_1)
        print('Additional voxels:', np.sum(c2>0))
        unmatched_area -= np.sum(c2>0)
        #matched_area += np.sum(c2>0)
        total_matched_vol = 100-100/total_area*unmatched_area
        print('Total Matched Volume:', total_matched_vol)
        print('Additional volume:', total_matched_vol - matched_area_1)
        #print('Matched Area:', 100/total_area*matched_area)
        #print('Missing particles:', int(round((total_area - matched_area) / np.mean(ln[1:]))))

    #=======================================================================================
    # train
    #=======================================================================================
    if bm.train:
        from biomedisa.deeplearning import deep_learning
        epochs=15

        # hyper parameters
        if bm.patch_size==64:
            batch_size = 24
        elif bm.patch_size==16:
            batch_size = 48

        # load image data
        images = imread(BASE+'/training_img.tif')
        val_img = imread(BASE+'/validation_img.tif')

        if bm.model == 'explicit':
            # load labels (explicit)
            labels = np.load(BASE+'/training_labels_explicit.npy')
            val_label = np.load(BASE+'/validation_labels_explicit.npy')

            # model path
            path_to_model = BASE+'/model_explicit.h5'
            pretrained_model = None

            # train explicit boundary prediction
            deep_learning(images, labels, train=True, patch_normalization=True,
                path_to_model=path_to_model, scaling=False, val_dice=False,
                pretrained_model=pretrained_model,
                flip_x=True, flip_y=True, flip_z=True, swapaxes=True,
                val_img_data=val_img, val_label_data=val_label, epochs=epochs,
                x_patch=bm.patch_size, y_patch=bm.patch_size, z_patch=bm.patch_size,
                batch_size=batch_size, stride_size=bm.stride_size,
                validation_stride_size=bm.patch_size, ignore_mask=True)

        if bm.model == 'implicit':
            # load labels (implicit)
            labels,_ = load_data(BASE+'/training_labels_implicit.nrrd')
            val_label,_ = load_data(BASE+'/validation_labels_implicit.nrrd')

            # model path
            path_to_model = BASE+'/model_implicit.h5'
            pretrained_model = None

            # train network
            deep_learning(images, labels, train=True, patch_normalization=True,
                path_to_model=path_to_model, scaling=False, val_dice=False,
                pretrained_model=pretrained_model,
                flip_x=True, flip_y=True, flip_z=True, swapaxes=True,
                val_img_data=val_img, val_label_data=val_label, epochs=epochs,
                x_patch=bm.patch_size, y_patch=bm.patch_size, z_patch=bm.patch_size, batch_size=batch_size,
                stride_size=bm.stride_size, validation_stride_size=bm.patch_size, separation=True)

