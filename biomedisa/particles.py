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

import sys
import numpy as np
from biomedisa.features.biomedisa_helper import *
from biomedisa.matching import *
from tifffile import imread
from scipy import ndimage
import time

if __name__ == "__main__":

    # initialize arguments
    parser = argparse.ArgumentParser(description='Particle segmentation.',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required arguments
    parser.add_argument('img_path', type=str, metavar='PATH_TO_IMAGE',
                        help='Location of image data (must be Multipage-TIFF)')
    parser.add_argument('mask_path', type=str, metavar='PATH_TO_MASK',
                        help='Location of mask (must be Multipage-TIFF)')

    # optional arguments
    parser.add_argument('-v', '--version', action='version', version=f'{biomedisa.__version__}',
                        help='Biomedisa version')
    parser.add_argument('-mp','--model_path', type=str, metavar='PATH', default=None,
                        help='Location of trained model for prediction')
    parser.add_argument('-m','--model', type=str, default='implicit',
                        help='Model used (sam, implicit)')
    parser.add_argument('-mps','--min_particle_size', type=int, default=1000,
                        help='Objects smaller than this value will be removed')
    bm = parser.parse_args()

    # image and mask data must be Multipage-TIFF
    if os.path.splitext(bm.img_path)[1] not in ['.tif','.tiff','.TIF','.TIFF']:
        raise RuntimeError("Unsupported format. Image file must be a Multipage-TIFF (.tif) file.")
    if os.path.splitext(bm.mask_path)[1] not in ['.tif','.tiff','.TIF','.TIFF']:
        raise RuntimeError("Unsupported format. Mask file must be a Multipage-TIFF (.tif) file.")

    #=======================================================================================
    # predict boundaries
    #=======================================================================================
    ngpus = get_gpu_count()
    print('Number of GPUs:', ngpus)

    # boundaries path
    boundaries_path = os.path.dirname(bm.img_path) + '/final.' + os.path.basename(bm.img_path)

    if bm.model == 'sam':
        from biomedisa.features.matching.sam_helper import sam_boundaries
        sam_boundaries(volume_path=bm.img_path, boundaries_path=boundaries_path,
            sam_checkpoint=bm.model_path, mask_path=bm.mask_path)
    else:
        cmd = [sys.executable, '-m', 'biomedisa.deeplearning',
            bm.img_path, bm.model_path, '-ss=4', '-s', f'-m={bm.mask_path}',
            '-xp=16', '-yp=16', '-zp=16', '-bs=1024']
        if ngpus>1:
            cmd = ['mpirun', '-n', f'{ngpus}'] + cmd
        subprocess.Popen(cmd).wait()

    #=======================================================================================
    # label particles
    #=======================================================================================
    # path to result
    path_to_result = boundaries_path.replace('.tif','.nrrd').replace('/final.','/result.')

    # load mask
    TIC = time.time()
    print('Mask:', bm.mask_path)
    mask,_ = load_data(bm.mask_path)
    mask = mask.astype(np.uint8)

    # remove boundary
    labeled_array = mask.copy()
    boundaries = imread(boundaries_path)
    labeled_array[boundaries>0]=0
    print(boundaries.shape)
    del boundaries
    print('Data loaded:', time.time() - TIC)

    # label particles individually
    TIC = time.time()
    s = [[[0,0,0], [0,1,0], [0,0,0]], [[0,1,0], [1,1,1], [0,1,0]], [[0,0,0], [0,1,0], [0,0,0]]]
    labeled_array = labeled_array.astype(np.uint32)
    ndimage.label(labeled_array, structure=s, output=labeled_array)
    print('Label particles:', time.time() - TIC)

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
        if nv[0]>=bm.min_particle_size:
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

    # save results
    save_data(path_to_result, labeled_array)
    print('Saving done.')

