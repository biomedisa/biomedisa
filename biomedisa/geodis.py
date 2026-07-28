#!/usr/bin/python3

import os
import numpy as np
import numba
import argparse
from biomedisa.features.biomedisa_helper import unique

sqrt2 = np.sqrt(2)
sqrt3 = np.sqrt(3)

offsets = (
    (-1,-1,-1,sqrt3),
    (-1,-1, 0,sqrt2),
    (-1,-1, 1,sqrt3),
    (-1, 0,-1,sqrt2),
    (-1, 0, 0,1.0),
    (-1, 0, 1,sqrt2),
    (-1, 1,-1,sqrt3),
    (-1, 1, 0,sqrt2),
    (-1, 1, 1,sqrt3),
    ( 0,-1,-1,sqrt2),
    ( 0,-1, 0,1.0),
    ( 0,-1, 1,sqrt2),
    ( 0, 0,-1,1.0),
)

backward_offsets = (
    ( 1, 1, 1, sqrt3),
    ( 1, 1, 0, sqrt2),
    ( 1, 1,-1, sqrt3),
    ( 1, 0, 1, sqrt2),
    ( 1, 0, 0, 1.0),
    ( 1, 0,-1, sqrt2),
    ( 1,-1, 1, sqrt3),
    ( 1,-1, 0, sqrt2),
    ( 1,-1,-1, sqrt3),
    ( 0, 1, 1, sqrt2),
    ( 0, 1, 0, 1.0),
    ( 0, 1,-1, sqrt2),
    ( 0, 0, 1, 1.0),
)

@numba.jit(nopython=True)
def geodesic_raster_scan(img, c, lamb=1.0, iterations=4):
    zsh, ysh, xsh = c.shape

    for i in range(iterations):

        for z in range(1,zsh):
            for y in range(1,ysh-1):
                for x in range(1,xsh-1):
                    best = c[z,y,x]
                    for dz,dy,dx,d in offsets:
                        nz = z + dz
                        ny = y + dy
                        nx = x + dx
                        cost = c[nz,ny,nx] + (1.0-lamb) * d + lamb * (img[z,y,x]-img[nz,ny,nx])**2
                        if cost < best:
                            best = cost
                    c[z,y,x] = best

        for z in range(zsh-2,-1,-1):
            for y in range(ysh-2,0,-1):
                for x in range(xsh-2,0,-1):
                    best = c[z,y,x]
                    for dz,dy,dx,d in backward_offsets:
                        nz = z + dz
                        ny = y + dy
                        nx = x + dx
                        cost = c[nz,ny,nx] + (1.0-lamb) * d + lamb * (img[z,y,x]-img[nz,ny,nx])**2
                        if cost < best:
                            best = cost
                    c[z,y,x] = best

    return c


def geodesic_segment(image, labels, lamb=1.0, iterations=4):
    '''
    lamb: weighting betwween 0.0 and 1.0
          if lamb==0.0, return spatial euclidean distance without considering gradient
          if lamb==1.0, the distance is based on gradient only without using spatial distance
    iter: number of iteration for raster scanning.
    '''
    image = image.astype(np.float32)
    image -= np.amin(image)
    image /= np.amax(image)
    allLabels = unique(labels)
    index = np.argwhere(allLabels<0)
    allLabels = np.delete(allLabels, index)
    result = np.zeros(labels.shape, np.uint8)
    mask = np.empty(labels.shape, np.float32)
    for k, label in enumerate(allLabels):
        mask.fill(np.inf)
        mask[labels==label] = 0
        distance = geodesic_raster_scan(image, mask, 1.0, 4)
        if k == 0:
            min_distance = distance.copy()
        else:
            tmp = distance < min_distance
            result[tmp] = label
            min_distance[tmp] = distance[tmp]
    return result


