#!/usr/bin/python3
##########################################################################
##                                                                      ##
##  Copyright (c) 2024 Philipp Lösel. All rights reserved.              ##
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

import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import numpy as np
from biomedisa_features.biomedisa_helper import load_data
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from stl import mesh
import vtk
import re
import argparse

def marching_cubes(image, threshold, poly_reduction, smoothing_iterations):

    # marching cubes
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(image)
    mc.ComputeNormalsOn()
    mc.ComputeGradientsOn()
    mc.SetValue(0, threshold)
    mc.Update()

    # To remain largest region
    #confilter = vtk.vtkPolyDataConnectivityFilter()
    #confilter.SetInputData(mc.GetOutput())
    #confilter.SetExtractionModeToLargestRegion()
    #confilter.Update()

    # get volume and surface area
    #m = vtk.vtkMassProperties()
    #m.SetInputData(mc.GetOutput())
    #print("Mass properties\n"
    #      "-----------------\n"
    #      "Volume: " + str(round(m.GetVolume())) + " \n"
    #      "Surface Area: " + str(round(m.GetSurfaceArea())) + " \n")
    #m.Update()

    # reduce poly data
    inputPoly = vtk.vtkPolyData()
    inputPoly.ShallowCopy(mc.GetOutput())

    #print("Before decimation\n"
    #      "-----------------\n"
    #      "There are " + str(inputPoly.GetNumberOfPoints()) + " points.\n"
    #      "There are " + str(inputPoly.GetNumberOfPolys()) + " polygons.\n")

    #decimate = vtk.vtkDecimatePro()
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(inputPoly)
    decimate.SetTargetReduction(poly_reduction)
    decimate.Update()

    decimatedPoly = vtk.vtkPolyData()
    decimatedPoly.ShallowCopy(decimate.GetOutput())

    #print("After decimation \n"
    #      "-----------------\n"
    #      "There are " + str(decimatedPoly.GetNumberOfPoints()) + " points.\n"
    #      "There are " + str(decimatedPoly.GetNumberOfPolys()) + " polygons.\n")

    # smooth surface
    smoothFilter = vtk.vtkSmoothPolyDataFilter()
    smoothFilter.SetInputData(decimatedPoly)
    smoothFilter.SetNumberOfIterations(smoothing_iterations)
    smoothFilter.SetRelaxationFactor(0.1)
    smoothFilter.FeatureEdgeSmoothingOff()
    smoothFilter.BoundarySmoothingOn()
    smoothFilter.Update()

    decimatedPoly = vtk.vtkPolyData()
    decimatedPoly.ShallowCopy(smoothFilter.GetOutput())

    return decimatedPoly#confilter.GetOutput()

def save_mesh(path_to_data, image, x_res=1, y_res=1, z_res=1, poly_reduction=0.9, smoothing_iterations=15):

    # get labels
    zsh, ysh, xsh = image.shape
    allLabels = np.unique(image)
    b = np.empty_like(image)
    arr = np.empty((0,3,3))
    nTotalCells = [0]

    for label in allLabels[1:]:

        # get label
        b.fill(0)
        b[image==label] = 1

        # numpy to vtk
        sc = numpy_to_vtk(num_array=b.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        imageData = vtk.vtkImageData()
        imageData.SetOrigin(0, 0, 0)
        imageData.SetSpacing(x_res, y_res, z_res)
        #imageData.SetDimensions(zsh, ysh, xsh)
        imageData.SetExtent(0,xsh-1,0,ysh-1,0,zsh-1)
        imageData.GetPointData().SetScalars(sc)

        # get poly data
        poly = marching_cubes(imageData, 1, poly_reduction, smoothing_iterations)

        # get number of cells
        nPoints = poly.GetNumberOfPoints()
        nPolys = poly.GetNumberOfPolys()
        if nPoints and nPolys:

            # get points
            points = poly.GetPoints()
            array = points.GetData()
            numpy_points = vtk_to_numpy(array)

            # get cells
            cells = poly.GetPolys()
            array = cells.GetData()
            numpy_cells = vtk_to_numpy(array)
            numpy_cells = numpy_cells.reshape(-1,4)

            # create mesh
            nCells, nCols = numpy_cells.shape
            tmp = np.empty((nCells,3,3))
            for k in range(nCells):
                for l in range(1,4):
                    id = numpy_cells[k,l]
                    tmp[k,l-1] = numpy_points[id]   # x,y,z

            # append output
            arr = np.append(arr, tmp, axis=0)
            nTotalCells.append(nTotalCells[-1]+nCells)

    # save data as stl mesh
    data = np.zeros(nTotalCells[-1], dtype=mesh.Mesh.dtype)
    data['vectors'] = arr
    mesh_final = mesh.Mesh(data.copy())
    for i in range(len(nTotalCells)-1):
        start = nTotalCells[i]
        stop = nTotalCells[i+1]
        mesh_final.attr[start:stop,0] = i+1
    mesh_final.save(path_to_data)

def get_voxel_spacing(header, data, extension):

    if extension == '.am':
        # read header as string
        b = header[0].tobytes()
        try:
            s = b.decode("utf-8")
        except:
            s = b.decode("latin1")

        # get physical size from image header
        bounding_box = re.search('BoundingBox (.*),\n', s)
        if bounding_box:
            bounding_box = bounding_box.group(1)
            i0, i1, i2, i3, i4, i5 = bounding_box.split(' ')

            # voxel spacing
            zsh, ysh, xsh = data.shape
            xres = (float(i1)-float(i0)) / xsh
            yres = (float(i3)-float(i2)) / ysh
            zres = (float(i5)-float(i4)) / zsh
        else:
            xres, yres, zres = 1, 1, 1

    elif extension in ['.hdr', '.mhd', '.mha', '.nrrd', '.nii', '.nii.gz']:
        xres, yres, zres = header.get_voxel_spacing()
    elif extension == '.zip':
        header = header[0][0]
        try:
            xres, yres, zres = header.get_voxel_spacing()
        except:
            xres, yres = header.get_voxel_spacing()
            zres = 1.0
    else:
        print('Warning: could not get voxel spacing. Using x_spacing, y_spacing, z_spacing = 1, 1, 1 instead.')
        xres, yres, zres = 1, 1, 1

    return xres, yres, zres

if __name__ == "__main__":

    # initialize arguments
    parser = argparse.ArgumentParser(description='Biomedisa mesh generator.',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required arguments
    parser.add_argument('path_to_data', type=str, metavar='PATH_TO_LABELS',
                        help='Location of label data')

    # optional arguments
    parser.add_argument('-pr', '--poly_reduction', type=float, default=0.9,
                        help='Reduce number of polygons by this factor')
    parser.add_argument('-s', '--smoothing_iterations', type=int, default=15,
                        help='Iteration steps for smoothing')
    parser.add_argument('-xres','--x_res', type=int, default=None,
                        help='Voxel spacing/resolution x-axis')
    parser.add_argument('-yres','--y_res', type=int, default=None,
                        help='Voxel spacing/resolution y-axis')
    parser.add_argument('-zres','--z_res', type=int, default=None,
                        help='Voxel spacing/resolution z-axis')
    args = parser.parse_args()

    # load data
    data, header, extension = load_data(args.path_to_data, return_extension=True)

    # get voxel spacing
    if not all([args.x_res, args.y_res, args.z_res]):
        x_res, y_res, z_res = get_voxel_spacing(header, data, extension)
        if args.x_res:
            x_res = args.x_res
        if args.y_res:
            y_res = args.y_res
        if args.z_res:
            z_res = args.z_res
        print(f'Voxel spacing: x_spacing, y_spacing, z_spacing = {x_res}, {y_res}, {z_res}')
    else:
        x_res = args.x_res
        y_res = args.y_res
        z_res = args.z_res

    # save stl file
    path_to_data = args.path_to_data.replace(os.path.splitext(args.path_to_data)[1],'.stl')
    save_mesh(path_to_data, data, x_res, y_res, z_res, args.poly_reduction, args.smoothing_iterations)

