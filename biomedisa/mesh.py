#!/usr/bin/python3
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

import os
import numpy as np
import biomedisa
from biomedisa.features.biomedisa_helper import load_data, unique_file_path
from biomedisa.features.django_env import create_pid_object
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from stl import mesh
import vtk
import re
import argparse
import traceback
import subprocess

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

def save_mesh(path_to_result, labels, x_res=1, y_res=1, z_res=1,
    poly_reduction=0.9, smoothing_iterations=15):

    # get labels
    zsh, ysh, xsh = labels.shape
    allLabels = np.unique(labels)
    b = np.empty_like(labels)
    arr = np.empty((0,3,3))
    nTotalCells = [0]

    for label in allLabels[1:]:

        # get label
        b.fill(0)
        b[labels==label] = 1

        # numpy to vtk
        sc = numpy_to_vtk(num_array=b.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        imageData = vtk.vtkImageData()
        imageData.SetOrigin(0, 0, 0)
        imageData.SetSpacing(x_res, y_res, z_res)
        imageData.SetDimensions(xsh, ysh, zsh)
        #imageData.SetExtent(0,xsh-1,0,ysh-1,0,zsh-1)
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
    mesh_final.save(path_to_result)

def get_voxel_spacing(header, extension):
    if extension == '.am':
        # read header as string
        b = header[0].tobytes()
        try:
            s = b.decode("utf-8")
        except:
            s = b.decode("latin1")
        # get physical size from image header
        lattice = re.search('define Lattice (.*)\n', s)
        bounding_box = re.search('BoundingBox (.*),\n', s)
        if bounding_box and lattice:
            # get number of voxels
            lattice = lattice.group(1)
            xsh, ysh, zsh = lattice.split(' ')
            xsh, ysh, zsh = float(xsh), float(ysh), float(zsh)
            # get bounding box
            bounding_box = bounding_box.group(1)
            i0, i1, i2, i3, i4, i5 = bounding_box.split(' ')
            # calculate voxel spacing
            xres = (float(i1)-float(i0)) / xsh
            yres = (float(i3)-float(i2)) / ysh
            zres = (float(i5)-float(i4)) / zsh
        else:
            xres, yres, zres = 1, 1, 1
    elif extension in ['.hdr', '.mhd', '.mha', '.nrrd', '.nii', '.nii.gz']:
        xres, yres, zres = header.GetSpacing()
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

def init_create_mesh(id):

    import django
    django.setup()
    from biomedisa_app.models import Upload
    from biomedisa_app.config import config
    from biomedisa_app.views import send_data_to_host, qsub_start, qsub_stop

    # get object
    try:
        img = Upload.objects.get(pk=id)
    except Upload.DoesNotExist:
        img.status = 0
        img.save()
        message = 'File has been removed.'
        Upload.objects.create(user=img.user, project=img.project, log=1, imageType=None, shortfilename=message)

    # get host information
    host = ''
    host_base = biomedisa.BASE_DIR
    subhost, qsub_pid = None, None
    if 'REMOTE_QUEUE_HOST' in config:
        host = config['REMOTE_QUEUE_HOST']
    if host and 'REMOTE_QUEUE_BASE_DIR' in config:
        host_base = config['REMOTE_QUEUE_BASE_DIR']

    # check if aborted
    if img.status > 0:

        # return error
        if img.imageType not in [2,3]:
            Upload.objects.create(user=img.user, project=img.project,
                log=1, imageType=None, shortfilename='No valid label data.')

        else:
            # set status to processing
            img.status = 2
            img.save()

            # create path to result
            filename, extension = os.path.splitext(img.pic.path)
            if extension == '.gz':
                extension = '.nii.gz'
                filename = filename[:-4]
            path_to_result = unique_file_path(filename + '.stl')
            new_short_name = os.path.basename(path_to_result)
            pic_path = 'images/%s/%s' %(img.user.username, new_short_name)

            # remote server
            if host:

                # command
                cmd = ['python3', host_base+'/biomedisa/mesh.py', img.pic.path.replace(biomedisa.BASE_DIR,host_base)]
                cmd += [f'-iid={img.id}', '-r']

                # create user directory
                subprocess.Popen(['ssh', host, 'mkdir', '-p', host_base+'/private_storage/images/'+img.user.username]).wait()

                # send data to host
                success = send_data_to_host(img.pic.path, host+':'+img.pic.path.replace(biomedisa.BASE_DIR,host_base))

                # qsub start
                if 'REMOTE_QUEUE_QSUB' in config and config['REMOTE_QUEUE_QSUB']:
                    subhost, qsub_pid = qsub_start(host, host_base, 7)

                # check if aborted
                img = Upload.objects.get(pk=img.id)
                if img.status==2 and img.queue==7 and success==0:

                    # set pid and processing status
                    img.message = 'Processing'
                    img.pid = -1
                    img.save()

                    # create mesh
                    if subhost:
                        cmd = ['ssh', '-t', host, 'ssh', subhost] + cmd
                    else:
                        cmd = ['ssh', host] + cmd
                    subprocess.Popen(cmd).wait()

                    # check if aborted
                    success = subprocess.Popen(['scp', host+':'+host_base+f'/log/pid_7', biomedisa.BASE_DIR+f'/log/pid_7']).wait()

                    # get result
                    if success==0:
                        # remove pid file
                        subprocess.Popen(['ssh', host, 'rm', host_base+f'/log/pid_7']).wait()

                        result_on_host = img.pic.path.replace(biomedisa.BASE_DIR,host_base)
                        result_on_host = result_on_host.replace(os.path.splitext(result_on_host)[1],'.stl')
                        success = subprocess.Popen(['scp', host+':'+result_on_host, path_to_result]).wait()

                        if success==0:
                            # create object
                            Upload.objects.create(pic=pic_path, user=img.user, project=img.project,
                                imageType=5, shortfilename=new_short_name)
                        else:
                            # return error
                            Upload.objects.create(user=img.user, project=img.project,
                                log=1, imageType=None, shortfilename='Invalid label data.')

            # local server
            else:

                # set pid and processing status
                img.pid = int(os.getpid())
                img.message = 'Processing'
                img.save()

                # load data
                data, header = load_data(img.pic.path, process='converter')
                if data is None:
                    # return error
                    Upload.objects.create(user=img.user, project=img.project,
                        log=1, imageType=None, shortfilename='Invalid label data.')
                else:
                    # get voxel spacing
                    xres, yres, zres = get_voxel_spacing(header, extension)
                    print(f'Voxel spacing: x_spacing, y_spacing, z_spacing = {xres}, {yres}, {zres}')

                    # create stl file
                    save_mesh(path_to_result, data, xres, yres, zres)

                    # create object
                    Upload.objects.create(pic=pic_path, user=img.user, project=img.project,
                        imageType=5, shortfilename=new_short_name)

        # close process
        img.status = 0
        img.pid = 0
        img.save()

    # qsub stop
    if 'REMOTE_QUEUE_QSUB' in config and config['REMOTE_QUEUE_QSUB']:
        qsub_stop(host, host_base, 7, 'create_mesh', subhost, qsub_pid)

if __name__ == "__main__":

    # initialize arguments
    parser = argparse.ArgumentParser(description='Biomedisa mesh generator.',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required arguments
    parser.add_argument('path_to_labels', type=str, metavar='PATH_TO_LABELS',
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
    parser.add_argument('-iid','--img_id', type=str, default=None,
                        help='Label ID within django environment/browser version')
    parser.add_argument('-r','--remote', action='store_true', default=False,
                        help='The mesh is created on a remote server. Must be set up in config.py')
    bm = parser.parse_args()

    # set pid
    if bm.remote:
        create_pid_object(os.getpid(), True, 7, bm.img_id)

    # load data
    bm.labels, header, extension = load_data(bm.path_to_labels, return_extension=True)

    if bm.labels is None:
        print('Error: Invalid label data.')

    else:
        # path to result
        path_to_result = bm.path_to_labels.replace(os.path.splitext(bm.path_to_labels)[1],'.stl')

        # get voxel spacing
        if not all([bm.x_res, bm.y_res, bm.z_res]):
            x_res, y_res, z_res = get_voxel_spacing(header, extension)
            if not bm.x_res:
                bm.x_res = x_res
            if not bm.y_res:
                bm.y_res = y_res
            if not bm.z_res:
                bm.z_res = z_res
            print(f'Voxel spacing: x_spacing, y_spacing, z_spacing = {bm.x_res}, {bm.y_res}, {bm.z_res}')

        # create mesh
        try:
            save_mesh(path_to_result, bm.labels, bm.x_res, bm.y_res, bm.z_res, bm.poly_reduction, bm.smoothing_iterations)
        except Exception as e:
            print(traceback.format_exc())

