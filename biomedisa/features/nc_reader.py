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
import glob
import numpy as np

def save_nc_block(path_to_dst, arr, path_to_src, offset):
    try:
        import netCDF4
    except:
        raise Exception("netCDF4 not found. please use `pip install netCDF4`")
    with netCDF4.Dataset(path_to_src, 'r') as src:
        with netCDF4.Dataset(path_to_dst, 'w') as dst:
            # copy global attributes all at once via dictionary
            dst.setncatts(src.__dict__)
            # copy dimensions
            for name, dimension in src.dimensions.items():
                dst.createDimension(
                    name, (len(dimension) if not dimension.isunlimited() else None))
            # copy all file data
            for name, variable in src.variables.items():
                if name in ['labels','segmented']:
                    srcarr = src[name][:]
                    zsh, ysh, xsh = srcarr.shape
                    x = dst.createVariable(name, variable.datatype, variable.dimensions, compression='zlib')
                    dst[name][:] = arr[offset:offset+zsh]
                elif name == 'tomo':
                    srcarr = src[name][:]
                    zsh, ysh, xsh = srcarr.shape
                    x = dst.createVariable(name, variable.datatype, variable.dimensions)
                    dst[name][:] = arr[offset:offset+zsh]
                else:
                    x = dst.createVariable(name, variable.datatype, variable.dimensions)
                    dst[name][:] = src[name][:]
                # copy variable attributes all at once via dictionary
                dst[name].setncatts(src[name].__dict__)
    return offset+zsh

def np_to_nc(results_dir, labeled_array, header=None, reference_dir=None, reference_file=None, start=0, stop=None):
    try:
        import netCDF4
    except:
        raise Exception("netCDF4 not found. please use `pip install netCDF4`")

    # save as file or directory
    is_file = False
    if os.path.splitext(results_dir)[1] == '.nc':
        is_file = True

    # get reference information
    if reference_dir:
        ref_files = glob.glob(reference_dir+'/*.nc')
        ref_files.sort()
    elif header:
        ref_files = header[1]
        reference_dir = os.path.dirname(ref_files[0])
    elif reference_file:
        ref_files = [reference_file]
    else:
        raise Exception("reference file(s) required")

    if is_file and len(ref_files) > 1:
        raise Exception("reference needs to be a file")

    if not stop:
        stop = len(ref_files)-1

    # save volume by volume
    offset = 0
    for path_to_src in ref_files[start:stop+1]:
        if is_file:
            path_to_dst = results_dir
        else:
            path_to_dst = results_dir + '/' + os.path.basename(path_to_src)
        offset = save_nc_block(path_to_dst, labeled_array, path_to_src, offset)

def nc_to_np(base_dir, start=0, stop=None, show_keys=False):
    try:
        import netCDF4
    except:
        raise Exception("netCDF4 not found. please use `pip install netCDF4`")

    if os.path.isfile(base_dir):
        # decompress bz2 files
        if '.bz2' in base_dir:
            import bz2
            zipfile = bz2.BZ2File(base_dir) # open the file
            data = zipfile.read() # get the decompressed data
            newfilepath = base_dir[:-4] # assuming the filepath ends with .bz2
            open(newfilepath, 'wb').write(data)
            f = netCDF4.Dataset(newfilepath,'r')
        else:
            f = netCDF4.Dataset(base_dir,'r')
        if show_keys:
            print(f.variables.keys())
        for n in ['labels', 'segmented', 'tomo']:
            if n in f.variables.keys():
                name = n
        output = f.variables[name]
        output = np.copy(output, order='C')
        # remove tmp file
        if '.bz2' in base_dir:
            os.remove(newfilepath)
        header = [name, [base_dir], output.dtype]

    elif os.path.isdir(base_dir):
        # read volume by volume
        files = glob.glob(base_dir+'/**/*.nc', recursive=True)
        files += glob.glob(base_dir+'/**/*.bz2', recursive=True)
        files.sort()

        # check for compression
        if os.path.splitext(files[0])[1]=='.bz2':
            import bz2

        if not stop:
            stop = len(files)-1

        for i,filepath in enumerate(files[start:stop+1]):

            # decompress bz2 files
            if '.bz2' in filepath:
                zipfile = bz2.BZ2File(filepath) # open the file
                data = zipfile.read() # get the decompressed data
                newfilepath = filepath[:-4] # assuming the filepath ends with .bz2
                open(newfilepath, 'wb').write(data)
                f = netCDF4.Dataset(newfilepath,'r')
            else:
                f = netCDF4.Dataset(filepath,'r')

            if show_keys:
                print(f.variables.keys())
            for n in ['labels', 'segmented', 'tomo']:
                if n in f.variables.keys():
                    name = n

            a = f.variables[name]
            a = np.copy(a, order='C')

            # remove tmp file
            if '.bz2' in filepath:
                os.remove(newfilepath)

            # append output array
            if i==0:
                output = a
            else:
                output = np.append(output, a, axis=0)

        header = [name, files[start:stop+1], a.dtype]

    return output, header

