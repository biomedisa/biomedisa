##########################################################################
##                                                                      ##
##  Copyright (c) 2023 Philipp Lösel. All rights reserved.              ##
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
                if name != 'labels':
                    x = dst.createVariable(name, variable.datatype, variable.dimensions)
                    dst[name][:] = src[name][:]
                else:
                    srcarr = src[name][:]
                    zsh, ysh, xsh = srcarr.shape
                    x = dst.createVariable(name, variable.datatype, variable.dimensions, compression='zlib')
                    dst[name][:] = arr[offset:offset+zsh]
                # copy variable attributes all at once via dictionary
                dst[name].setncatts(src[name].__dict__)
    return offset+zsh

def np_to_nc(results_dir, labeled_array, reference_dir, start=0, stop=None):
    try:
        import netCDF4
    except:
        raise Exception("netCDF4 not found. please use `pip install netCDF4`")
    offset = 0
    if not stop:
        stop = len(glob.glob(reference_dir+'/*.nc'))
    for i in range(start, stop+1):
        filepath = reference_dir+f'/block{str(i).zfill(8)}.nc'
        offset = save_nc_block(results_dir+f'/block{str(i).zfill(8)}.nc', labeled_array, filepath, offset)

def nc_to_np(base_dir, start=0, stop=None, file=False, show_keys=False, compressed=False):
    try:
        import netCDF4
    except:
        raise Exception("netCDF4 not found. please use `pip install netCDF4`")
    if file:
        # read nc object
        f = netCDF4.Dataset(base_dir,'r')
        if show_keys:
            print(f.variables.keys())
        for n in ['labels', 'segmented', 'tomo']:
            if n in f.variables.keys():
                name = n
        output = f.variables[name]
        output = np.copy(output)
    else:
        # read volume by volume
        extension='.nc'
        if compressed:
            extension='.nc.bz2'
        if not stop:
            stop = len(glob.glob(base_dir+'/*'+extension))
        for i in range(start, stop+1):
            filepath = base_dir+f'/block{str(i).zfill(8)}'+extension

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
            a = np.copy(a)

            if '.bz2' in filepath:
                os.remove(newfilepath)

            # append output array
            if i==start:
                output = a
            else:
                output = np.append(output, a, axis=0)

    return output, name

