##########################################################################
##                                                                      ##
##  Copyright (c) 2020 Philipp Lösel. All rights reserved.              ##
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

import os, sys, glob
from biomedisa_helper import load_data, save_data
import numpy as np
import subprocess
import platform

if __name__ == '__main__':

    # path to data
    path_to_data = sys.argv[1]
    path_to_labels = sys.argv[2]

    # get arguments
    nump = 1
    smooth = 0
    overlap = 100
    sub_z, sub_y, sub_x = 1, 1, 1
    for i, val in enumerate(sys.argv):
        if val in ['--split_z','-sz']:
            sub_z = max(int(sys.argv[i+1]), 1)
        if val in ['--split_y','-sy']:
            sub_y = max(int(sys.argv[i+1]), 1)
        if val in ['--split_x','-sx']:
            sub_x = max(int(sys.argv[i+1]), 1)
        if val in ['--overlap','-ol']:
            overlap = max(int(sys.argv[i+1]), 0)
        if val in ['-n','-np']:
            nump = max(int(sys.argv[i+1]), 1)
        if val in ['--smooth','-s']:
            smooth = int(sys.argv[i+1])
    uq = True if any(x in sys.argv for x in ['--uncertainty','-uq']) else False
    allx = 1 if '-allx' in sys.argv else 0

    # base directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # clean tmp folder
    filelist = glob.glob(BASE_DIR+'/tmp/*.tif')
    for f in filelist:
        os.remove(f)

    # data shape
    data, _ = load_data(path_to_data, 'split_volume')
    shape = np.copy(np.array(data.shape))
    zsh, ysh, xsh = shape
    del data

    # split volume
    sub_size_z = np.ceil(zsh / sub_z)
    sub_size_y = np.ceil(ysh / sub_y)
    sub_size_x = np.ceil(xsh / sub_x)

    # iterate over subvolumes
    for sub_z_i in range(sub_z):
        for sub_y_i in range(sub_y):
            for sub_x_i in range(sub_x):
                subvolume = sub_z_i*sub_y*sub_x + sub_y_i*sub_x + sub_x_i + 1
                print('Subvolume:', subvolume, '/', sub_z*sub_y*sub_x)

                # determine z subvolume
                blockmin_z = int(sub_z_i * sub_size_z)
                blockmax_z = int((sub_z_i+1) * sub_size_z)
                datamin_z = max(blockmin_z - overlap, 0)
                datamax_z = min(blockmax_z + overlap, zsh)

                # determine y subvolume
                blockmin_y = int(sub_y_i * sub_size_y)
                blockmax_y = int((sub_y_i+1) * sub_size_y)
                datamin_y = max(blockmin_y - overlap, 0)
                datamax_y = min(blockmax_y + overlap, ysh)

                # determine x subvolume
                blockmin_x = int(sub_x_i * sub_size_x)
                blockmax_x = int((sub_x_i+1) * sub_size_x)
                datamin_x = max(blockmin_x - overlap, 0)
                datamax_x = min(blockmax_x + overlap, xsh)

                # extract image subvolume
                data, _ = load_data(path_to_data, 'split_volume')
                save_data(BASE_DIR+f'/tmp/sub_volume_{subvolume}.tif', data[datamin_z:datamax_z,datamin_y:datamax_y,datamin_x:datamax_x], False)
                del data

                # extract label subvolume
                labelData, header, final_image_type = load_data(path_to_labels, 'split_volume', True)
                save_data(BASE_DIR+'/tmp/labels.sub_volume.tif', labelData[datamin_z:datamax_z,datamin_y:datamax_y,datamin_x:datamax_x])
                del labelData

                # configure command
                cmd = ['mpiexec', '-np', f'{nump}', 'python3', 'biomedisa_interpolation.py', BASE_DIR+f'/tmp/sub_volume_{subvolume}.tif', BASE_DIR+'/tmp/labels.sub_volume.tif', '-s', f'{smooth}']
                if uq:
                    cmd.append('-uq')
                if allx:
                    cmd.append('-allx')
                cwd = BASE_DIR + '/demo/'

                # run segmentation
                if platform.system() == 'Windows':
                    cmd[3] = 'python'
                    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE)
                    for line in iter(p.stdout.readline, b''):
                        line = str(line,'utf-8')
                        print(line.rstrip())
                    p.stdout.close()
                else:
                    p = subprocess.Popen(cmd, cwd=cwd)
                p.wait()

                # remove tmp files
                os.remove(BASE_DIR+f'/tmp/sub_volume_{subvolume}.tif')
                os.remove(BASE_DIR+'/tmp/labels.sub_volume.tif')

    # create path_to_final
    filename, extension = os.path.splitext(os.path.basename(path_to_data))
    if extension == '.gz':
        filename = filename[:-4]
    filename = 'final.' + filename
    path_to_final = path_to_data.replace(os.path.basename(path_to_data), filename + final_image_type)

    # path_to_uq and path_to_smooth
    filename, extension = os.path.splitext(path_to_final)
    if extension == '.gz':
        filename = filename[:-4]
    path_to_smooth = filename + '.smooth' + final_image_type
    path_to_uq = filename + '.uncertainty.tif'

    # iterate over subvolumes
    final = np.zeros((zsh, ysh, xsh), dtype=np.uint8)
    for sub_z_i in range(sub_z):
        for sub_y_i in range(sub_y):
            for sub_x_i in range(sub_x):
                subvolume = sub_z_i*sub_y*sub_x + sub_y_i*sub_x + sub_x_i + 1
                print('Subvolume:', subvolume, '/', sub_z*sub_y*sub_x)

                # determine z subvolume
                blockmin_z = int(sub_z_i * sub_size_z)
                blockmax_z = int((sub_z_i+1) * sub_size_z)
                datamin_z = max(blockmin_z - overlap, 0)
                datamax_z = min(blockmax_z + overlap, zsh)

                # determine y subvolume
                blockmin_y = int(sub_y_i * sub_size_y)
                blockmax_y = int((sub_y_i+1) * sub_size_y)
                datamin_y = max(blockmin_y - overlap, 0)
                datamax_y = min(blockmax_y + overlap, ysh)

                # determine x subvolume
                blockmin_x = int(sub_x_i * sub_size_x)
                blockmax_x = int((sub_x_i+1) * sub_size_x)
                datamin_x = max(blockmin_x - overlap, 0)
                datamax_x = min(blockmax_x + overlap, xsh)

                # load subvolume
                path_to_subvolume = BASE_DIR+f'/tmp/final.sub_volume_{subvolume}.tif'
                if os.path.isfile(path_to_subvolume):
                    tmp, _ = load_data(path_to_subvolume)
                    final[blockmin_z:blockmax_z,blockmin_y:blockmax_y,blockmin_x:blockmax_x] \
                    = tmp[blockmin_z-datamin_z:blockmax_z-datamin_z,blockmin_y-datamin_y:blockmax_y-datamin_y,blockmin_x-datamin_x:blockmax_x-datamin_x]
                    os.remove(path_to_subvolume)

    # save result
    save_data(path_to_final, final, header)

    # iterate over subvolumes (smooth)
    smooth = 0
    final.fill(0)
    for sub_z_i in range(sub_z):
        for sub_y_i in range(sub_y):
            for sub_x_i in range(sub_x):
                subvolume = sub_z_i*sub_y*sub_x + sub_y_i*sub_x + sub_x_i + 1
                print('Subvolume:', subvolume, '/', sub_z*sub_y*sub_x)

                # determine z subvolume
                blockmin_z = int(sub_z_i * sub_size_z)
                blockmax_z = int((sub_z_i+1) * sub_size_z)
                datamin_z = max(blockmin_z - overlap, 0)
                datamax_z = min(blockmax_z + overlap, zsh)

                # determine y subvolume
                blockmin_y = int(sub_y_i * sub_size_y)
                blockmax_y = int((sub_y_i+1) * sub_size_y)
                datamin_y = max(blockmin_y - overlap, 0)
                datamax_y = min(blockmax_y + overlap, ysh)

                # determine x subvolume
                blockmin_x = int(sub_x_i * sub_size_x)
                blockmax_x = int((sub_x_i+1) * sub_size_x)
                datamin_x = max(blockmin_x - overlap, 0)
                datamax_x = min(blockmax_x + overlap, xsh)

                # load subvolume
                path_to_subvolume = BASE_DIR+f'/tmp/final.sub_volume_{subvolume}.smooth.tif'
                if os.path.isfile(path_to_subvolume):
                    tmp, _ = load_data(path_to_subvolume)
                    final[blockmin_z:blockmax_z,blockmin_y:blockmax_y,blockmin_x:blockmax_x] \
                    = tmp[blockmin_z-datamin_z:blockmax_z-datamin_z,blockmin_y-datamin_y:blockmax_y-datamin_y,blockmin_x-datamin_x:blockmax_x-datamin_x]
                    os.remove(path_to_subvolume)
                    smooth = 1

    # save result
    if smooth:
        save_data(path_to_smooth, final, header)

    # iterate over subvolumes (uncertainty)
    uncertainty = 0
    final.fill(0)
    for sub_z_i in range(sub_z):
        for sub_y_i in range(sub_y):
            for sub_x_i in range(sub_x):
                subvolume = sub_z_i*sub_y*sub_x + sub_y_i*sub_x + sub_x_i + 1
                print('Subvolume:', subvolume, '/', sub_z*sub_y*sub_x)

                # determine z subvolume
                blockmin_z = int(sub_z_i * sub_size_z)
                blockmax_z = int((sub_z_i+1) * sub_size_z)
                datamin_z = max(blockmin_z - overlap, 0)
                datamax_z = min(blockmax_z + overlap, zsh)

                # determine y subvolume
                blockmin_y = int(sub_y_i * sub_size_y)
                blockmax_y = int((sub_y_i+1) * sub_size_y)
                datamin_y = max(blockmin_y - overlap, 0)
                datamax_y = min(blockmax_y + overlap, ysh)

                # determine x subvolume
                blockmin_x = int(sub_x_i * sub_size_x)
                blockmax_x = int((sub_x_i+1) * sub_size_x)
                datamin_x = max(blockmin_x - overlap, 0)
                datamax_x = min(blockmax_x + overlap, xsh)

                # load subvolume
                path_to_subvolume = BASE_DIR+f'/tmp/final.sub_volume_{subvolume}.uncertainty.tif'
                if os.path.isfile(path_to_subvolume):
                    tmp, _ = load_data(path_to_subvolume)
                    final[blockmin_z:blockmax_z,blockmin_y:blockmax_y,blockmin_x:blockmax_x] \
                    = tmp[blockmin_z-datamin_z:blockmax_z-datamin_z,blockmin_y-datamin_y:blockmax_y-datamin_y,blockmin_x-datamin_x:blockmax_x-datamin_x]
                    os.remove(path_to_subvolume)
                    uncertainty = 1

    # save result
    if uncertainty:
        save_data(path_to_uq, final, header)

