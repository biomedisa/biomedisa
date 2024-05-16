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
from biomedisa.features.biomedisa_helper import load_data, save_data
from biomedisa.interpolation import smart_interpolation
from tifffile import imread, imwrite, TiffFile
import numpy as np
import argparse

if __name__ == '__main__':

    # initialize arguments
    parser = argparse.ArgumentParser(description='Biomedisa interpolation.',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required arguments
    parser.add_argument('path_to_data', type=str, metavar='PATH_TO_IMAGE',
                        help='Location of image data')
    parser.add_argument('path_to_labels', type=str, metavar='PATH_TO_LABELS',
                        help='Location of label data')

    # optional arguments
    parser.add_argument('-allx', '--allaxis', action='store_true', default=False,
                        help='If pre-segmentation is not exlusively in xy-plane')
    parser.add_argument('-u', '--uncertainty', action='store_true', default=False,
                        help='Return uncertainty of segmentation result')
    parser.add_argument('-s', '--smooth', nargs='?', type=int, const=100, default=0,
                        help='Number of smoothing iterations for segmentation result')
    parser.add_argument('-ol', '--overlap', type=int, default=50,
                        help='Overlap of sub-blocks')
    parser.add_argument('-sx','--split_x', type=int, default=1,
                        help='Number of sub-blocks in x-direction')
    parser.add_argument('-sy','--split_y', type=int, default=1,
                        help='Number of sub-blocks in y-direction')
    parser.add_argument('-sz','--split_z', type=int, default=1,
                        help='Number of sub-blocks in z-direction')
    args = parser.parse_args()

    # image size
    if args.path_to_data[-4:] == '.tif':
        tif = TiffFile(args.path_to_data)
        zsh = len(tif.pages)
        ysh, xsh = tif.pages[0].shape
    else:
        print('Warning: This script is optimized for TIFF files. Please consider saving your data in TIFF format.')
        data, _ = load_data(args.path_to_data)
        shape = np.array(data.shape).copy()
        zsh, ysh, xsh = shape
        del data

    if args.path_to_labels[-4:] != '.tif':
        print('Warning: This script is optimized for TIFF files. Please consider saving your labels in TIFF format.')

    # split volume
    sub_size_z = np.ceil(zsh / args.split_z)
    sub_size_y = np.ceil(ysh / args.split_y)
    sub_size_x = np.ceil(xsh / args.split_x)

    # allocate memory
    final = np.zeros((zsh, ysh, xsh), dtype=np.uint8)
    if args.smooth:
        final_smooth = np.zeros_like(final)
    if args.uncertainty:
        final_uncertainty = np.zeros_like(final)

    # iterate over subvolumes
    for z_i in range(args.split_z):
        for y_i in range(args.split_y):
            for x_i in range(args.split_x):
                subvolume = z_i * args.split_y * args.split_x + y_i * args.split_x + x_i + 1
                print('Subvolume:', subvolume, '/', args.split_z * args.split_y * args.split_x)

                # determine z subvolume
                blockmin_z = int(z_i * sub_size_z)
                blockmax_z = int((z_i+1) * sub_size_z)
                datamin_z = max(blockmin_z - args.overlap, 0)
                datamax_z = min(blockmax_z + args.overlap, zsh)

                # determine y subvolume
                blockmin_y = int(y_i * sub_size_y)
                blockmax_y = int((y_i+1) * sub_size_y)
                datamin_y = max(blockmin_y - args.overlap, 0)
                datamax_y = min(blockmax_y + args.overlap, ysh)

                # determine x subvolume
                blockmin_x = int(x_i * sub_size_x)
                blockmax_x = int((x_i+1) * sub_size_x)
                datamin_x = max(blockmin_x - args.overlap, 0)
                datamax_x = min(blockmax_x + args.overlap, xsh)

                # extract image subvolume
                if args.path_to_data[-4:] == '.tif':
                    data = imread(args.path_to_data, key=range(datamin_z,datamax_z))
                    data = data[:,datamin_y:datamax_y,datamin_x:datamax_x].copy()
                else:
                    data, _ = load_data(args.path_to_data)
                    data = data[datamin_z:datamax_z,datamin_y:datamax_y,datamin_x:datamax_x].copy()

                # extract label subvolume
                if args.path_to_labels[-4:] == '.tif':
                    header, final_image_type = None, '.tif'
                    labelData = imread(args.path_to_labels, key=range(datamin_z,datamax_z))
                    labelData = labelData[:,datamin_y:datamax_y,datamin_x:datamax_x].copy()
                else:
                    labelData, header, final_image_type = load_data(args.path_to_labels, return_extension=True)
                    labelData = labelData[datamin_z:datamax_z,datamin_y:datamax_y,datamin_x:datamax_x].copy()

                # interpolation
                results = smart_interpolation(data, labelData, uncertainty=args.uncertainty, allaxis=args.allaxis, smooth=args.smooth)

                # append results
                final[blockmin_z:blockmax_z,blockmin_y:blockmax_y,blockmin_x:blockmax_x] \
                    = results['regular'][blockmin_z-datamin_z:blockmax_z-datamin_z,blockmin_y-datamin_y:blockmax_y-datamin_y,blockmin_x-datamin_x:blockmax_x-datamin_x]
                if 'smooth' in results and results['smooth'] is not None:
                    final_smooth[blockmin_z:blockmax_z,blockmin_y:blockmax_y,blockmin_x:blockmax_x] \
                        = results['smooth'][blockmin_z-datamin_z:blockmax_z-datamin_z,blockmin_y-datamin_y:blockmax_y-datamin_y,blockmin_x-datamin_x:blockmax_x-datamin_x]
                if 'uncertainty' in results and results['uncertainty'] is not None:
                    final_uncertainty[blockmin_z:blockmax_z,blockmin_y:blockmax_y,blockmin_x:blockmax_x] \
                        = results['uncertainty'][blockmin_z-datamin_z:blockmax_z-datamin_z,blockmin_y-datamin_y:blockmax_y-datamin_y,blockmin_x-datamin_x:blockmax_x-datamin_x]

    # path to regular result
    filename, extension = os.path.splitext(os.path.basename(args.path_to_data))
    if extension == '.gz':
        filename = filename[:-4]
    filename = 'final.' + filename
    path_to_final = args.path_to_data.replace(os.path.basename(args.path_to_data), filename + final_image_type)

    # path to optional results
    filename, extension = os.path.splitext(path_to_final)
    if extension == '.gz':
        filename = filename[:-4]
    path_to_smooth = filename + '.smooth' + final_image_type
    path_to_uq = filename + '.uncertainty.tif'

    # save results
    save_data(path_to_final, final, header)
    if args.smooth:
        save_data(path_to_smooth, final_smooth, header)
    if args.uncertainty:
        imwrite(path_to_uq, final_uncertainty)

