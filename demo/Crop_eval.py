##########################################################################
##                                                                      ##
##  Copyright (c) 2022 Philipp Lösel. All rights reserved.              ##
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

from crop_helper import *
import sys

path_to_img = sys.argv[1]
path_to_labels = sys.argv[2]
path_to_model  = sys.argv[3]

# parameters
parameters = sys.argv
balance = 1 if any(x in parameters for x in ['-balance','-b']) else 0     # balance class members of training patches
flip_x = True if any(x in parameters for x in ['-flip_x']) else False     # flip axis during training
flip_y = True if any(x in parameters for x in ['-flip_y']) else False     # flip axis during training
flip_z = True if any(x in parameters for x in ['-flip_z']) else False     # flip axis during training

compress = 6            # wheter final result should be compressed or not
epochs = 200            # epochs the network is trained
channels = 1            # use voxel coordinates
normalize = 1           # normalize images before training
x_scale = 256           # images are scaled at x-axis to this size before training
y_scale = 256           # images are scaled at y-axis to this size before training
z_scale = 256           # images are scaled at z-axis to this size before training
rotate = 0              # randomly rotate during training
validation_split = 0.0  # percentage used for validation
stride_size = 32        # stride size for patches
batch_size = 24         # batch size

for k in range(len(parameters)):
    if parameters[k] in ['-compress','-c']:
        compress = int(parameters[k+1])
    if parameters[k] in ['-epochs','-e']:
        epochs = int(parameters[k+1])
    if parameters[k] in ['-channels']:
        channels = int(parameters[k+1])
    if parameters[k] in ['-normalize']:
        normalize = int(parameters[k+1])
    if parameters[k] in ['-xsize','-xs']:
        x_scale = int(parameters[k+1])
    if parameters[k] in ['-ysize','-ys']:
        y_scale = int(parameters[k+1])
    if parameters[k] in ['-zsize','-zs']:
        z_scale = int(parameters[k+1])
    if parameters[k] in ['-rotate','-r']:
        rotate = int(parameters[k+1])
    if parameters[k] in ['-validation_split','-vs']:
        validation_split = float(parameters[k+1])
    if parameters[k] in ['-stride_size','-ss']:
        stride_size = int(parameters[k+1])
    if parameters[k] in ['-batch_size','-bs']:
        batch_size = int(parameters[k+1])


evaluate_network(normalize,path_to_img,path_to_labels,path_to_model,
                batch_size,x_scale,y_scale,z_scale,
                flip_x,flip_y,flip_z,rotate)

