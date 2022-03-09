#!/bin/bash
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

if [[ ${1: -1} = p ]]; then 
    export CUDA_HOME=/usr/local/cuda-11.0
else
    export CUDA_HOME=/usr/local/cuda-11.4
fi
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

if [[ ${2} = -n ]]; then 
    mpiexec ${2} ${3} python3 ~/git/biomedisa/demo/biomedisa_interpolation.py "${@:4}"
else
    python3 ~/git/biomedisa/demo/biomedisa_interpolation.py "${@:2}"
fi

exit 0
