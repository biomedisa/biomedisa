#!/bin/bash
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

SERVICE="apache2"
if pgrep -x "$SERVICE" >/dev/null
then
    echo "Biomedisa is running."
    export BROWSER="powershell.exe /C start"
    xdg-open http://localhost
else
    export CUDA_HOME=/usr/local/cuda-11.8
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
    export PATH=${CUDA_HOME}/bin:${PATH}
    export SCREENDIR=/home/biomedisa/.screen

    service apache2 restart
    service mysql start
    service redis-server restart
    runuser --user biomedisa -- screen -wipe
    runuser --user biomedisa -- /home/biomedisa/git/biomedisa/start_workers.sh
    export BROWSER="powershell.exe /C start"
    xdg-open http://localhost

    echo "      #####################################################"
    echo "      ##                                                 ##"
    echo "      ##        Biomedisa started successfully!          ##"
    echo "      ##                                                 ##"
    echo "      ##    If Biomedisa didn't start automatically,     ##"
    echo "      ##     open http://localhost in a web broser.      ##"
    echo "      ##                                                 ##"
    echo "      #####################################################"
fi

exit 0

