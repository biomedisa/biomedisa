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

if __name__ == "__main__":

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # check if executable
    executable = comm.recv(source=0, tag=0)

    if executable:

        # get small or large
        small = comm.recv(source=0, tag=1)

        # get number of gpus
        ngpus = comm.recv(source=0, tag=2)

        # create sub communicator
        if rank >= ngpus:
            sub_comm = MPI.Comm.Split(comm, 0, rank)     # set process to idle
        else:
            sub_comm = MPI.Comm.Split(comm, 1, rank)

            if small:
                from rw_small import _diffusion_child
                _diffusion_child(sub_comm)
            else:
                from rw_large import _diffusion_child
                _diffusion_child(sub_comm)
