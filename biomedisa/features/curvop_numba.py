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

import numpy as np
import numba

@numba.jit(nopython=True)
def evolution(mean, A, data, alpha):
    zsh, ysh, xsh = A.shape
    B = np.copy(A)
    for k in range(1, zsh-1):
        for l in range(1, ysh-1):
            for m in range(1, xsh-1):
                refLabel = A[k,l,m]
                finLabel = refLabel
                img = data[k,l,m]
                if refLabel == 0:
                    ref = alpha * abs(mean[refLabel] - img)
                else:
                    ref = abs(mean[refLabel] - img)
                for n in range(-1, 2):
                    for o in range(-1, 2):
                        for p in range(-1, 2):
                            tmpLabel = A[k+n, l+o, m+p]
                            if tmpLabel != refLabel:
                                if tmpLabel == 0:
                                    val = alpha * abs(mean[tmpLabel] - img)
                                else:
                                    val = abs(mean[tmpLabel] - img)
                                if val < ref:
                                    ref = val
                                    finLabel = tmpLabel
                B[k,l,m] = finLabel
    return B

@numba.jit(nopython=True)
def erosion(start, final, _P3, zsh, ysh, xsh, label):
    for plane in range(1, zsh-1):
        for row in range(1, ysh-1):
            for column in range(1, xsh-1):
                found = 0
                for n in range(-1, 2):
                    for o in range(-1, 2):
                        for p in range(-1, 2):
                            if start[plane+n, row+o, column+p] != label:
                                found = 1
                if start[plane, row, column] == label and found == 1:
                    t, found = 0, 0
                    while t < 9 and found == 0:
                        value = 0
                        for k in range(3):
                            for l in range(3):
                                for m in range(3):
                                    if _P3[t,k,l,m] == 1:
                                        value += abs(start[plane-1+k, row-1+l, column-1+m] - label)
                        if value == 0:
                            found = 1
                        t += 1
                    if found == 0:
                        subLabel = label
                        for n in range(-1, 2):
                            for o in range(-1, 2):
                                for p in range(-1, 2):
                                    tmpLabel = start[plane+n, row+o, column+p]
                                    if tmpLabel != label:
                                        subLabel = tmpLabel
                        final[plane, row, column] = subLabel
    return start, final

@numba.jit(nopython=True)
def dilation(start, final, _P3, zsh, ysh, xsh, label):
    for plane in range(1, zsh-1):
        for row in range(1, ysh-1):
            for column in range(1, xsh-1):
                found = 0
                for n in range(-1, 2):
                    for o in range(-1, 2):
                        for p in range(-1, 2):
                            if start[plane+n, row+o, column+p] == label:
                                found = 1
                if start[plane, row, column] != label and found == 1:
                    t, found = 0, 0
                    while t < 9 and found == 0:
                        value = 0
                        for k in range(3):
                            for l in range(3):
                                for m in range(3):
                                    if _P3[t,k,l,m] == 1 and start[plane-1+k, row-1+l, column-1+m] != label:
                                        value += 0
                                    else:
                                        value += 1
                        if value == 0:
                            found = 1
                        t += 1
                    if found == 0:
                        final[plane, row, column] = label
    return start, final

def curvop(start, steps, label, allLabels):
    zsh, ysh, xsh = start.shape
    final = np.copy(start, order='C')
    _P3 = np.zeros((9,3,3,3), dtype=np.int32)
    _P3[0,:,:,1] = 1
    _P3[1,:,1,:] = 1
    _P3[2,1,:,:] = 1
    _P3[3,:,[0,1,2],[0,1,2]] = 1
    _P3[4,:,[0,1,2],[2,1,0]] = 1
    _P3[5,[0,1,2],:,[0,1,2]] = 1
    _P3[6,[0,1,2],:,[2,1,0]] = 1
    _P3[7,[0,1,2],[0,1,2],:] = 1
    _P3[8,[0,1,2],[2,1,0],:] = 1
    for k in range(steps):
        if k % 2 == 0:
            start, final = erosion(start, final, _P3, zsh, ysh, xsh, label)
            start = np.copy(final, order='C')
            final, start = dilation(final, start, _P3, zsh, ysh, xsh, label)
            final = np.copy(start, order='C')
        else:
            start, final = dilation(start, final, _P3, zsh, ysh, xsh, label)
            start = np.copy(final, order='C')
            final, start = erosion(final, start, _P3, zsh, ysh, xsh, label)
            final = np.copy(start, order='C')
    return start

