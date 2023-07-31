#!/usr/bin/python3
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

import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from biomedisa_features.biomedisa_helper import *
from biomedisa_features.remove_outlier import clean
from scipy import ndimage
from pandas import DataFrame
import pandas as pd
import numpy as np
import subprocess
from shutil import move
import argparse
import tarfile
import glob
import scipy

# honeybees scanned upside down
test_inv = ['Head10','Head17','Head27','Head29','Head32','Head36','Head38','Head39','Head41','Head43','Head69','Head71','Head75','Head84','Head95','Head98','S21','S25','S37','S38','S45','S55','S58']
train_inv = ['Head1','Head4','Head7','Head8','Head12','Head19','Head21','Head22','Head57','Head78','Head81']

if __name__ == "__main__":

    # initialize arguments
    parser = argparse.ArgumentParser(description='Bees evaluation.',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # arguments
    parser.add_argument('-hb', '--honeybees', action='store_true', default=False,
                        help='analyse honeybee brains')
    parser.add_argument('-bb', '--bumblebees', action='store_true', default=False,
                        help='analyse bumblebee brains')
    parser.add_argument('-s', '--segmentation', action='store_true', default=False,
                        help='automatic segmentation of bee brains')
    parser.add_argument('-a', '--accuracy', action='store_true', default=False,
                        help='segmentation accuracy')
    parser.add_argument('--assd', action='store_true', default=False,
                        help='calculate ASSD')
    parser.add_argument('-ba', '--brain-areas', action='store_true', default=False,
                        help='calculate brain area volumes')
    parser.add_argument('-st', '--statistics', action='store_true', default=False,
                        help='calculate statistics')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show calculation for each volume')
    parser.add_argument('-nn', '--neural-network', type=str, default=None,
                        help='path to trained network (data will be downloaded from Biomedisa if not specified)')
    parser.add_argument('-ti', '--test-images', type=str, default=None,
                        help='path to test images (data will be downloaded from Biomedisa if not specified)')
    parser.add_argument('-tl', '--test-labels', type=str, default=None,
                        help='path to test labels (data will be downloaded from Biomedisa if not specified)')
    parser.add_argument('-tr', '--test-results', type=str, default=None,
                        help='path to test results (data will be created in segmentation mode or downloaded from Biomedisa if not specified)')
    args = parser.parse_args()

    if not any([args.segmentation, args.accuracy, args.brain_areas, args.statistics]):
        print('Please parse any of "--segmentation", "--accuracy", "--brain-areas", or "--statistics". See "--help" for more information.')
    if not any([args.honeybees, args.bumblebees]):
        print('Please parse any of "--honeybees" or "--bumblebees". See "--help" for more information.')
    elif args.honeybees and args.bumblebees:
        print('Please parse either "--honeybees" or "--bumblebees" but not both of them.')
    elif args.honeybees:
        dataset = 'honeybees'
    elif args.bumblebees:
        dataset = 'bumblebees'

    #=======================================================================================
    # segmentation
    #=======================================================================================

    if args.segmentation:

        # path to data
        path_to_images = os.getcwd()+f'/{dataset}_test_images'
        path_to_results = os.getcwd()+f'/{dataset}_test_results'
        path_to_model = os.getcwd()+f'/{dataset}_network.h5'
        if args.test_images:
            path_to_images = args.test_images
        if args.test_results:
            path_to_results = args.test_results
        if args.neural_network:
            path_to_model = args.neural_network

        # download and extract image data
        if not os.path.isdir(path_to_images):
            os.system(f'wget -nc --no-check-certificate https://biomedisa.org/download/demo/?id={dataset}_test_images.tar -O {path_to_images}.tar')
            tar = tarfile.open(f'{path_to_images}.tar')
            tar.extractall(path=path_to_images)
            tar.close()

        # download trained network
        if not os.path.exists(path_to_model):
            os.system(f'wget -nc --no-check-certificate https://biomedisa.org/download/demo/?id={dataset}_network.h5 -O {path_to_model}')

        # crate directory for results
        if not os.path.exists(path_to_results):
            os.mkdir(path_to_results)

        # segment bee brains
        liste = glob.glob(path_to_images+'/**/*.am', recursive=True)
        for path_to_data in liste:

            if not os.path.exists(path_to_results+'/final.'+os.path.basename(path_to_data)):
                cwd = BASE_DIR+'/demo/'
                p = subprocess.Popen(['python3', 'biomedisa_deeplearning.py', path_to_data, path_to_model, '-p'], cwd=cwd)
                p.wait()
                move(os.path.dirname(path_to_data)+'/final.'+os.path.basename(path_to_data), path_to_results+'/final.'+os.path.basename(path_to_data))

    #=======================================================================================
    # segmentation accuracy
    #=======================================================================================

    if args.accuracy:

        # path to data
        path_to_refs = os.getcwd()+f'/{dataset}_test_labels'
        path_to_results = os.getcwd()+f'/{dataset}_test_results'
        if args.test_labels:
            path_to_refs = args.test_labels
        if args.test_results:
            path_to_results = args.test_results

        # download and extract reference data
        if not os.path.isdir(path_to_refs):
            os.system(f'wget -nc --no-check-certificate https://biomedisa.org/download/demo/?id={dataset}_test_labels.tar -O {path_to_refs}.tar')
            tar = tarfile.open(f'{path_to_refs}.tar')
            tar.extractall(path=path_to_refs)
            tar.close()

        # download and extract segmentation results (if not created)
        if not os.path.isdir(path_to_results):
            os.system(f'wget -nc --no-check-certificate https://biomedisa.org/download/demo/?id={dataset}_test_results.tar -O {path_to_results}.tar')
            tar = tarfile.open(f'{path_to_results}.tar')
            tar.extractall(path=path_to_results)
            tar.close()

       # calculate ASSD
        if args.assd:
            try:
                from biomedisa_features.assd import ASSD_one_label
            except:
                print('Error: no CUDA device found. ASSD is not available.')
                args.assd = False

        # path to results
        liste = glob.glob(path_to_results+'/**/*.am', recursive=True)

        # initialization
        D = np.zeros((len(liste),7))
        A = np.zeros((len(liste),7))
        liste = sorted(liste)

        # iterate over results
        for n, path_to_data in enumerate(liste):

            # path to reference
            sample = os.path.basename(path_to_data).replace('final.','').replace('.am','')
            if args.honeybees:
                path_to_ref = path_to_refs+'/Clean.'+sample+'.am'
            elif args.bumblebees:
                path_to_ref = path_to_refs+'/'+sample+'_OK.am'

            if os.path.exists(path_to_ref):

                # load data
                a, _ = load_data(path_to_ref)
                b, _ = load_data(path_to_data)

                # remove outliers
                c = np.copy(b)
                c = clean(c, 0.1)
                c[b==3] = 3     # CX is not cleaned
                b = np.copy(c)

                # total Dice score
                D[n,-1] = Dice_score(a,b)
                all_dist = 0
                all_elements = 0

                # iterate over labels
                for j,i in enumerate([2,3,4,5,6,7]):
                    d = np.zeros_like(a)
                    e = np.zeros_like(b)
                    d[a==i] = 1
                    e[b==i] = 1

                    # Dice score of each label
                    D[n,j] = Dice_score(d,e)

                    # ASSD of each label
                    if args.assd:
                        distances, number_of_elements, _ = ASSD_one_label(a,b,i)
                        all_dist += distances
                        all_elements += number_of_elements
                        A[n,j] = distances / float(number_of_elements)

                # total ASSD
                if args.assd:
                    A[n,-1] = all_dist / float(all_elements)

                # print accuracy for each brain
                if args.verbose:
                    print(sample, 'Dice:', D[n], 'ASSD:', A[n])

        print('Number of images:', len(liste))
        print('Dice:', np.mean(D, axis=0))
        print('ASSD:', np.mean(A, axis=0))

    #=======================================================================================
    # bee brain area volumes
    #=======================================================================================

    if args.brain_areas:

        path_to_volumes = os.getcwd()+f'/{dataset}_brain_areas.txt'
        training_images = os.getcwd()+f'/{dataset}_training_images'
        training_labels = os.getcwd()+f'/{dataset}_training_labels'
        test_images = os.getcwd()+f'/{dataset}_test_images'
        test_labels = os.getcwd()+f'/{dataset}_test_labels'

        # download and extract data
        if not os.path.isdir(training_images):
            os.system(f'wget -nc --no-check-certificate https://biomedisa.org/download/demo/?id={dataset}_training_images.tar -O {training_images}.tar')
            tar = tarfile.open(f'{training_images}.tar')
            tar.extractall(path=training_images)
            tar.close()
        if not os.path.isdir(training_labels):
            os.system(f'wget -nc --no-check-certificate https://biomedisa.org/download/demo/?id={dataset}_training_labels.tar -O {training_labels}.tar')
            tar = tarfile.open(f'{training_labels}.tar')
            tar.extractall(path=training_labels)
            tar.close()
        if not os.path.isdir(test_images):
            os.system(f'wget -nc --no-check-certificate https://biomedisa.org/download/demo/?id={dataset}_test_images.tar -O {test_images}.tar')
            tar = tarfile.open(f'{test_images}.tar')
            tar.extractall(path=test_images)
            tar.close()
        if not os.path.isdir(test_labels):
            os.system(f'wget -nc --no-check-certificate https://biomedisa.org/download/demo/?id={dataset}_test_labels.tar -O {test_labels}.tar')
            tar = tarfile.open(f'{test_labels}.tar')
            tar.extractall(path=test_labels)
            tar.close()

        with open(path_to_volumes, 'w') as file:

            df_train = glob.glob(training_labels+'/**/*.am', recursive=True)
            df_test = glob.glob(test_labels+'/**/*.am', recursive=True)
            df = sorted(df_train + df_test)

            for iterator, path in enumerate(df):

                # load data
                image, img_header = load_data(path)

                # honeybees
                if args.honeybees:

                    # get sample
                    sample = os.path.basename(path).replace('.labels.am','').replace('Clean.','').replace('.am','')

                    # get id
                    if 'Head' in sample:
                        id = sample.replace('Head','')
                    else:
                        id = sample.replace('S','')
                        id = 1000+int(id)

                    # load header from image data (because voxel spacing was not originally transfered)
                    if path in df_train:
                        _, img_header = load_data(training_images+'/'+sample+'.am')
                    else:
                        _, img_header = load_data(test_images+'/'+sample+'.am')

                    # align honeybee data if scanned upside down
                    if args.honeybees and (sample in test_inv or sample in train_inv):
                        image = np.copy(image[::-1,:,::-1])

                # bumblebees
                if args.bumblebees:

                    # get id
                    id = os.path.basename(path).replace('C_OK.am','')
                    id = id[1:]

                # read img_header as string
                b = img_header[0].tobytes()
                s = b.decode("utf-8")

                # get physical size from image header
                lattice = re.search('BoundingBox (.*),\n', s)
                lattice = lattice.group(1)
                i0, i1, i2, i3, i4, i5 = lattice.split(' ')
                #bounding_box_i = re.search('&BoundingBox (.*),\n', s)
                #bounding_box_i = bounding_box_i.group(1)

                # voxel resolution
                zsh, ysh, xsh = image.shape
                zsh -= 1
                ysh -= 1
                xsh -= 1
                v_res = (float(i5)-float(i4))*(float(i3)-float(i2))*(float(i1)-float(i0)) / (zsh*ysh*xsh)
                xres = (float(i1)-float(i0)) / xsh
                yres = (float(i3)-float(i2)) / ysh
                zres = (float(i5)-float(i4)) / zsh
                #string = str(id)+','+str(xsh+1)+','+str(ysh+1)+','+str(zsh+1)+','+str(xres)+','+str(yres)+','+str(zres)
                string = str(id)
                if args.verbose:
                    print(string, v_res, xsh+1, ysh+1, zsh+1, xres, yres, zres)

                mask = np.empty_like(image)
                s = [[[0,0,0], [0,1,0], [0,0,0]], [[0,1,0], [1,1,1], [0,1,0]], [[0,0,0], [0,1,0], [0,0,0]]]

                # loop over all neuropils
                for j, k in enumerate([2,3,4,5,6,7]):
                    label_size = np.sum(image==k)
                    string += ','+str(label_size*v_res)

                # total brain volume
                label_size = np.sum(image>0)
                string += ','+str(label_size*v_res)

                # loop over paired areas
                for j, k in enumerate([2,4,5,7]):

                    # get mask
                    mask.fill(0)
                    mask[image==k] = 1

                    # get objects
                    labeled_array, _ = ndimage.label(mask, structure=s)
                    size = np.bincount(labeled_array.ravel())

                    # remove background
                    size = size[1:]

                    # first label
                    first_label = np.argmax(size)
                    first_label_size = size[first_label]

                    if size.size > 1:

                        # remove biggest label
                        size[first_label] = 0

                        # second label
                        second_label = np.argmax(size)
                        second_label_size = size[second_label]

                        if second_label_size > 0.5*first_label_size:

                            # left or right side
                            mask.fill(0)
                            mask[labeled_array==first_label+1] = 1
                            mask[labeled_array==second_label+1] = 2
                            for x in range(mask.shape[2]):
                                if np.any(mask[:,:,x]==1):
                                    string += ','+str(first_label_size*v_res)+','+str(second_label_size*v_res)
                                    break
                                elif np.any(mask[:,:,x]==2):
                                    string += ','+str(second_label_size*v_res)+','+str(first_label_size*v_res)
                                    break

                        else:
                            string += ','+str(-1)+','+str(-1)
                    else:
                        string += ','+str(-1)+','+str(-1)
                print(string, file=file)

        # TXT to sorted XLSX
        l = np.loadtxt(path_to_volumes, delimiter=',', dtype='str')
        ysh, xsh = l.shape
        for i in range(1,xsh):
            t = []
            for k in range(ysh):
                t.append((int(l[k,0]),float(l[k,i])))
            t = sorted(t, key=lambda x: x[0])
            for k in range(ysh):
                l[k,i] = t[k][1]
        for k in range(ysh):
            l[k,0] = t[k][0]

        # convert to float and save -1 as Nan
        l = l.astype(float)
        l[l<0] = np.nan

        # save data as excel
        df = DataFrame({'BeeID': l[:,0].astype(int), 'MB': l[:,1], 'CX': l[:,2], 'AL': l[:,3],
                        'ME': l[:,4], 'OTH': l[:,5], 'LO': l[:,6], 'OL': l[:,4]+l[:,6],
                        'Brain': l[:,7],
                        'MB_Right': l[:,8], 'MB_Left': l[:,9],
                        'AL_Right': l[:,10], 'AL_Left': l[:,11],
                        'ME_Right': l[:,12], 'ME_Left': l[:,13],
                        'LO_Right': l[:,14], 'LO_Left': l[:,15],
                        'OL_Right': l[:,12]+l[:,14], 'OL_Left': l[:,13]+l[:,15]
                        })
        df.to_excel(path_to_volumes.replace('.txt','.xlsx'), sheet_name='sheet1', index=False)

    #=======================================================================================
    # statistics
    #=======================================================================================

    if args.statistics:

        path_to_volumes = os.getcwd()+f'/{dataset}_brain_areas.xlsx'
        df = pd.read_excel(path_to_volumes, engine='openpyxl', sheet_name='sheet1')

        # variations
        for key in df.keys()[1:]:
            a = np.array(df[key]).astype(float)
            a = a[a>0]
            print('=============')
            print(key+' Statistics:')
            print('=============')
            print('mean:', np.mean(a))
            print('std:', np.std(a))
            print('min:', min(a))
            print('max:', max(a))
            print('variation:', (max(a) - min(a)) / max(a))
            print('\n')

        # asymmetry
        for key1, key2 in [('MB_Left','MB_Right'),('AL_Left','AL_Right'),('ME_Left','ME_Right'),('LO_Left','LO_Right'),('OL_Left','OL_Right')]:
            print('=============')
            print(key1[:2]+' Asymmetry:')
            print('=============')
            a = np.array(df[key1]).astype(float)
            b = np.array(df[key2]).astype(float)
            c = np.array(df['Brain']).astype(float)
            c = c[a>0]
            b = b[a>0]
            a = a[a>0]
            print('(Total)', f'N={a.size}')
            print('(Left < right)', f'N={np.sum(a<b)}', f'{round(100 / a.size * np.sum(a<b))}%')
            print('(Absolute Volume)', f'p-value={round(scipy.stats.ttest_rel(np.ravel(a), np.ravel(b)).pvalue,4)}')
            print('(Relative Volume)', f'p-value={round(scipy.stats.ttest_1samp(np.ravel(a-b)/np.ravel(c), 0).pvalue,4)}')
            print('\n')

        # correlation right-left differences
        for i, tuple in enumerate([('MB_Left','MB_Right'),('AL_Left','AL_Right'),('OL_Left','OL_Right')]):
            key3, key4 = tuple
            for j, tuple in enumerate([('MB_Left','MB_Right'),('AL_Left','AL_Right'),('OL_Left','OL_Right')]):
                key1, key2 = tuple
                if j>i:
                    print('=============')
                    print(key3[:2], '/', key1[:2])
                    print('=============')
                    l1 = np.array(df[key1]).astype(float)
                    r1 = np.array(df[key2]).astype(float)
                    l2 = np.array(df[key3]).astype(float)
                    r2 = np.array(df[key4]).astype(float)
                    total = np.array(df['Brain']).astype(float)
                    nas = np.logical_and(np.logical_and(l1>0, l2>0), np.logical_and(r1>0, r2>0))
                    l1 = l1[nas]
                    r1 = r1[nas]
                    l2 = l2[nas]
                    r2 = r2[nas]
                    total = total[nas]
                    print(f'Correlation ({key4})-({key3}) and ({key2})-({key1})', scipy.stats.pearsonr((r2-l2), (r1-l1)))
                    summe = np.sum(np.logical_and(l2>r2, l1>r1))
                    print(f'larger {key3} and larger {key1}:', f'N={summe},', round(100* summe / np.sum(nas),1), '%')
                    summe = np.sum(np.logical_and(l2>r2, l1<r1))
                    print(f'larger {key3} and larger {key2}:', f'N={summe},', round(100* summe / np.sum(nas),1), '%')
                    summe = np.sum(np.logical_and(l2<r2, l1>r1))
                    print(f'larger {key4} and larger {key1}:', f'N={summe},', round(100* summe / np.sum(nas),1), '%')
                    summe = np.sum(np.logical_and(l2<r2, l1<r1))
                    print(f'larger {key4} and larger {key2}:', f'N={summe},', round(100* summe / np.sum(nas),1), '%')
                    print('\n')

        # correlations
        labels = ['MB','CX','AL','ME','OTH','LO','OL']

        # between neuropil volumes
        '''for i, key1 in enumerate(labels):
            for j, key2 in enumerate(labels):
                if j > i:
                    a = np.array(df[key1]).astype(float)
                    b = np.array(df[key2]).astype(float)
                    c = np.array(df['Brain']).astype(float)
                    c = c[a>0]
                    a = a[a>0]
                    b = b[b>0]
                    print(key1, '/', key2)
                    print('----------------------------')
                    #print('abs:', scipy.stats.pearsonr(a, b))
                    print('rel:', scipy.stats.pearsonr(a/c, b/c))
                    print('')'''

        # between neuropil volumes and total brain volume
        '''for key1 in labels:
            a = np.array(df[key1]).astype(float)
            b = np.array(df['Brain']).astype(float)
            b = b[a>0]
            a = a[a>0]
            print(key1, '/', 'Brain')
            print('----------------------------')
            print('abs:', scipy.stats.pearsonr(a, b))
            print('rel:', scipy.stats.pearsonr(a/b, b))
            print('')'''

