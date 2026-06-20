import os, sys
import numpy as np
from tifffile import imread
from biomedisa.features.biomedisa_helper import *
import numba

@numba.jit(nopython=True)
def assign_labels(ref, result, labels_matrix):
    zsh, ysh, xsh = ref.shape
    for z in range(zsh):
        for y in range(ysh):
            for x in range(xsh):
                ref_value = ref[z,y,x]
                result_value = result[z,y,x]
                labels_matrix[ref_value,result_value] += 1
    return labels_matrix

def dice_score_(ref, result, ref_val, result_val):
    dice = 2 * np.logical_and(ref==ref_val, result==result_val).sum() / \
    float((ref==ref_val).sum() + (result==result_val).sum())
    return dice

@numba.njit
def dice_score_numba(ref, result, ref_val, result_val):
    intersection = 0
    ref_count = 0
    result_count = 0

    # Assumes ref and result have the same shape
    for i in range(ref.shape[0]):
        for j in range(ref.shape[1]):
            for k in range(ref.shape[2]):
                ref_match = ref[i,j,k] == ref_val
                result_match = result[i,j,k] == result_val
                if ref_match:
                    ref_count += 1
                if result_match:
                    result_count += 1
                if ref_match and result_match:
                    intersection += 1

    denom = ref_count + result_count

    if denom == 0:
        return 1.0  # or 0.0 depending on your convention

    return 2.0 * intersection / denom

if __name__ == "__main__":
    # R: 1000 img slices; 500 labelled slices => 300 training; 100 validation; 100 test
    # H: 1040 img slices; 500 labelled slices => 300 training; 100 validation; 100 test
    BASE = ''
    #=======================================================================================
    # train
    #=======================================================================================
    if '-t' in sys.argv:
        from biomedisa.deeplearning import deep_learning

        # image data
        img = imread(BASE+'EM30-R-im.tif',key=range(0,300))
        img_val = imread(BASE+'EM30-R-im.tif',key=range(300,400))

        # label data
        label = imread(BASE+'EM30-R-labels.tif',key=range(0,300))
        label_val = imread(BASE+'EM30-R-labels.tif',key=range(300,400))

        # image data
        img = np.append(img, imread(BASE+'EM30-H-im.tif',key=range(0,300)), axis=0)
        img_val = np.append(img_val, imread(BASE+'EM30-H-im.tif',key=range(300,400)), axis=0)

        # label data
        label = np.append(label, imread(BASE+'EM30-H-labels.tif',key=range(0,300)), axis=0)
        label_val = np.append(label_val, imread(BASE+'EM30-H-labels.tif',key=range(300,400)), axis=0)

        print(img.shape, img_val.shape)
        print(label.shape, label_val.shape)

        # train separation model
        deep_learning(img, label, train=True, scaling=False, val_dice=False,
            path_to_model=BASE+'EM30-HR-separation-balance.h5', balance=True,
            flip_x=True, flip_y=True, flip_z=True, swapaxes=True,
            val_img_data=img_val, val_label_data=label_val,
            x_patch=16, y_patch=16, z_patch=16, batch_size=48,
            stride_size=8, validation_stride_size=16, separation=True)

        # train mask model
        label[label>0]=1
        label_val[label_val>0]=1
        deep_learning(img, label, train=True, scaling=False, val_dice=True,
            path_to_model=BASE+'EM30-HR-mask.h5', balance=True,
            flip_x=True, flip_y=True, flip_z=True, swapaxes=True,
            val_img_data=img_val, val_label_data=label_val,
            validation_stride_size=64)

    #=======================================================================================
    # mask segmentation
    #=======================================================================================
    if '-m' in sys.argv:
        for sample in ['R','H']:
            path_to_img = BASE+f'EM30-{sample}-im_400-500.tif'
            if not os.path.exist(path_to_img):
                img = imread((BASE+'EM30-R-im.tif', key=range(400,500))
                imwrite(path_to_img, img)
            path_to_model = BASE+'EM30-HR-mask.h5'
            subprocess.Popen([sys.executable, '-m', 'biomedisa.deeplearning',
                path_to_img, path_to_model]).wait()
            result = imread(BASE+f'final.EM30-{sample}-im_400-500.tif')
            test_label = imread(BASE+f'EM30-{sample}-labels.tif', key=range(400,500))
            test_label[test_label>0]=1
            print(sample, dice_score_(result, test_label, 1, 1))

    #=======================================================================================
    # separate mitochondria
    #=======================================================================================
    if '-s' in sys.argv:
        for sample in ['R','H']:
            path_to_img = BASE+f'EM30-{sample}-im_400-500.tif'
            if not os.path.exist(path_to_img):
                img = imread((BASE+'EM30-R-im.tif', key=range(400,500))
                imwrite(path_to_img, img)
            path_to_mask = BASE+f'final.EM30-{sample}-im_400-500.tif'
            path_to_model = BASE+'EM30-HR-separation-balance.h5'
            subprocess.Popen([sys.executable, '-m', 'biomedisa.deeplearning',
                path_to_img, path_to_model, f'-m={path_to_mask}',
                '-ext=.nrrd', '--min_particle_size=100', '-bs=512']).wait()

    #=======================================================================================
    # validate results
    #=======================================================================================
    if '-v' in sys.argv:
        for sample in ['R','H']:

            # path to data
            path_to_ref = BASE+f'EM30-{sample}-labels.tif'
            path_to_result = BASE+f'final.EM30-{sample}-im_400-500.nrrd'

            # load reference
            ref = imread(path_to_ref, key=range(400,500))
            labels, labels_counts = unique(ref, return_counts=True)

            # load result
            result = load_data(path_to_result)[0]
            result_labels = unique(result)

            # assign reference value to the largest overlapping label
            print(np.amax(labels), np.amax(result_labels), len(labels)-1, len(result_labels)-1)
            labels_matrix = np.zeros((np.amax(labels)+1, np.amax(result_labels)+1), np.uint64)
            labels_matrix = assign_labels(ref, result, labels_matrix)

            # Dice score
            sizes = []
            dice_scores = []
            for i, ref_val in enumerate(labels[1:]):
                result_val = np.argmax(labels_matrix[ref_val])
                dice = dice_score_numba(ref, result, ref_val, result_val)
                dice_scores.append(dice)
                sizes.append(labels_counts[i+1])

            # print scores
            print('Weighted Dice score:', np.sum( np.array(sizes) * np.array(dice_scores)) / np.sum(sizes))
            print('Average Dice score:', np.mean(dice_scores))
            print('Less than 80%:', np.sum(np.array(dice_scores)<0.8), 'of', len(labels)-1)

